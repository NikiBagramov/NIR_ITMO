from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .evaluate import (
    best_f1_threshold,
    collect_labels_scores_gaps,
    collect_labels_scores_tokens,
    metrics_from_scores,
)
from .fasttext_embed import train_fasttext
from .features import gap_level_features, token_level_features
from .modeling import extract_feature_matrix, feature_weights, train_token_classifier
from .ngram_lm import train_bidirectional_lm
from .preprocess import Sentence, load_corpus, save_jsonl
from .reporting import build_top_examples_markdown
from .suggest import top_frequent_vocab
from .synthetic import CorruptedSentence, corrupt_corpus, to_jsonable


def _split_train_test(items: Sequence[Sentence], train_frac: float, seed: int) -> tuple[List[Sentence], List[Sentence]]:
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(len(items) * train_frac)
    train = [items[i] for i in idx[:cut]]
    test = [items[i] for i in idx[cut:]]
    return train, test


def _save_corrupted_jsonl(corpus: Sequence[CorruptedSentence], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cs in corpus:
            f.write(json.dumps(to_jsonable(cs), ensure_ascii=False) + "\n")


def _plot_pr_curve(pr: List[float], rc: List[float], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(rc, pr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _add_labels_token_rows(corrupted: Sequence[CorruptedSentence], token_rows: List[Dict]) -> None:
    label_by_sid: Dict[str, List[int]] = {}
    for cs in corrupted:
        labels = [0] * len(cs.tokens_corrupted)
        for s in cs.substitutions:
            if 0 <= s.pos < len(labels):
                labels[s.pos] = 1
        label_by_sid[cs.id] = labels

    for r in token_rows:
        sid = r["sent_id"]
        idx = int(r["token_idx"])
        labs = label_by_sid.get(sid)
        r["label"] = int(labs[idx]) if labs is not None and 0 <= idx < len(labs) else 0


def _add_labels_gap_rows(corrupted: Sequence[CorruptedSentence], gap_rows: List[Dict]) -> None:
    label_by_sid: Dict[str, List[int]] = {}
    for cs in corrupted:
        n = len(cs.tokens_corrupted)
        labels = [0] * max(0, n - 1)
        for d in cs.deletions:
            if d.left is None or d.right is None:
                continue
            for i in range(n - 1):
                if cs.tokens_corrupted[i] == d.left and cs.tokens_corrupted[i + 1] == d.right:
                    labels[i] = 1
                    break
        label_by_sid[cs.id] = labels

    for r in gap_rows:
        sid = r["sent_id"]
        idx = int(r["gap_idx"])
        labs = label_by_sid.get(sid)
        r["label"] = int(labs[idx]) if labs is not None and 0 <= idx < len(labs) else 0


def _correction_topk_metrics(
    corrupted: Sequence[CorruptedSentence],
    *,
    fwd_lm,
    candidate_vocab: Sequence[str],
    top_k: int = 5,
) -> Dict[str, float]:
    """Оценка качества подсказок (вставка/замена) на синтетических ошибках."""
    from .suggest import suggest_insertion, suggest_replacements

    subs_total = 0
    subs_top1 = 0
    subs_topk = 0

    dels_total = 0
    dels_top1 = 0
    dels_topk = 0

    for cs in corrupted:
        toks = cs.tokens_corrupted

        # substitutions: предсказать исходное слово
        for s in cs.substitutions:
            subs_total += 1
            sugg = suggest_replacements(toks, s.pos, fwd_lm, candidate_vocab, top_k=top_k)
            words = [w for w, _ in sugg]
            if not words:
                continue
            if words[0] == s.orig:
                subs_top1 += 1
            if s.orig in words[:top_k]:
                subs_topk += 1

        # deletions: предсказать удалённое слово
        for d in cs.deletions:
            if d.left is None or d.right is None:
                continue
            # найти gap_idx
            gidx = None
            for i in range(len(toks) - 1):
                if toks[i] == d.left and toks[i + 1] == d.right:
                    gidx = i
                    break
            if gidx is None:
                continue
            dels_total += 1
            sugg = suggest_insertion(toks, gidx, fwd_lm, candidate_vocab, top_k=top_k)
            words = [w for w, _ in sugg]
            if not words:
                continue
            if words[0] == d.token:
                dels_top1 += 1
            if d.token in words[:top_k]:
                dels_topk += 1

    return {
        "subs_total": float(subs_total),
        "subs_top1": float(subs_top1 / subs_total) if subs_total else 0.0,
        "subs_topk": float(subs_topk / subs_total) if subs_total else 0.0,
        "dels_total": float(dels_total),
        "dels_top1": float(dels_top1 / dels_total) if dels_total else 0.0,
        "dels_topk": float(dels_topk / dels_total) if dels_total else 0.0,
        "top_k": float(top_k),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_docx", nargs="*", default=[], help="Пути к .docx")
    ap.add_argument("--input_txt", nargs="*", default=[], help="Пути к .txt")
    ap.add_argument("--artifacts_dir", default="artifacts", help="Папка для артефактов")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--min_tokens", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=80)

    ap.add_argument("--p_delete", type=float, default=0.15)
    ap.add_argument("--p_substitute", type=float, default=0.15)
    ap.add_argument("--max_events", type=int, default=2)
    ap.add_argument("--prefer_confusable", action="store_true")

    ap.add_argument("--lm_alpha", type=float, default=0.1)

    ap.add_argument("--ft_dim", type=int, default=100)
    ap.add_argument("--ft_window", type=int, default=5)
    ap.add_argument("--ft_bucket", type=int, default=50000)
    ap.add_argument("--ft_epochs", type=int, default=40)

    ap.add_argument("--top_vocab_n", type=int, default=3000)
    ap.add_argument("--top_examples", type=int, default=15)

    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    # 1) Load corpus
    corpus = load_corpus(
        input_docx=args.input_docx,
        input_txt=args.input_txt,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )
    save_jsonl(corpus, artifacts / "corpus_sentences.jsonl")

    if len(corpus) < 50:
        print(f"[warn] corpus is small: {len(corpus)} sentences. Results will be noisy.")

    # 2) split
    train_s, test_s = _split_train_test(corpus, train_frac=args.train_frac, seed=args.seed)
    save_jsonl(train_s, artifacts / "train_sentences.jsonl")
    save_jsonl(test_s, artifacts / "test_sentences.jsonl")

    # 3) train models on clean train
    train_tokens = [s.tokens for s in train_s]
    fwd, bwd = train_bidirectional_lm(train_tokens, alpha=args.lm_alpha)

    with (artifacts / "ngram_lm_fwd.pkl").open("wb") as f:
        pickle.dump(fwd, f)
    with (artifacts / "ngram_lm_bwd.pkl").open("wb") as f:
        pickle.dump(bwd, f)

    ft = train_fasttext(
        train_tokens,
        vector_size=args.ft_dim,
        window=args.ft_window,
        bucket=args.ft_bucket,
        epochs=args.ft_epochs,
        workers=os.cpu_count() or 4,
    )
    ft.save(str(artifacts / "fasttext.model"))

    # vocab for corruption + suggestions
    vocab_full = [w for w in fwd.vocab if w not in ("<s>", "</s>")]
    top_vocab = top_frequent_vocab(fwd, top_n=args.top_vocab_n)

    # 4) corrupt corpora (synthetic labels)
    train_pairs = [(s.id, s.tokens) for s in train_s]
    test_pairs = [(s.id, s.tokens) for s in test_s]

    train_cor = corrupt_corpus(
        train_pairs,
        vocab_full,
        seed=args.seed + 1,
        p_delete=args.p_delete,
        p_substitute=args.p_substitute,
        max_events=args.max_events,
        prefer_confusable=args.prefer_confusable,
    )
    test_cor = corrupt_corpus(
        test_pairs,
        vocab_full,
        seed=args.seed + 2,
        p_delete=args.p_delete,
        p_substitute=args.p_substitute,
        max_events=args.max_events,
        prefer_confusable=args.prefer_confusable,
    )

    _save_corrupted_jsonl(train_cor, artifacts / "train_corrupted.jsonl")
    _save_corrupted_jsonl(test_cor, artifacts / "test_corrupted.jsonl")

    # 5) feature extraction
    token_rows_train: List[Dict] = []
    gap_rows_train: List[Dict] = []
    sent_tokens_train: Dict[str, List[str]] = {}

    for cs in tqdm(train_cor, desc="features(train)"):
        sid = cs.id
        toks = cs.tokens_corrupted
        sent_tokens_train[sid] = toks
        token_rows_train.extend(token_level_features(sid, toks, fwd=fwd, bwd=bwd, fasttext_model=ft))
        gap_rows_train.extend(gap_level_features(sid, toks, fwd=fwd, bwd=bwd))

    _add_labels_token_rows(train_cor, token_rows_train)
    _add_labels_gap_rows(train_cor, gap_rows_train)

    token_rows_test: List[Dict] = []
    gap_rows_test: List[Dict] = []
    sent_tokens_test: Dict[str, List[str]] = {}

    for cs in tqdm(test_cor, desc="features(test)"):
        sid = cs.id
        toks = cs.tokens_corrupted
        sent_tokens_test[sid] = toks
        token_rows_test.extend(token_level_features(sid, toks, fwd=fwd, bwd=bwd, fasttext_model=ft))
        gap_rows_test.extend(gap_level_features(sid, toks, fwd=fwd, bwd=bwd))

    _add_labels_token_rows(test_cor, token_rows_test)
    _add_labels_gap_rows(test_cor, gap_rows_test)

    # 6) train classifier on synthetic train
    X_train = extract_feature_matrix(token_rows_train)
    y_train = np.asarray([r["label"] for r in token_rows_train], dtype=int)

    clf = train_token_classifier(X_train, y_train)
    with (artifacts / "token_classifier.pkl").open("wb") as f:
        pickle.dump(clf, f)

    weights = feature_weights(clf)

    # 7) predict on train/test
    proba_train = clf.predict_proba(X_train)[:, 1]
    for r, p in zip(token_rows_train, proba_train):
        r["lr_score"] = float(p)

    X_test = extract_feature_matrix(token_rows_test)
    y_test = np.asarray([r["label"] for r in token_rows_test], dtype=int)
    proba_test = clf.predict_proba(X_test)[:, 1]
    for r, p in zip(token_rows_test, proba_test):
        r["lr_score"] = float(p)

    # 8) evaluate: token
    token_metrics: Dict[str, Dict] = {}

    # threshold chosen on train (to avoid peeking at test)
    thr, best_f1_tr = best_f1_threshold(y_train.tolist(), proba_train.tolist()) if y_train.any() else (0.5, 0.0)

    for field in ["ngram_surp", "ft_outlier", "lr_score"]:
        y_true, y_score = collect_labels_scores_tokens(test_cor, token_rows_test, score_field=field)
        token_metrics[field] = metrics_from_scores(y_true, y_score, threshold=thr if field == "lr_score" else 0.5)

    # 9) evaluate: gaps
    gap_metrics: Dict[str, Dict] = {}
    y_true_g, y_score_g = collect_labels_scores_gaps(test_cor, gap_rows_test, score_field="gap_score")
    gap_metrics["gap_score"] = metrics_from_scores(y_true_g, y_score_g, threshold=0.5)

    # 10) correction quality (top-k)
    correction = _correction_topk_metrics(test_cor, fwd_lm=fwd, candidate_vocab=top_vocab, top_k=5)

    # 11) Save tables
    df_tok = pd.DataFrame(token_rows_test)
    df_tok.to_csv(artifacts / "scores_tokens.csv", index=False)

    df_gap = pd.DataFrame(gap_rows_test)
    df_gap.to_csv(artifacts / "scores_gaps.csv", index=False)

    # 12) PR curve plot (token lr)
    pr = token_metrics["lr_score"]["pr_curve"]["precision"]
    rc = token_metrics["lr_score"]["pr_curve"]["recall"]
    _plot_pr_curve(pr, rc, artifacts / "pr_curve.png", title="Token anomalies PR curve (LR)")

    # 13) Top examples markdown
    md = build_top_examples_markdown(
        token_rows=token_rows_test,
        gap_rows=gap_rows_test,
        sentence_tokens=sent_tokens_test,
        lm=fwd,
        candidate_vocab=top_vocab,
        token_score_field="lr_score",
        gap_score_field="gap_score",
        top_n=args.top_examples,
    )
    (artifacts / "top_examples.md").write_text(md, encoding="utf-8")

    # 14) Metrics json
    metrics = {
        "corpus": {"sentences": len(corpus), "train": len(train_s), "test": len(test_s)},
        "params": {
            "seed": args.seed,
            "train_frac": args.train_frac,
            "p_delete": args.p_delete,
            "p_substitute": args.p_substitute,
            "max_events": args.max_events,
            "prefer_confusable": bool(args.prefer_confusable),
            "lm_alpha": args.lm_alpha,
            "ft_dim": args.ft_dim,
            "ft_window": args.ft_window,
            "ft_bucket": args.ft_bucket,
            "ft_epochs": args.ft_epochs,
            "top_vocab_n": args.top_vocab_n,
        },
        "token_metrics": token_metrics,
        "gap_metrics": gap_metrics,
        "classifier_weights": weights,
        "threshold_lr": float(thr),
        "correction": correction,
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved to:", artifacts.resolve())
    print("Token metrics:")
    print(json.dumps(token_metrics, ensure_ascii=False, indent=2)[:2000], "...")
    print("Gap metrics:")
    print(json.dumps(gap_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
