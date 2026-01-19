"""Небольшая демонстрация MLM‑скоринга (BERT) для древнерусских токенов.

Запуск:

  pip install transformers tokenizers accelerate
  python bert_demo.py --input_jsonl artifacts/corpus_sentences.jsonl --model npedrazzini/BERTislav --n 5

Скрипт:
- берет первые N предложений из jsonl
- считает word-level NLL (masked LM pseudo log-likelihood)
- выводит топ подозрительных слов и top-k подсказки

Примечание: это диагностический скрипт, не часть основного pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oldrus_anomaly.preprocess import load_jsonl  # noqa: E402
from oldrus_anomaly.bert_mlm import load_mlm, pll_word_nll, topk_word_replacements  # noqa: E402
from oldrus_anomaly.reporting import render_sentence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--model", default="npedrazzini/BERTislav")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--top_words", type=int, default=5)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    sents = load_jsonl(args.input_jsonl)
    sents = sents[: args.n]

    tok, model = load_mlm(args.model, device=args.device)

    for s in sents:
        res = pll_word_nll(s.tokens, tokenizer=tok, model=model, device=args.device)
        pairs = list(enumerate(res.word_nll_norm))
        pairs.sort(key=lambda x: x[1], reverse=True)

        print("\n===", s.id, "===")
        print(render_sentence(s.tokens))

        for rank, (idx, sc) in enumerate(pairs[: args.top_words], start=1):
            print(f"  {rank}. idx={idx} token='{s.tokens[idx]}' nll_norm={sc:.3f}")
            repl = topk_word_replacements(s.tokens, idx, tokenizer=tok, model=model, device=args.device, top_k=args.topk)
            if repl:
                print("     topk:", ", ".join([f"{w}({lp:.2f})" for w, lp in repl]))


if __name__ == "__main__":
    main()
