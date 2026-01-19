from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)

from .synthetic import CorruptedSentence


def build_token_labels(cs: CorruptedSentence) -> List[int]:
    labels = [0] * len(cs.tokens_corrupted)
    for s in cs.substitutions:
        if 0 <= s.pos < len(labels):
            labels[s.pos] = 1
    return labels


def build_gap_labels(cs: CorruptedSentence) -> List[int]:
    """Метки для промежутков между словами.

    Возвращает список длины len(tokens_corrupted)-1, где 1 означает,
    что эта пара (left,right) образовалась из-за удаления одного слова.

    При нескольких одинаковых парах внутри предложения выбираем первое совпадение.
    """
    n = len(cs.tokens_corrupted)
    if n < 2:
        return []

    labels = [0] * (n - 1)

    for d in cs.deletions:
        if d.left is None or d.right is None:
            continue
        left = d.left
        right = d.right
        # ищем место, где left и right стали соседями
        for i in range(n - 1):
            if cs.tokens_corrupted[i] == left and cs.tokens_corrupted[i + 1] == right:
                labels[i] = 1
                break

    return labels


def prf_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "threshold": float(threshold)}


def metrics_from_scores(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> Dict:
    y_true_a = np.asarray(y_true, dtype=int)
    y_score_a = np.asarray(y_score, dtype=float)

    ap = float(average_precision_score(y_true_a, y_score_a)) if y_true_a.any() else 0.0
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_a,
        (y_score_a >= threshold).astype(int),
        average="binary",
        zero_division=0,
    )
    pr, rc, thr = precision_recall_curve(y_true_a, y_score_a)

    return {
        "average_precision": ap,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "threshold": float(threshold),
        "pr_curve": {
            "precision": pr.tolist(),
            "recall": rc.tolist(),
            "thresholds": thr.tolist(),
        },
    }


def best_f1_threshold(y_true: Sequence[int], y_score: Sequence[float]) -> Tuple[float, float]:
    """Подбирает порог по максимуму F1 на PR‑кривой."""
    y_true_a = np.asarray(y_true, dtype=int)
    y_score_a = np.asarray(y_score, dtype=float)
    pr, rc, thr = precision_recall_curve(y_true_a, y_score_a)

    # pr,rc длиннее thr на 1
    best_t = 0.5
    best_f1 = -1.0
    for i, t in enumerate(thr):
        p = pr[i + 1]
        r = rc[i + 1]
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def collect_labels_scores_tokens(
    corrupted: Sequence[CorruptedSentence],
    token_rows: Sequence[Dict],
    *,
    score_field: str,
) -> Tuple[List[int], List[float]]:
    """Склеивает token labels и scores в один массив."""
    # token_rows предполагаются в том же порядке, что и предложения,
    # и каждая строка содержит sent_id/token_idx.
    # Проще: создадим мапку sentence->labels.

    labels_by_sid = {cs.id: build_token_labels(cs) for cs in corrupted}

    y_true: List[int] = []
    y_score: List[float] = []
    for r in token_rows:
        sid = r["sent_id"]
        idx = int(r["token_idx"])
        lab_list = labels_by_sid.get(sid)
        if lab_list is None:
            continue
        if 0 <= idx < len(lab_list):
            y_true.append(int(lab_list[idx]))
            y_score.append(float(r[score_field]))
    return y_true, y_score


def collect_labels_scores_gaps(
    corrupted: Sequence[CorruptedSentence],
    gap_rows: Sequence[Dict],
    *,
    score_field: str = "gap_score",
) -> Tuple[List[int], List[float]]:
    labels_by_sid = {cs.id: build_gap_labels(cs) for cs in corrupted}

    y_true: List[int] = []
    y_score: List[float] = []
    for r in gap_rows:
        sid = r["sent_id"]
        idx = int(r["gap_idx"])
        lab_list = labels_by_sid.get(sid)
        if lab_list is None:
            continue
        if 0 <= idx < len(lab_list):
            y_true.append(int(lab_list[idx]))
            y_score.append(float(r[score_field]))
    return y_true, y_score
