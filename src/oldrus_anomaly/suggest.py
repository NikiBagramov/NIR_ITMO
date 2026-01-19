from __future__ import annotations

import heapq
from typing import List, Sequence, Tuple

from .ngram_lm import BOS, EOS, NgramLM


def top_frequent_vocab(lm: NgramLM, *, top_n: int = 5000) -> List[str]:
    """Список наиболее частотных слов из LM (без BOS/EOS)."""
    c = lm.counts
    if c is None:
        raise RuntimeError("LM не обучена")
    items = [(cnt, w) for w, cnt in c.unigram.items() if w not in (BOS, EOS)]
    items.sort(reverse=True)
    return [w for _, w in items[:top_n]]


def _context_left(tokens: Sequence[str], idx: int) -> Tuple[str, str]:
    prev1 = tokens[idx - 1] if idx - 1 >= 0 else BOS
    prev2 = tokens[idx - 2] if idx - 2 >= 0 else BOS
    return prev2, prev1


def suggest_replacements(
    tokens: Sequence[str],
    idx: int,
    lm: NgramLM,
    candidates: Sequence[str],
    *,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Предлагает замены слова tokens[idx].

    Скор = logP(cand | left2,left1) + logP(right1 | left1,cand)

    Это простая, но полезная эвристика для практики и отчёта.
    """
    if not (0 <= idx < len(tokens)):
        return []

    left2, left1 = _context_left(tokens, idx)
    right1 = tokens[idx + 1] if idx + 1 < len(tokens) else EOS

    # max-heap через отрицательный ключ
    heap: List[Tuple[float, str]] = []
    for cand in candidates:
        if cand == tokens[idx]:
            continue
        sc = lm.logprob(cand, (left2, left1)) + lm.logprob(right1, (left1, cand))
        if len(heap) < top_k:
            heapq.heappush(heap, (sc, cand))
        else:
            if sc > heap[0][0]:
                heapq.heapreplace(heap, (sc, cand))

    best = sorted(heap, reverse=True)
    return [(cand, float(sc)) for sc, cand in best]


def suggest_insertion(
    tokens: Sequence[str],
    gap_idx: int,
    lm: NgramLM,
    candidates: Sequence[str],
    *,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Предлагает слово для вставки в gap между tokens[gap_idx] и tokens[gap_idx+1]."""
    if gap_idx < 0 or gap_idx >= len(tokens) - 1:
        return []

    # gap между left=tokens[gap_idx] и right=tokens[gap_idx+1]
    left = tokens[gap_idx]
    right = tokens[gap_idx + 1]
    left2 = tokens[gap_idx - 1] if gap_idx - 1 >= 0 else BOS

    heap: List[Tuple[float, str]] = []
    for cand in candidates:
        sc = lm.logprob(cand, (left2, left)) + lm.logprob(right, (left, cand))
        if len(heap) < top_k:
            heapq.heappush(heap, (sc, cand))
        else:
            if sc > heap[0][0]:
                heapq.heapreplace(heap, (sc, cand))

    best = sorted(heap, reverse=True)
    return [(cand, float(sc)) for sc, cand in best]
