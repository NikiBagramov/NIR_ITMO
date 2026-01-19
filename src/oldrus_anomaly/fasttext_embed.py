from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# --- Compatibility patch -------------------------------------------------
# Gensim 4.x ожидает scipy.linalg.triu/tril, но в некоторых сборках SciPy
# (например, 1.14+) эти символы могут отсутствовать.
# Подкладываем аналоги из numpy, чтобы не падало при импорте gensim.
try:
    import scipy.linalg  # type: ignore

    if not hasattr(scipy.linalg, "triu"):
        scipy.linalg.triu = np.triu  # type: ignore
    if not hasattr(scipy.linalg, "tril"):
        scipy.linalg.tril = np.tril  # type: ignore
except Exception:
    # Если SciPy вовсе не установлен, gensim может работать в урезанном режиме.
    pass

from gensim.models import FastText


def train_fasttext(
    sentences: Iterable[Sequence[str]],
    *,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    bucket: int = 50000,
    epochs: int = 30,
    workers: int = 4,
) -> FastText:
    """Тренирует FastText (Gensim) на корпусе.

    Для малых корпусов важно:
    - min_count=1 (иначе выкинет слишком много слов)
    - epochs побольше
    """
    sents = [list(s) for s in sentences]
    # Важно: gensim.FastText по умолчанию использует bucket=2_000_000,
    # что делает файл wv.vectors_ngrams.npy ~800MB даже на маленьком корпусе.
    # Для дипломной практики это обычно избыточно, поэтому bucket уменьшаем.
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,  # skip-gram
        min_n=3,
        max_n=6,
        bucket=bucket,
    )
    model.build_vocab(corpus_iterable=sents)
    model.train(corpus_iterable=sents, total_examples=len(sents), epochs=epochs)
    return model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def token_outlier_scores(
    tokens: Sequence[str],
    model: FastText,
    *,
    window: int = 5,
) -> List[float]:
    """Outlier score каждого токена относительно локального контекста.

    score = 1 - cosine(v(tok), mean(v(context_window)))

    Чем больше score, тем сильнее токен выбивается семантически.
    """
    if not tokens:
        return []

    wv = model.wv
    dim = model.vector_size

    out: List[float] = []
    for i, tok in enumerate(tokens):
        # берём окно слева/справа
        left = max(0, i - window)
        right = min(len(tokens), i + window + 1)
        ctx = [tokens[j] for j in range(left, right) if j != i]

        ctx_vecs: List[np.ndarray] = []
        for c in ctx:
            if c in wv:
                ctx_vecs.append(wv[c])

        if tok in wv:
            v_tok = wv[tok]
        else:
            # OOV/редкое слово — считаем подозрительным
            # (в реальной задаче можно смягчать, если слово просто редкое)
            out.append(1.0)
            continue

        if not ctx_vecs:
            out.append(0.0)
            continue

        v_ctx = np.mean(ctx_vecs, axis=0)
        cos = _cosine(v_tok, v_ctx)
        out.append(1.0 - cos)

    return out
