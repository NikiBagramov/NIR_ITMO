from __future__ import annotations

import math
from typing import Dict, List, Sequence

from gensim.models import FastText

from .fasttext_embed import token_outlier_scores
from .ngram_lm import NgramLM, bidirectional_token_surprisal, gap_scores_bigram_bidirectional


def token_level_features(
    sent_id: str,
    tokens: Sequence[str],
    *,
    fwd: NgramLM,
    bwd: NgramLM,
    fasttext_model: FastText,
) -> List[Dict]:
    """Фичи для каждого токена в предложении."""
    surp = bidirectional_token_surprisal(fwd, bwd, tokens)
    outlier = token_outlier_scores(tokens, fasttext_model)

    # частоты можно взять из LM
    uni = fwd.counts.unigram if fwd.counts is not None else None

    rows: List[Dict] = []
    for i, tok in enumerate(tokens):
        freq = int(uni[tok]) if uni is not None else 0
        rows.append(
            {
                "sent_id": sent_id,
                "token_idx": i,
                "token": tok,
                "ngram_surp": float(surp[i]),
                "ft_outlier": float(outlier[i]),
                "log_freq": float(math.log(freq + 1.0)),
                "char_len": float(len(tok)),
            }
        )
    return rows


def gap_level_features(
    sent_id: str,
    tokens: Sequence[str],
    *,
    fwd: NgramLM,
    bwd: NgramLM,
) -> List[Dict]:
    """Фичи для каждого промежутка между словами (под пропуск)."""
    scores = gap_scores_bigram_bidirectional(fwd, bwd, tokens)
    rows: List[Dict] = []
    for i, sc in enumerate(scores):
        rows.append(
            {
                "sent_id": sent_id,
                "gap_idx": i,
                "left": tokens[i],
                "right": tokens[i + 1],
                "gap_score": float(sc),
            }
        )
    return rows
