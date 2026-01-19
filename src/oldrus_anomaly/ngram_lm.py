from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


BOS = "<s>"
EOS = "</s>"


def _pad(seq: Sequence[str], order: int) -> List[str]:
    return [BOS] * (order - 1) + list(seq) + [EOS]


def _ngrams(seq: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(seq) - n + 1):
        yield tuple(seq[i : i + n])


@dataclass
class NgramCounts:
    unigram: Counter
    bigram: Counter
    trigram: Counter

    total_unigrams: int
    vocab: List[str]


class NgramLM:
    """Простая интерполированная 3-граммная LM с add-alpha сглаживанием.

    Делает то, что нужно для практики:
    - считает log P(w_i | w_{i-2}, w_{i-1}) с бэкоффом
    - отдаёт surprisal ( -log p )

    Важно: это НЕ Kneser-Ney и не SOTA, но достаточно для отчёта как baseline.
    """

    def __init__(self, order: int = 3, alpha: float = 0.1, lambdas: Tuple[float, float, float] = (0.6, 0.3, 0.1)):
        if order != 3:
            raise ValueError("В этой реализации фиксирован order=3")
        if not math.isclose(sum(lambdas), 1.0, abs_tol=1e-6):
            raise ValueError("lambdas должны суммироваться в 1")
        self.order = order
        self.alpha = float(alpha)
        self.l3, self.l2, self.l1 = map(float, lambdas)

        self.counts: NgramCounts | None = None

    def fit(self, sentences: Iterable[Sequence[str]]) -> "NgramLM":
        uni = Counter()
        bi = Counter()
        tri = Counter()

        vocab_set = set([BOS, EOS])

        for sent in sentences:
            padded = _pad(sent, self.order)
            vocab_set.update(padded)

            uni.update(padded)
            bi.update(_ngrams(padded, 2))
            tri.update(_ngrams(padded, 3))

        vocab = sorted(vocab_set)
        total = sum(uni.values())
        self.counts = NgramCounts(unigram=uni, bigram=bi, trigram=tri, total_unigrams=total, vocab=vocab)
        return self

    @property
    def vocab(self) -> List[str]:
        if self.counts is None:
            raise RuntimeError("LM не обучена")
        return self.counts.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _p_unigram(self, w: str) -> float:
        c = self.counts
        assert c is not None
        # add-alpha
        return (c.unigram[w] + self.alpha) / (c.total_unigrams + self.alpha * self.vocab_size)

    def _p_bigram(self, w_prev: str, w: str) -> float:
        c = self.counts
        assert c is not None
        denom = c.unigram[w_prev]
        return (c.bigram[(w_prev, w)] + self.alpha) / (denom + self.alpha * self.vocab_size)

    def _p_trigram(self, w_prev2: str, w_prev1: str, w: str) -> float:
        c = self.counts
        assert c is not None
        denom = c.bigram[(w_prev2, w_prev1)]
        return (c.trigram[(w_prev2, w_prev1, w)] + self.alpha) / (denom + self.alpha * self.vocab_size)

    def prob(self, w: str, context2: Tuple[str, str]) -> float:
        """Интерполированная вероятность слова по (w_{i-2}, w_{i-1})."""
        if self.counts is None:
            raise RuntimeError("LM не обучена")
        w2, w1 = context2
        p3 = self._p_trigram(w2, w1, w)
        p2 = self._p_bigram(w1, w)
        p1 = self._p_unigram(w)
        return self.l3 * p3 + self.l2 * p2 + self.l1 * p1

    def logprob(self, w: str, context2: Tuple[str, str]) -> float:
        p = self.prob(w, context2)
        return math.log(p)

    def sentence_token_surprisal(self, sent: Sequence[str]) -> List[float]:
        """Surprisal ( -log p ) для каждого токена предложения."""
        if self.counts is None:
            raise RuntimeError("LM не обучена")

        padded = _pad(sent, self.order)
        # индексы реальных токенов внутри padded
        # padded = [BOS,BOS] + sent + [EOS]
        out: List[float] = []
        for i in range(self.order - 1, self.order - 1 + len(sent)):
            w2, w1 = padded[i - 2], padded[i - 1]
            w = padded[i]
            out.append(-self.logprob(w, (w2, w1)))
        return out

    def bigram_neglogprob(self, left: str, right: str) -> float:
        """-log P(right | left). Используется для gap-score."""
        if self.counts is None:
            raise RuntimeError("LM не обучена")
        p2 = self._p_bigram(left, right)
        return -math.log(p2)


def train_bidirectional_lm(sentences: Iterable[Sequence[str]], *, alpha: float = 0.1) -> Tuple[NgramLM, NgramLM]:
    """Учит forward LM и backward LM (на развернутых предложениях)."""
    fwd = NgramLM(alpha=alpha).fit(sentences)
    bwd = NgramLM(alpha=alpha).fit([list(reversed(s)) for s in sentences])
    return fwd, bwd


def bidirectional_token_surprisal(fwd: NgramLM, bwd: NgramLM, sent: Sequence[str]) -> List[float]:
    """Средний surprisal по forward и backward LM."""
    s_f = fwd.sentence_token_surprisal(sent)
    s_b_rev = bwd.sentence_token_surprisal(list(reversed(sent)))
    s_b = list(reversed(s_b_rev))
    assert len(s_f) == len(s_b) == len(sent)
    return [(a + b) / 2.0 for a, b in zip(s_f, s_b)]


def gap_scores_bigram_bidirectional(fwd: NgramLM, bwd: NgramLM, sent: Sequence[str]) -> List[float]:
    """Gap-score для каждого промежутка между словами.

    Возвращает список длины len(sent)-1:
    score_i соответствует разрыву между sent[i] и sent[i+1].

    Идея: если P(w_{i+1}|w_i) и P(w_i|w_{i+1}) одновременно малы,
    это может означать, что в оригинале между ними было слово, которое пропали.
    """
    if len(sent) < 2:
        return []

    out: List[float] = []
    for i in range(len(sent) - 1):
        left = sent[i]
        right = sent[i + 1]
        s1 = fwd.bigram_neglogprob(left, right)
        s2 = bwd.bigram_neglogprob(right, left)  # backward LM = P(left | right)
        out.append((s1 + s2) / 2.0)
    return out
