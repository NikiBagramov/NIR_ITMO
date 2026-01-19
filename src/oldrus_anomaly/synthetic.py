from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Substitution:
    pos: int
    orig: str
    new: str


@dataclass
class Deletion:
    pos: int  # позиция удалённого токена в чистой последовательности
    token: str
    left: str | None
    right: str | None


@dataclass
class CorruptedSentence:
    id: str
    tokens_clean: List[str]
    tokens_corrupted: List[str]
    substitutions: List[Substitution]
    deletions: List[Deletion]


def _levenshtein_limit(a: str, b: str, max_dist: int) -> Optional[int]:
    """Levenshtein distance with early stop. Returns distance if <= max_dist else None."""
    if abs(len(a) - len(b)) > max_dist:
        return None
    # DP with pruning
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        # Track row min for pruning
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            del_ = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(ins, del_, sub)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > max_dist:
            return None
        prev = cur
    d = prev[-1]
    return d if d <= max_dist else None


def sample_confusable_token(
    word: str,
    vocab: Sequence[str],
    rng: random.Random,
    *,
    max_dist: int = 2,
    max_checks: int = 2000,
) -> Optional[str]:
    """Пытается подобрать слово из vocab, похожее на word по Levenshtein.

    max_checks ограничивает перебор для больших словарей.
    """
    if not vocab:
        return None

    # Чуть сузим кандидатов: по длине и по первому символу (часто помогает)
    candidates: List[str] = []
    wlen = len(word)
    first = word[0] if word else ""
    for v in vocab:
        if v == word:
            continue
        if abs(len(v) - wlen) > max_dist:
            continue
        if first and v and v[0] != first:
            continue
        candidates.append(v)
        if len(candidates) >= max_checks:
            break

    if not candidates:
        return None

    rng.shuffle(candidates)
    for v in candidates:
        d = _levenshtein_limit(word, v, max_dist=max_dist)
        if d is not None:
            return v
    return None


def corrupt_sentence(
    tokens: Sequence[str],
    vocab: Sequence[str],
    rng: random.Random,
    *,
    p_delete: float = 0.15,
    p_substitute: float = 0.15,
    max_events: int = 2,
    prefer_confusable: bool = True,
) -> Tuple[List[str], List[Substitution], List[Deletion]]:
    """Делает синтетические аномалии в одном предложении.

    - удаление слова (пропуск)
    - замена слова

    max_events ограничивает общее число изменений.
    """
    toks = list(tokens)
    subs: List[Substitution] = []
    dels: List[Deletion] = []

    if len(toks) < 6:
        return toks, subs, dels

    # Список позиций, которые можно трогать
    # (можно добавить стоп-слова/фильтры, если нужно)
    positions = list(range(len(toks)))
    rng.shuffle(positions)

    events_left = max_events

    # 1) Удаление
    if events_left > 0 and rng.random() < p_delete:
        pos = positions.pop() if positions else rng.randrange(len(toks))
        left = toks[pos - 1] if pos - 1 >= 0 else None
        right = toks[pos + 1] if pos + 1 < len(toks) else None
        token = toks[pos]
        del toks[pos]
        dels.append(Deletion(pos=pos, token=token, left=left, right=right))
        events_left -= 1

        # после удаления индексы съезжают; проще не делать второй delete в этот же раз

    # 2) Замена
    if events_left > 0 and rng.random() < p_substitute and len(toks) >= 3:
        pos = rng.randrange(len(toks))
        orig = toks[pos]

        new: Optional[str] = None
        if prefer_confusable:
            new = sample_confusable_token(orig, vocab, rng)
        if new is None:
            # случайная подмена
            new = rng.choice(vocab) if vocab else orig
            if new == orig and len(vocab) > 1:
                for _ in range(10):
                    cand = rng.choice(vocab)
                    if cand != orig:
                        new = cand
                        break

        toks[pos] = new
        subs.append(Substitution(pos=pos, orig=orig, new=new))
        events_left -= 1

    return toks, subs, dels


def corrupt_corpus(
    sentences: Sequence[Tuple[str, Sequence[str]]],
    vocab: Sequence[str],
    *,
    seed: int = 42,
    p_delete: float = 0.15,
    p_substitute: float = 0.15,
    max_events: int = 2,
    prefer_confusable: bool = True,
) -> List[CorruptedSentence]:
    rng = random.Random(seed)

    out: List[CorruptedSentence] = []
    for sid, toks in sentences:
        toks_cor, subs, dels = corrupt_sentence(
            toks,
            vocab,
            rng,
            p_delete=p_delete,
            p_substitute=p_substitute,
            max_events=max_events,
            prefer_confusable=prefer_confusable,
        )
        out.append(
            CorruptedSentence(
                id=str(sid),
                tokens_clean=list(toks),
                tokens_corrupted=list(toks_cor),
                substitutions=subs,
                deletions=dels,
            )
        )

    return out


def to_jsonable(cs: CorruptedSentence) -> Dict:
    return {
        "id": cs.id,
        "tokens_clean": cs.tokens_clean,
        "tokens_corrupted": cs.tokens_corrupted,
        "substitutions": [s.__dict__ for s in cs.substitutions],
        "deletions": [d.__dict__ for d in cs.deletions],
    }
