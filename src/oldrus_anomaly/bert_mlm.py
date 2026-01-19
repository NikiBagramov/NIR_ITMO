"""BERT/MLM скоринг для древних текстов.

Этот модуль ОПЦИОНАЛЬНЫЙ: требует установленных `transformers` и модели.

Рекомендуемая модель для славянских древних текстов:
- npedrazzini/BERTislav  (HuggingFace)

Идея: считаем pseudo log-likelihood (PLL) каждого слова:
- токенизируем предложение как список слов (is_split_into_words=True)
- для каждого *подтокена* слова по очереди ставим [MASK] и оцениваем log P(original_subtoken)
- суммируем по подтокенам -> NLL на слово

Чем выше NLL (или surprisal), тем более «неожиданное» слово.

Ограничение: MLM scoring дорогой (O(N) прогонов). Для практики это ок на коротких предложениях.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass
class MLMScoringResult:
    word_nll: List[float]
    word_nll_norm: List[float]


def load_mlm(model_name: str = "npedrazzini/BERTislav", device: str = "cpu"):
    """Загружает токенизатор и модель masked LM."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def pll_word_nll(
    tokens: Sequence[str],
    *,
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 24,
) -> MLMScoringResult:
    """Pseudo log-likelihood NLL по словам.

    Возвращает:
    - word_nll: суммарная NLL по подтокенам слова
    - word_nll_norm: нормированная (делим на число подтокенов)
    """
    import math

    import torch

    enc = tokenizer(
        list(tokens),
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    word_ids = enc.word_ids()  # для fast tokenizer

    # mapping: word_idx -> list of positions in input_ids
    positions_by_word: List[List[int]] = [[] for _ in range(len(tokens))]
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        positions_by_word[wid].append(pos)

    mask_id = tokenizer.mask_token_id

    word_nll = [0.0] * len(tokens)
    word_nll_norm = [0.0] * len(tokens)

    # Подготовим задачи: (word_idx, subtoken_pos)
    tasks: List[Tuple[int, int]] = []
    for widx, poss in enumerate(positions_by_word):
        for p in poss:
            tasks.append((widx, p))

    # батчами делаем маски
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        if not batch:
            continue

        # (B, L)
        batch_input = input_ids.repeat(len(batch), 1)
        batch_attn = attention_mask.repeat(len(batch), 1)

        orig_ids = []
        mask_positions = []
        for i, (widx, pos) in enumerate(batch):
            orig_ids.append(int(batch_input[i, pos].item()))
            mask_positions.append(pos)
            batch_input[i, pos] = mask_id

        with torch.no_grad():
            out = model(input_ids=batch_input, attention_mask=batch_attn)
            logits = out.logits  # (B,L,V)

        for i, (widx, pos) in enumerate(batch):
            logit_vec = logits[i, pos]
            log_probs = torch.log_softmax(logit_vec, dim=-1)
            lp = float(log_probs[orig_ids[i]].item())
            word_nll[widx] += -lp

    for widx, poss in enumerate(positions_by_word):
        if poss:
            word_nll_norm[widx] = word_nll[widx] / len(poss)
        else:
            word_nll_norm[widx] = word_nll[widx]

    return MLMScoringResult(word_nll=word_nll, word_nll_norm=word_nll_norm)


def topk_word_replacements(
    tokens: Sequence[str],
    idx: int,
    *,
    tokenizer,
    model,
    device: str = "cpu",
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Топ-k подсказок для замены слова tokens[idx].

    Упрощение: маскируем *первый* подтокен слова и декодируем top-k.
    Для точной реконструкции слова из нескольких подтокенов нужен beam search.
    """
    import torch

    enc = tokenizer(list(tokens), is_split_into_words=True, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    word_ids = enc.word_ids()

    positions = [p for p, wid in enumerate(word_ids) if wid == idx]
    if not positions:
        return []
    pos0 = positions[0]

    mask_id = tokenizer.mask_token_id
    input_ids = input_ids.clone()
    input_ids[0, pos0] = mask_id

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits

    log_probs = torch.log_softmax(logits[0, pos0], dim=-1)
    vals, ids = torch.topk(log_probs, k=top_k)

    out: List[Tuple[str, float]] = []
    for lp, tid in zip(vals.tolist(), ids.tolist()):
        out.append((tokenizer.decode([tid]).strip(), float(lp)))
    return out
