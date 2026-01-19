from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from .docx_loader import load_docx
from .normalize import normalize_raw, normalize_token


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё\u0400-\u052F\u2DE0-\u2DFF\uA640-\uA69F]+")

# Разделители предложений (после normalize_raw многие знаки превращаются в '.')
SENT_SPLIT_RE = re.compile(r"[.!?]+")


@dataclass
class Sentence:
    id: str
    tokens: List[str]


def read_text_files(paths: Sequence[str | Path]) -> str:
    chunks: List[str] = []
    for p in paths:
        pp = Path(p)
        chunks.append(pp.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def read_docx_files(paths: Sequence[str | Path]) -> str:
    chunks: List[str] = []
    for p in paths:
        lt = load_docx(p)
        chunks.append(lt.text)
    return "\n".join(chunks)


def text_to_sentences(
    text: str,
    *,
    min_tokens: int = 6,
    max_tokens: int = 80,
    drop_numeric_tokens: bool = True,
) -> List[Sentence]:
    """Конвертирует сырой текст в список предложений (списков токенов)."""

    norm = normalize_raw(text)

    # сначала режем по пустым строкам, чтобы не склеивать сильно разные блоки
    blocks = [b.strip() for b in norm.split("\n") if b.strip()]

    sentences: List[Sentence] = []
    sid = 0
    for block in blocks:
        # грубое разбиение на предложения
        parts = [p.strip() for p in SENT_SPLIT_RE.split(block) if p.strip()]
        for part in parts:
            toks = [normalize_token(t) for t in TOKEN_RE.findall(part)]
            toks = [t for t in toks if t]
            if drop_numeric_tokens:
                toks = [t for t in toks if not any(ch.isdigit() for ch in t)]

            if min_tokens <= len(toks) <= max_tokens:
                sentences.append(Sentence(id=f"s{sid}", tokens=toks))
                sid += 1

    return sentences


def load_corpus(
    *,
    input_docx: Optional[Sequence[str | Path]] = None,
    input_txt: Optional[Sequence[str | Path]] = None,
    min_tokens: int = 6,
    max_tokens: int = 80,
) -> List[Sentence]:
    input_docx = list(input_docx or [])
    input_txt = list(input_txt or [])

    if not input_docx and not input_txt:
        raise ValueError("Укажите хотя бы один источник: --input_docx или --input_txt")

    text_parts: List[str] = []
    if input_docx:
        text_parts.append(read_docx_files(input_docx))
    if input_txt:
        text_parts.append(read_text_files(input_txt))

    text = "\n".join(text_parts)
    return text_to_sentences(text, min_tokens=min_tokens, max_tokens=max_tokens)


def save_jsonl(sentences: Sequence[Sentence], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps({"id": s.id, "tokens": s.tokens}, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Sentence]:
    p = Path(path)
    out: List[Sentence] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(Sentence(id=str(obj["id"]), tokens=list(obj["tokens"])))
    return out
