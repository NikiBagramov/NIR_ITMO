from __future__ import annotations

import re

# Некоторые тексты содержат разметку (например, <6:1>, I 314, Sir 1:1–3 и т.п.)
# В практической задаче это шум для языковой модели.
TAG_RE = re.compile(r"<[^>]+>")

# Цифровые ссылки вида 1:23, 38:9 и т.п.
VERSE_RE = re.compile(r"\b\d+[:.]\d+[a-z]?\b", flags=re.IGNORECASE)

# Мягкие переносы / NBSP
WHITESPACE_RE = re.compile(r"[\u00A0\u00AD\u200B\uFEFF]")

# Упрощённая нормализация некоторых знаков
REPLACEMENTS = {
    "|": " ",
    "·": " . ",
    "⁘": " . ",
    "·": " . ",
    ";": " . ",
    ":": " . ",
    "—": " ",
    "–": " ",
    "‑": " ",
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
}


def normalize_raw(text: str, *, drop_tags: bool = True, drop_verse_refs: bool = True) -> str:
    """Нормализация «сырых» строк перед разбиением на предложения.

    Делает:
    - удаление мягких переносов и спец‑пробелов
    - унификацию разделителей (|, ·, ⁘)
    - опциональное удаление <...> тегов и ссылок вида 1:23
    """
    text = WHITESPACE_RE.sub(" ", text)

    if drop_tags:
        text = TAG_RE.sub(" ", text)

    if drop_verse_refs:
        text = VERSE_RE.sub(" ", text)

    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)

    # схлопываем пробелы
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Нормализация отдельных токенов (консервативная)
TOKEN_CLEAN_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё\u0400-\u052F\u2DE0-\u2DFF\uA640-\uA69F]+")

# Надстрочные/служебные знаки, которые часто мешают токенизации и языковому моделированию.
# Если для вашей задачи важно сохранять их (строгая графемика), удалите/измените этот список.
DROP_CHARS = {
    "҃",  # titlo
    "҆",  # combining cyrillic psili
    "҄",  # combining cyrillic... (варианты)
    "҅",
    "҈",
    "҉",
    "\ue015",  # private use символы из некоторых docx (видимые как "")
}


def normalize_token(tok: str) -> str:
    """Чистит токен от пунктуации по краям и приводит к lower()."""
    tok = tok.strip()
    for ch in DROP_CHARS:
        tok = tok.replace(ch, "")
    tok = TOKEN_CLEAN_RE.sub("", tok)
    return tok.lower()
