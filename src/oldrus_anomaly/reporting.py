from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .ngram_lm import NgramLM
from .suggest import suggest_insertion, suggest_replacements


def render_sentence(tokens: Sequence[str], highlight_token_idx: Optional[int] = None) -> str:
    parts: List[str] = []
    for i, t in enumerate(tokens):
        if highlight_token_idx is not None and i == highlight_token_idx:
            parts.append(f"**{t}**")
        else:
            parts.append(t)
    return " ".join(parts)


def render_gap(tokens: Sequence[str], gap_idx: int) -> str:
    """gap_idx между tokens[gap_idx] и tokens[gap_idx+1]."""
    if gap_idx < 0 or gap_idx >= len(tokens) - 1:
        return " ".join(tokens)
    parts: List[str] = []
    for i, t in enumerate(tokens):
        parts.append(t)
        if i == gap_idx:
            parts.append("**[?]**")
    return " ".join(parts)


def build_top_examples_markdown(
    *,
    token_rows: List[Dict],
    gap_rows: List[Dict],
    sentence_tokens: Dict[str, List[str]],
    lm: NgramLM,
    candidate_vocab: Sequence[str],
    token_score_field: str,
    gap_score_field: str,
    top_n: int = 15,
) -> str:
    """Генерирует markdown с топ‑аномалиями."""

    md: List[str] = []

    # --- token anomalies ---
    md.append("# Top examples")
    md.append("")
    md.append("## Token anomalies (замены)")
    md.append("")

    token_sorted = sorted(token_rows, key=lambda r: float(r[token_score_field]), reverse=True)[:top_n]
    for rank, r in enumerate(token_sorted, start=1):
        sid = r["sent_id"]
        idx = int(r["token_idx"])
        score = float(r[token_score_field])
        token = r["token"]
        label = r.get("label", None)

        toks = sentence_tokens.get(sid, [])
        sent_str = render_sentence(toks, highlight_token_idx=idx)

        sugg = suggest_replacements(toks, idx, lm, candidate_vocab, top_k=5)
        sugg_str = ", ".join([f"{w} ({sc:.2f})" for w, sc in sugg]) if sugg else "(нет)"

        md.append(f"{rank}. **score={score:.4f}** token=`{token}` label={label}")
        md.append("")
        md.append(sent_str)
        md.append("")
        md.append(f"suggest: {sugg_str}")
        md.append("")

    # --- gaps ---
    md.append("## Gap anomalies (пропуски)")
    md.append("")

    gap_sorted = sorted(gap_rows, key=lambda r: float(r[gap_score_field]), reverse=True)[:top_n]
    for rank, r in enumerate(gap_sorted, start=1):
        sid = r["sent_id"]
        gidx = int(r["gap_idx"])
        score = float(r[gap_score_field])
        left = r["left"]
        right = r["right"]
        label = r.get("label", None)

        toks = sentence_tokens.get(sid, [])
        sent_str = render_gap(toks, gidx)

        sugg = suggest_insertion(toks, gidx, lm, candidate_vocab, top_k=5)
        sugg_str = ", ".join([f"{w} ({sc:.2f})" for w, sc in sugg]) if sugg else "(нет)"

        md.append(f"{rank}. **gap_score={score:.4f}** pair=(`{left}` → `{right}`) label={label}")
        md.append("")
        md.append(sent_str)
        md.append("")
        md.append(f"insert suggest: {sugg_str}")
        md.append("")

    return "\n".join(md)
