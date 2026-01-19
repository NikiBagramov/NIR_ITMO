"""Печатает краткую сводку по metrics.json.

Пример:
  python summarize_metrics.py artifacts/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python summarize_metrics.py <path/to/metrics.json>")
        raise SystemExit(2)

    p = Path(sys.argv[1])
    m = json.loads(p.read_text(encoding="utf-8"))

    top_k = int(m.get("correction", {}).get("top_k", 5))

    print("Corpus:", m.get("corpus"))
    print("\nToken methods:")
    tm = m.get("token_metrics", {})
    for k, v in tm.items():
        print(
            f"  {k:>10} | AP={v.get('average_precision',0):.4f}  P={v.get('precision',0):.3f}  R={v.get('recall',0):.3f}  F1={v.get('f1',0):.3f} (thr={v.get('threshold',0):.3f})"
        )

    print("\nGap methods:")
    gm = m.get("gap_metrics", {})
    for k, v in gm.items():
        print(
            f"  {k:>10} | AP={v.get('average_precision',0):.4f}  P={v.get('precision',0):.3f}  R={v.get('recall',0):.3f}  F1={v.get('f1',0):.3f} (thr={v.get('threshold',0):.3f})"
        )

    print("\nCorrection (top-k) on synthetic errors:")
    c = m.get("correction", {})
    print(
        f"  substitutions: top1={c.get('subs_top1',0):.3f} top{top_k}={c.get('subs_topk',0):.3f} (n={int(c.get('subs_total',0))})"
    )
    print(
        f"  deletions:      top1={c.get('dels_top1',0):.3f} top{top_k}={c.get('dels_topk',0):.3f} (n={int(c.get('dels_total',0))})"
    )


if __name__ == "__main__":
    main()
