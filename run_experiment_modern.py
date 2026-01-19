from __future__ import annotations

import json
import sys
from pathlib import Path

# Чтобы можно было запускать без установки пакета
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_CORPUS = ROOT / "data" / "modern_russian.txt"
DEFAULT_ARTIFACTS = ROOT / "artifacts_modern"


def _ensure_arg(argv: list[str], flag: str, values: list[str]) -> None:
    if flag in argv:
        return
    argv.extend([flag, *values])


def _get_arg_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def _ensure_corpus_exists(argv: list[str]) -> None:
    if "--input_txt" in argv or "--input_docx" in argv:
        return
    if DEFAULT_CORPUS.exists():
        return
    msg = (
        "Не найден файл современного корпуса. "
        f"Ожидается: {DEFAULT_CORPUS}. "
        "Скачайте большой современный русский текст в .txt "
        "(например, Л. Н. Толстой: «Война и мир», «Анна Каренина») "
        "и сохраните по этому пути либо укажите --input_txt."
    )
    raise FileNotFoundError(msg)


def _patch_metrics(artifacts_dir: Path, corpus_type: str) -> None:
    metrics_path = artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["corpus_type"] = corpus_type
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    argv = sys.argv[1:]
    _ensure_corpus_exists(argv)

    _ensure_arg(argv, "--input_txt", [str(DEFAULT_CORPUS)])
    _ensure_arg(argv, "--artifacts_dir", [str(DEFAULT_ARTIFACTS)])

    artifacts_dir = Path(_get_arg_value(argv, "--artifacts_dir") or str(DEFAULT_ARTIFACTS))
    corpus_type = "modern_russian"

    sys.argv = ["run_experiment_modern.py", *argv]

    from oldrus_anomaly.run_experiment import main as run_main  # noqa: E402

    run_main()
    _patch_metrics(artifacts_dir, corpus_type)
    print(f"[info] modern sanity-check done: corpus_type={corpus_type}")


if __name__ == "__main__":
    main()
