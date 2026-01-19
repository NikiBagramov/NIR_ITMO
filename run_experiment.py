from __future__ import annotations

import sys
from pathlib import Path

# Чтобы можно было запускать без установки пакета
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- ХАРДКОД ВХОДНЫХ ФАЙЛОВ ---
DATA_DIR = ROOT / "data"
DOCX_FILES = [
    DATA_DIR / "P1032_Copy.docx",
    DATA_DIR / "Гим Воскр 110_Copy.docx",
]

ARTIFACTS_DIR = ROOT / "artifacts"

# Подсовываем аргументы argparse внутри oldrus_anomaly.run_experiment
sys.argv = [
    "run_experiment.py",
    "--input_docx",
    *map(str, DOCX_FILES),
    "--artifacts_dir",
    str(ARTIFACTS_DIR),
]

from oldrus_anomaly.run_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
