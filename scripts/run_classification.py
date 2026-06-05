from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.classification import run_domain_shape_classification

from src.config import (
    ENGINEERED_FEATURES_PATH,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
)

FEATURE_CSV_PATH = ENGINEERED_FEATURES_PATH



def main() -> None:
    if not FEATURE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_CSV_PATH}\n"
            "Run scripts/build_dataset.py first."
        )

    run_domain_shape_classification(
        feature_csv_path=FEATURE_CSV_PATH,
        results_table_dir=RESULTS_TABLE_DIR,
        results_figure_dir=RESULTS_FIGURE_DIR,
    )


if __name__ == "__main__":
    main()