from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_TABLE_DIR, RESULTS_FIGURE_DIR

from src.plotting import (
    plot_feature_comparison,
    plot_features_by_injection_pattern,
    plot_injection_pattern_comparison,
    plot_model_comparison,
    plot_models_by_injection_pattern,
)


def main() -> None:
    RESULTS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(
        table_path=RESULTS_TABLE_DIR / "domain_shape_model_results.csv",
        output_dir=RESULTS_FIGURE_DIR,
    )

    plot_feature_comparison(
        table_path=RESULTS_TABLE_DIR / "domain_shape_feature_comparison.csv",
        output_dir=RESULTS_FIGURE_DIR,
    )

    plot_injection_pattern_comparison(
        table_path=RESULTS_TABLE_DIR / "domain_shape_by_injection_pattern.csv",
        output_dir=RESULTS_FIGURE_DIR,
    )

    plot_models_by_injection_pattern(
        table_path=RESULTS_TABLE_DIR / "domain_shape_models_by_injection_pattern.csv",
        output_dir=RESULTS_FIGURE_DIR,
    )

    plot_features_by_injection_pattern(
        table_path=RESULTS_TABLE_DIR / "domain_shape_features_by_injection_pattern.csv",
        output_dir=RESULTS_FIGURE_DIR,
    )

    print("\nPlot generation complete.")


if __name__ == "__main__":
    main()