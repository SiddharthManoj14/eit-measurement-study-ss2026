from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.io_utils import (
    find_txt_files,
    load_txt_file_with_report,
    save_dataframe_to_csv,
)

from src.preprocessing import replace_outliers_with_channel_median

from src.features import (
    create_feature_row,
    get_expected_channels_for_file,
    get_injection_pattern_from_path,
)


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "engineered_features.csv"
)

OUTLIER_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "summaries"
    / "outlier_summary.csv"
)

PARSING_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "summaries"
    / "parsing_report.csv"
)

BUILD_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "summaries"
    / "dataset_build_summary.csv"
)


MIN_VALID_VOLTAGE = 0.0
MAX_VALID_VOLTAGE = 5.0


def build_dataset() -> None:
    txt_files = find_txt_files(RAW_DATA_DIR)

    feature_rows = []
    outlier_summaries = []
    parsing_reports = []
    build_rows = []

    print(f"Found {len(txt_files)} txt files.")

    for index, file_path in enumerate(txt_files, start=1):
        print(f"\n[{index}/{len(txt_files)}] {file_path}")

        try:
            relative_path = file_path.relative_to(RAW_DATA_DIR)

            injection_pattern = get_injection_pattern_from_path(
                file_path=file_path,
                raw_root=RAW_DATA_DIR,
            )

            expected_channels = get_expected_channels_for_file(
                file_path=file_path,
                raw_root=RAW_DATA_DIR,
            )

            raw_data, parsing_report = load_txt_file_with_report(
                file_path=file_path,
                expected_channels=expected_channels,
            )

            if not parsing_report.empty:
                parsing_reports.append(parsing_report)

            if raw_data.shape[0] == 0:
                print("  skipped, no valid measurement rows")

                build_rows.append(
                    {
                        "relative_path": str(relative_path),
                        "injection_pattern": injection_pattern,
                        "expected_channels": expected_channels,
                        "raw_rows": 0,
                        "status": "skipped",
                        "reason": "no_valid_measurements",
                    }
                )

                continue

            cleaned_data, outlier_summary = replace_outliers_with_channel_median(
                data=raw_data,
                expected_channels=expected_channels,
                min_valid_voltage=MIN_VALID_VOLTAGE,
                max_valid_voltage=MAX_VALID_VOLTAGE,
            )

            total_outliers = int(outlier_summary["n_outliers"].sum())

            outlier_summary.insert(0, "relative_path", str(relative_path))
            outlier_summary.insert(1, "injection_pattern", injection_pattern)
            outlier_summaries.append(outlier_summary)

            feature_row = create_feature_row(
                cleaned_data=cleaned_data,
                file_path=file_path,
                raw_root=RAW_DATA_DIR,
                expected_channels=expected_channels,
                n_outliers_replaced=total_outliers,
            )

            feature_rows.append(feature_row)

            build_rows.append(
                {
                    "relative_path": str(relative_path),
                    "injection_pattern": injection_pattern,
                    "expected_channels": expected_channels,
                    "raw_rows": raw_data.shape[0],
                    "status": "processed",
                    "reason": "",
                    "n_outliers_replaced": total_outliers,
                }
            )

            print(f"  processed, outliers replaced: {total_outliers}")

        except Exception as error:
            print(f"  error: {error}")

            build_rows.append(
                {
                    "relative_path": str(file_path),
                    "injection_pattern": "unknown",
                    "expected_channels": None,
                    "raw_rows": 0,
                    "status": "error",
                    "reason": str(error),
                    "n_outliers_replaced": 0,
                }
            )

    feature_df = pd.DataFrame(feature_rows)
    outlier_df = pd.concat(outlier_summaries, ignore_index=True) if outlier_summaries else pd.DataFrame()
    parsing_df = pd.concat(parsing_reports, ignore_index=True) if parsing_reports else pd.DataFrame()
    build_df = pd.DataFrame(build_rows)

    save_dataframe_to_csv(feature_df, FEATURE_OUTPUT_PATH)
    save_dataframe_to_csv(outlier_df, OUTLIER_SUMMARY_PATH)
    save_dataframe_to_csv(parsing_df, PARSING_REPORT_PATH)
    save_dataframe_to_csv(build_df, BUILD_SUMMARY_PATH)

    print("\nDone.")
    print(f"Feature rows created: {len(feature_df)}")
    print(f"Saved: {FEATURE_OUTPUT_PATH}")


if __name__ == "__main__":
    build_dataset()