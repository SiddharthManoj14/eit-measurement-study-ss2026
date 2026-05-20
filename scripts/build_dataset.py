from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Make sure Python can import from src/ when this script is run
# as:
#     python scripts/build_dataset.py
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.io_utils import find_txt_files, load_txt_file_with_report
from src.preprocessing import replace_outliers_with_channel_median
from src.features import create_feature_row, create_feature_dataframe


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

SUMMARY_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "summaries"
)

OUTLIER_SUMMARY_OUTPUT_PATH = (
    SUMMARY_OUTPUT_DIR
    / "outlier_summary.csv"
)

PARSING_REPORT_OUTPUT_PATH = (
    SUMMARY_OUTPUT_DIR
    / "parsing_report.csv"
)

BUILD_SUMMARY_OUTPUT_PATH = (
    SUMMARY_OUTPUT_DIR
    / "dataset_build_summary.csv"
)


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

AGGREGATION_METHOD = "median"

MIN_VALID_VOLTAGE = 0.0
MAX_VALID_VOLTAGE = 5.0


# Channel count depends on injection pattern.
# Adjacent has 208 values per measurement line.
# Opposite and Skip3 have 192 values per measurement line.
EXPECTED_CHANNELS_BY_PATTERN = {
    "Adjacent": 208,
    "Opposite": 192,
    "Skip3": 192,
}


FEATURE_OUTPUT_PATHS_BY_PATTERN = {
    "Adjacent": FEATURE_OUTPUT_DIR / "feature_vectors_adjacent.csv",
    "Opposite": FEATURE_OUTPUT_DIR / "feature_vectors_opposite.csv",
    "Skip3": FEATURE_OUTPUT_DIR / "feature_vectors_skip3.csv",
}


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save a DataFrame to CSV and create the parent folder if needed.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def get_injection_pattern_from_path(file_path: Path) -> str:
    """
    Extract injection pattern from file path.

    Expected structure:
        data/raw/{InjectionPattern}/{TankShape}/...

    Example:
        data/raw/Adjacent/rectangle/non-conductive/centre.txt
    """

    relative_path = file_path.relative_to(RAW_DATA_DIR)
    parts = relative_path.parts

    if len(parts) < 1:
        raise ValueError(f"Cannot infer injection pattern from: {relative_path}")

    raw_pattern = parts[0].strip()

    normalized = raw_pattern.lower().replace("-", "").replace("_", "")

    if normalized == "adjacent":
        return "Adjacent"

    if normalized == "opposite":
        return "Opposite"

    if normalized in ["skip3", "skip"]:
        return "Skip3"

    raise ValueError(
        f"Unknown injection pattern '{raw_pattern}' in path: {relative_path}"
    )


def build_dataset() -> None:
    """
    Build cleaned ML feature datasets from raw EIT .txt files.

    Important:
        One .txt file = one physical condition = one ML sample.

    Because injection patterns have different feature lengths:
        Adjacent  -> 208 voltage features
        Opposite  -> 192 voltage features
        Skip3     -> 192 voltage features

    This script creates separate feature CSVs:
        feature_vectors_adjacent.csv
        feature_vectors_opposite.csv
        feature_vectors_skip3.csv
    """

    print("Starting EIT dataset build...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data directory: {RAW_DATA_DIR}")

    txt_files = find_txt_files(RAW_DATA_DIR)

    print(f"Found {len(txt_files)} .txt files.")

    feature_rows_by_pattern = {
        "Adjacent": [],
        "Opposite": [],
        "Skip3": [],
    }

    all_outlier_summaries = []
    all_parsing_reports = []
    skipped_files = []
    build_summary_rows = []

    for file_index, file_path in enumerate(txt_files, start=1):
        print(f"\n[{file_index}/{len(txt_files)}] Processing: {file_path}")

        try:
            injection_pattern = get_injection_pattern_from_path(file_path)

            expected_channels = EXPECTED_CHANNELS_BY_PATTERN[injection_pattern]

            print(f"  Injection pattern: {injection_pattern}")
            print(f"  Expected channels: {expected_channels}")

            raw_data, parsing_report = load_txt_file_with_report(
                file_path=file_path,
                expected_channels=expected_channels
            )

            if not parsing_report.empty:
                all_parsing_reports.append(parsing_report)

            if raw_data.shape[0] == 0:
                print(
                    f"  Skipped: no valid {expected_channels}-value "
                    "measurements found."
                )

                skipped_files.append(
                    {
                        "file_path": str(file_path),
                        "relative_path": str(file_path.relative_to(RAW_DATA_DIR)),
                        "injection_pattern": injection_pattern,
                        "expected_channels": expected_channels,
                        "reason": "no_valid_measurements",
                    }
                )

                build_summary_rows.append(
                    {
                        "file_path": str(file_path),
                        "relative_path": str(file_path.relative_to(RAW_DATA_DIR)),
                        "injection_pattern": injection_pattern,
                        "expected_channels": expected_channels,
                        "raw_rows": 0,
                        "feature_row_created": False,
                        "n_outliers_replaced": 0,
                        "status": "skipped",
                        "reason": "no_valid_measurements",
                    }
                )

                continue

            print(f"  Raw data shape: {raw_data.shape}")

            cleaned_data, outlier_summary = replace_outliers_with_channel_median(
                data=raw_data,
                expected_channels=expected_channels,
                min_valid_voltage=MIN_VALID_VOLTAGE,
                max_valid_voltage=MAX_VALID_VOLTAGE
            )

            relative_path = file_path.relative_to(RAW_DATA_DIR)

            outlier_summary.insert(0, "source_file", file_path.name)
            outlier_summary.insert(1, "relative_path", str(relative_path))
            outlier_summary.insert(2, "injection_pattern", injection_pattern)
            outlier_summary.insert(3, "expected_channels", expected_channels)

            all_outlier_summaries.append(outlier_summary)

            total_outliers = int(outlier_summary["n_outliers"].sum())

            print(f"  Total spike outliers replaced: {total_outliers}")

            feature_row = create_feature_row(
                cleaned_data=cleaned_data,
                file_path=file_path,
                raw_root=RAW_DATA_DIR,
                aggregation=AGGREGATION_METHOD,
                expected_channels=expected_channels
            )

            feature_row["expected_channels"] = expected_channels

            feature_rows_by_pattern[injection_pattern].append(feature_row)

            print("  Feature row created.")

            build_summary_rows.append(
                {
                    "file_path": str(file_path),
                    "relative_path": str(relative_path),
                    "injection_pattern": injection_pattern,
                    "expected_channels": expected_channels,
                    "raw_rows": raw_data.shape[0],
                    "feature_row_created": True,
                    "n_outliers_replaced": total_outliers,
                    "status": "processed",
                    "reason": "",
                }
            )

        except Exception as error:
            print(f"  ERROR: {error}")

            try:
                relative_path = str(file_path.relative_to(RAW_DATA_DIR))
            except Exception:
                relative_path = str(file_path)

            skipped_files.append(
                {
                    "file_path": str(file_path),
                    "relative_path": relative_path,
                    "injection_pattern": "unknown",
                    "expected_channels": None,
                    "reason": str(error),
                }
            )

            build_summary_rows.append(
                {
                    "file_path": str(file_path),
                    "relative_path": relative_path,
                    "injection_pattern": "unknown",
                    "expected_channels": None,
                    "raw_rows": 0,
                    "feature_row_created": False,
                    "n_outliers_replaced": 0,
                    "status": "error",
                    "reason": str(error),
                }
            )

            continue

    print("\nSaving feature-vector CSV files...")

    for injection_pattern, feature_rows in feature_rows_by_pattern.items():
        feature_df = create_feature_dataframe(feature_rows)

        output_path = FEATURE_OUTPUT_PATHS_BY_PATTERN[injection_pattern]

        save_dataframe(feature_df, output_path)

        print(f"\nSaved {injection_pattern} feature vectors:")
        print(f"  {output_path}")
        print(f"  Shape: {feature_df.shape}")

        if not feature_df.empty:
            print("  Tank-shape counts:")
            print(feature_df["tank_shape"].value_counts().to_string())

    if len(all_outlier_summaries) > 0:
        outlier_summary_df = pd.concat(
            all_outlier_summaries,
            ignore_index=True
        )
    else:
        outlier_summary_df = pd.DataFrame()

    save_dataframe(outlier_summary_df, OUTLIER_SUMMARY_OUTPUT_PATH)

    print("\nSaved outlier summary:")
    print(f"  {OUTLIER_SUMMARY_OUTPUT_PATH}")
    print(f"  Shape: {outlier_summary_df.shape}")

    report_frames = []

    if len(all_parsing_reports) > 0:
        parsing_report_df = pd.concat(
            all_parsing_reports,
            ignore_index=True
        )
        report_frames.append(parsing_report_df)

    if len(skipped_files) > 0:
        skipped_files_df = pd.DataFrame(skipped_files)
        skipped_files_df["status"] = "skipped_file"
        report_frames.append(skipped_files_df)

    if len(report_frames) > 0:
        final_parsing_report_df = pd.concat(
            report_frames,
            ignore_index=True
        )
    else:
        final_parsing_report_df = pd.DataFrame(
            columns=[
                "file_path",
                "line_number",
                "measurement_number",
                "n_values_found",
                "expected_values",
                "status",
                "reason",
            ]
        )

    save_dataframe(final_parsing_report_df, PARSING_REPORT_OUTPUT_PATH)

    print("\nSaved parsing report:")
    print(f"  {PARSING_REPORT_OUTPUT_PATH}")
    print(f"  Shape: {final_parsing_report_df.shape}")

    build_summary_df = pd.DataFrame(build_summary_rows)

    save_dataframe(build_summary_df, BUILD_SUMMARY_OUTPUT_PATH)

    print("\nSaved dataset build summary:")
    print(f"  {BUILD_SUMMARY_OUTPUT_PATH}")
    print(f"  Shape: {build_summary_df.shape}")

    print("\nDataset build complete.")

    print("\nSummary:")
    print(f"  Files found: {len(txt_files)}")

    for injection_pattern, feature_rows in feature_rows_by_pattern.items():
        print(
            f"  {injection_pattern} feature rows created: "
            f"{len(feature_rows)}"
        )

    print(f"  Files skipped/errors: {len(skipped_files)}")

    print("\nMain ML output files:")
    print(f"  {FEATURE_OUTPUT_PATHS_BY_PATTERN['Adjacent']}")
    print(f"  {FEATURE_OUTPUT_PATHS_BY_PATTERN['Opposite']}")
    print(f"  {FEATURE_OUTPUT_PATHS_BY_PATTERN['Skip3']}")


if __name__ == "__main__":
    build_dataset()