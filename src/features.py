from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CHANNELS = 208


FEATURE_COLUMNS = [f"V_{i}" for i in range(1, EXPECTED_CHANNELS + 1)]


METADATA_COLUMNS = [
    "source_file",
    "relative_path",
    "injection_pattern",
    "tank_shape",
    "inclusion_type",
    "position",
    "position_class",
]


def validate_cleaned_data(
    cleaned_data: np.ndarray,
    expected_channels: int = EXPECTED_CHANNELS
) -> np.ndarray:
    """
    Validate cleaned measurement data.

    Expected shape:
        N x 208

    N = number of repeated measurements in one .txt file
    208 = number of voltage channels
    """

    cleaned_data = np.asarray(cleaned_data, dtype=float)

    if cleaned_data.ndim != 2:
        raise ValueError(
            f"Expected 2D array with shape (N, {expected_channels}), "
            f"but got shape {cleaned_data.shape}."
        )

    if cleaned_data.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, "
            f"but got {cleaned_data.shape[1]}."
        )

    if cleaned_data.shape[0] == 0:
        raise ValueError("Cannot create feature vector from empty cleaned data.")

    return cleaned_data


def create_feature_vector(
    cleaned_data: np.ndarray,
    aggregation: str = "median",
    expected_channels: int = EXPECTED_CHANNELS
) -> np.ndarray:
    """
    Convert one cleaned N x 208 matrix into one 208-value feature vector.

    Recommended:
        aggregation = "median"

    Why median?
        It is robust for repeated EIT measurements and less sensitive to
        remaining small fluctuations.
    """

    cleaned_data = validate_cleaned_data(
        cleaned_data,
        expected_channels=expected_channels
    )

    aggregation = aggregation.lower().strip()

    if aggregation == "median":
        feature_vector = np.median(cleaned_data, axis=0)
    elif aggregation == "mean":
        feature_vector = np.mean(cleaned_data, axis=0)
    else:
        raise ValueError(
            f"Unsupported aggregation method: {aggregation}. "
            "Use 'median' or 'mean'."
        )

    return feature_vector


def normalize_text(value: str) -> str:
    """
    Normalize folder and file names so labels are consistent.

    Examples:
        'non conductive'  -> 'non_conductive'
        'non-conductive' -> 'non_conductive'
        'centre'         -> 'centre'
        'center'         -> 'centre'
        'Skip-3'         -> 'Skip3'
    """

    value = value.strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    while "__" in value:
        value = value.replace("__", "_")

    replacements = {
        "center": "centre",
        "nonconductive": "non_conductive",
        "non_condutive": "non_conductive",
        "non_conductive": "non_conductive",
        "conductive": "conductive",
        "onlywater": "onlywater",
        "only_water": "onlywater",
        "skip_3": "Skip3",
        "skip3": "Skip3",
        "adjacent": "Adjacent",
        "opposite": "Opposite",
        "circle": "circle",
        "rectangle": "rectangle",
        "triangle": "triangle",
    }

    return replacements.get(value, value)


def infer_position_class(position: str) -> str:
    """
    Convert detailed position names into broader ML labels.

    Examples:
        centre.txt                  -> centre
        center.txt                  -> centre
        edge_toward_electrode0.txt  -> edge
        between_centre_edge.txt     -> between_centre_edge
        onlywater.txt               -> baseline
    """

    position_norm = normalize_text(position)

    if "onlywater" in position_norm or "baseline" in position_norm:
        return "baseline"

    if "between" in position_norm:
        return "between_centre_edge"

    if "edge" in position_norm:
        return "edge"

    if "centre" in position_norm:
        return "centre"

    return "unknown"


def infer_metadata_from_path(
    file_path: str | Path,
    raw_root: str | Path
) -> dict:
    """
    Extract ML labels from the raw data folder structure.

    Expected normal structure:
        data/raw/{InjectionPattern}/{TankShape}/{InclusionType}/{Position}.txt

    Example:
        data/raw/Adjacent/circle/conductive/centre.txt

    Also supports baseline files like:
        data/raw/Adjacent/circle/onlywater.txt
    """

    file_path = Path(file_path)
    raw_root = Path(raw_root)

    relative_path = file_path.relative_to(raw_root)
    parts = relative_path.parts

    if len(parts) < 3:
        raise ValueError(
            "File path is too short to infer metadata. "
            f"Expected at least injection/tank/file, got: {relative_path}"
        )

    injection_pattern = normalize_text(parts[0])
    tank_shape = normalize_text(parts[1])

    file_stem = normalize_text(file_path.stem)

    if len(parts) == 3:
        inclusion_type = "baseline" if "onlywater" in file_stem else "unknown"
        position = "baseline" if "onlywater" in file_stem else file_stem

    else:
        inclusion_type = normalize_text(parts[2])
        position = file_stem

        if "onlywater" in position:
            inclusion_type = "baseline"
            position = "baseline"

    position_class = infer_position_class(position)

    metadata = {
        "source_file": file_path.name,
        "relative_path": str(relative_path),
        "injection_pattern": injection_pattern,
        "tank_shape": tank_shape,
        "inclusion_type": inclusion_type,
        "position": position,
        "position_class": position_class,
    }

    return metadata


def create_feature_row(
    cleaned_data: np.ndarray,
    file_path: str | Path,
    raw_root: str | Path,
    aggregation: str = "median",
    expected_channels: int = EXPECTED_CHANNELS
) -> dict:
    """
    Create one complete ML-ready row from one cleaned .txt file.

    Output row contains:
        metadata columns
        V_1 to V_208 feature columns
    """

    feature_vector = create_feature_vector(
        cleaned_data=cleaned_data,
        aggregation=aggregation,
        expected_channels=expected_channels
    )

    metadata = infer_metadata_from_path(
        file_path=file_path,
        raw_root=raw_root
    )

    feature_values = {
        FEATURE_COLUMNS[i]: float(feature_vector[i])
        for i in range(expected_channels)
    }

    row = {}
    row.update(metadata)
    row.update(feature_values)

    return row


def create_feature_dataframe(feature_rows: list[dict]) -> pd.DataFrame:
    """
    Convert a list of feature rows into a clean DataFrame.

    This DataFrame is what build_dataset.py will later save as:

        data/processed/features/feature_vectors.csv
    """

    if len(feature_rows) == 0:
        columns = METADATA_COLUMNS + FEATURE_COLUMNS
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(feature_rows)

    ordered_columns = METADATA_COLUMNS + FEATURE_COLUMNS

    existing_ordered_columns = [
        column for column in ordered_columns
        if column in df.columns
    ]

    extra_columns = [
        column for column in df.columns
        if column not in existing_ordered_columns
    ]

    df = df[existing_ordered_columns + extra_columns]

    return df


def filter_classification_rows(
    feature_df: pd.DataFrame,
    include_baseline: bool = False
) -> pd.DataFrame:
    """
    Filter rows for ML classification.

    Usually, onlywater/baseline rows should not be used as normal
    conductive vs non-conductive classification samples.

    Baseline rows are mainly useful for baseline correction,
    SNR, and distinguishability.
    """

    if include_baseline:
        return feature_df.copy()

    if "inclusion_type" not in feature_df.columns:
        raise ValueError("feature_df must contain an 'inclusion_type' column.")

    filtered_df = feature_df[
        feature_df["inclusion_type"] != "baseline"
    ].copy()

    return filtered_df