from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EXPECTED_CHANNELS_BY_PATTERN, FEATURE_COLUMNS

def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("-", "_").replace(" ", "_")

    while "__" in value:
        value = value.replace("__", "_")

    replacements = {
        "center": "centre",
        "nonconductive": "non_conductive",
        "non_condutive": "non_conductive",
        "non_conductive": "non_conductive",
        "only_water": "onlywater",
        "skip_3": "Skip3",
        "skip3": "Skip3",
        "skip": "Skip3",
        "adjacent": "Adjacent",
        "opposite": "Opposite",
    }

    return replacements.get(value, value)


def get_injection_pattern_from_path(file_path: str | Path, raw_root: str | Path) -> str:
    file_path = Path(file_path)
    raw_root = Path(raw_root)

    relative_path = file_path.relative_to(raw_root)
    pattern = normalize_text(relative_path.parts[0])

    if pattern not in EXPECTED_CHANNELS_BY_PATTERN:
        raise ValueError(f"Unknown injection pattern: {pattern}")

    return pattern


def get_expected_channels_for_file(file_path: str | Path, raw_root: str | Path) -> int:
    pattern = get_injection_pattern_from_path(file_path, raw_root)
    return EXPECTED_CHANNELS_BY_PATTERN[pattern]


def infer_position_class(position: str) -> str:
    position = normalize_text(position)

    if "onlywater" in position or "baseline" in position:
        return "baseline"

    if "between" in position:
        return "between_centre_edge"

    if position == "edge" or position.startswith("edge_"):
        return "edge"

    if "electrode" in position:
        return "boundary"

    if "centre" in position:
        return "centre"

    return "unknown"


def infer_metadata_from_path(file_path: str | Path, raw_root: str | Path) -> dict:
    file_path = Path(file_path)
    raw_root = Path(raw_root)

    relative_path = file_path.relative_to(raw_root)
    parts = relative_path.parts

    if len(parts) < 3:
        raise ValueError(f"Path too short for metadata: {relative_path}")

    injection_pattern = get_injection_pattern_from_path(file_path, raw_root)
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

    return {
        "source_file": file_path.name,
        "relative_path": str(relative_path),
        "injection_pattern": injection_pattern,
        "tank_shape": tank_shape,
        "inclusion_type": inclusion_type,
        "position": position,
        "position_class": infer_position_class(position),
    }


def compute_bv_features(boundary_voltage_vector: np.ndarray) -> dict:
    bv = np.asarray(boundary_voltage_vector, dtype=float)
    bv = bv[np.isfinite(bv)]

    if bv.size == 0:
        return {
            "bv_min": np.nan,
            "bv_avg": np.nan,
            "bv_range": np.nan,
            "bv_avg_variation": np.nan,
        }

    bv_min = float(np.min(bv))
    bv_max = float(np.max(bv))
    bv_avg = float(np.mean(bv))
    bv_range = float(bv_max - bv_min)

    if bv.size > 1:
        bv_avg_variation = float(np.mean(np.abs(np.diff(bv))))
    else:
        bv_avg_variation = 0.0

    return {
        "bv_min": bv_min,
        "bv_avg": bv_avg,
        "bv_range": bv_range,
        "bv_avg_variation": bv_avg_variation,
    }


def create_feature_row(
    cleaned_data: np.ndarray,
    file_path: str | Path,
    raw_root: str | Path,
    expected_channels: int,
    n_outliers_replaced: int,
) -> dict:
    cleaned_data = np.asarray(cleaned_data, dtype=float)

    if cleaned_data.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {cleaned_data.shape}")

    if cleaned_data.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, got {cleaned_data.shape[1]}"
        )

    if cleaned_data.shape[0] == 0:
        raise ValueError("No valid measurements found.")

    measurement_features = [
        compute_bv_features(row)
        for row in cleaned_data
    ]

    feature_df = pd.DataFrame(measurement_features)

    output_row = infer_metadata_from_path(file_path, raw_root)
    output_row["expected_channels"] = expected_channels
    output_row["n_measurements"] = cleaned_data.shape[0]
    output_row["n_outliers_replaced"] = int(n_outliers_replaced)

    for column in FEATURE_COLUMNS:
        output_row[column] = float(feature_df[column].median())

    return output_row