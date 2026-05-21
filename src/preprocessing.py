from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MIN_VALID_VOLTAGE = 0.0
DEFAULT_MAX_VALID_VOLTAGE = 5.0


def validate_measurement_matrix(
    data: np.ndarray,
    expected_channels: int | None = None
) -> np.ndarray:
    """
    Validate that the input data is a 2D numeric measurement matrix.

    Expected shape:
        N x C

    N = number of repeated measurements in one .txt file
    C = number of voltage values per measurement line

    Important:
        Adjacent may have 208 channels.
        Opposite may have 192 channels.
        Skip3 may have 192 channels.

    Therefore expected_channels is optional.
    """

    data = np.asarray(data, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D matrix, but got shape {data.shape}."
        )

    if data.shape[0] == 0:
        raise ValueError("Measurement matrix contains zero valid measurements.")

    if data.shape[1] == 0:
        raise ValueError("Measurement matrix contains zero voltage channels.")

    if expected_channels is not None and data.shape[1] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} voltage channels, "
            f"but got {data.shape[1]}."
        )

    return data


def detect_spike_outliers(
    channel_values: np.ndarray,
    min_valid_voltage: float = DEFAULT_MIN_VALID_VOLTAGE,
    max_valid_voltage: float = DEFAULT_MAX_VALID_VOLTAGE
) -> np.ndarray:
    """
    Detect physically invalid voltage values in one channel.

    For our EIT measurements, normal voltages are usually around:
        0 V to 2.8 V

    Obvious hardware/serial spikes look like:
        48.61 V
        51.53 V
        52.81 V

    Outlier rule:
        value is NaN
        value is infinite
        value < min_valid_voltage
        value > max_valid_voltage

    Default:
        valid range = 0 V to 5 V
    """

    values = np.asarray(channel_values, dtype=float)

    outlier_mask = (
        ~np.isfinite(values)
        | (values < min_valid_voltage)
        | (values > max_valid_voltage)
    )

    return outlier_mask


def replace_outliers_with_channel_median(
    data: np.ndarray,
    expected_channels: int | None = None,
    min_valid_voltage: float = DEFAULT_MIN_VALID_VOLTAGE,
    max_valid_voltage: float = DEFAULT_MAX_VALID_VOLTAGE
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Replace spike outliers channel-wise.

    For each voltage channel:
        1. Detect values outside the valid voltage range.
        2. Calculate the channel median excluding those outliers.
        3. Replace the outliers with that clean channel median.

    This does NOT use IQR.
    This does NOT flag small natural variations.
    This only removes physically invalid voltage spikes.

    Input:
        data:
            Raw measurement matrix with shape N x C

    Output:
        cleaned_data:
            Cleaned measurement matrix with shape N x C

        summary_df:
            Per-channel outlier summary
    """

    data = validate_measurement_matrix(
        data=data,
        expected_channels=expected_channels
    )

    cleaned_data = data.copy()

    n_measurements, n_channels = cleaned_data.shape

    summary_rows = []

    for channel_idx in range(n_channels):
        channel_values = cleaned_data[:, channel_idx].copy()

        outlier_mask = detect_spike_outliers(
            channel_values=channel_values,
            min_valid_voltage=min_valid_voltage,
            max_valid_voltage=max_valid_voltage
        )

        clean_values = channel_values[~outlier_mask]
        outlier_values = channel_values[outlier_mask]

        if clean_values.size > 0:
            replacement_median = float(np.median(clean_values))
        else:
            replacement_median = np.nan

        cleaned_data[outlier_mask, channel_idx] = replacement_median

        outlier_measurement_indices = np.where(outlier_mask)[0] + 1

        if np.all(np.isnan(cleaned_data[:, channel_idx])):
            cleaned_min = np.nan
            cleaned_max = np.nan
        else:
            cleaned_min = float(np.nanmin(cleaned_data[:, channel_idx]))
            cleaned_max = float(np.nanmax(cleaned_data[:, channel_idx]))

        if np.all(np.isnan(channel_values)):
            raw_min = np.nan
            raw_max = np.nan
        else:
            raw_min = float(np.nanmin(channel_values))
            raw_max = float(np.nanmax(channel_values))

        summary_rows.append(
            {
                "channel": channel_idx + 1,
                "n_measurements": n_measurements,
                "n_outliers": int(outlier_mask.sum()),
                "outlier_measurements": ";".join(
                    str(index) for index in outlier_measurement_indices
                ),
                "outlier_values": ";".join(
                    str(float(value)) for value in outlier_values
                ),
                "replacement_median": replacement_median,
                "min_valid_voltage": min_valid_voltage,
                "max_valid_voltage": max_valid_voltage,
                "raw_min": raw_min,
                "raw_max": raw_max,
                "cleaned_min": cleaned_min,
                "cleaned_max": cleaned_max,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return cleaned_data, summary_df


def create_cleaned_channel_median_vector(
    cleaned_data: np.ndarray,
    expected_channels: int | None = None
) -> np.ndarray:
    """
    Create one feature vector from one cleaned .txt file.

    For each voltage channel:
        feature value = median of the cleaned repeated measurements

    Input:
        cleaned_data:
            N x C matrix

    Output:
        feature_vector:
            1 x C vector
    """

    cleaned_data = validate_measurement_matrix(
        data=cleaned_data,
        expected_channels=expected_channels
    )

    feature_vector = np.nanmedian(cleaned_data, axis=0)

    return feature_vector


def clean_and_create_feature_vector(
    data: np.ndarray,
    expected_channels: int | None = None,
    min_valid_voltage: float = DEFAULT_MIN_VALID_VOLTAGE,
    max_valid_voltage: float = DEFAULT_MAX_VALID_VOLTAGE
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Full preprocessing pipeline for one .txt file.

    Steps:
        1. Validate raw measurement matrix.
        2. Detect voltage spikes using a physical threshold.
        3. Replace spikes with channel median excluding those spikes.
        4. Create one feature vector using channel-wise medians.

    Returns:
        cleaned_data
        feature_vector
        summary_df
    """

    cleaned_data, summary_df = replace_outliers_with_channel_median(
        data=data,
        expected_channels=expected_channels,
        min_valid_voltage=min_valid_voltage,
        max_valid_voltage=max_valid_voltage
    )

    feature_vector = create_cleaned_channel_median_vector(
        cleaned_data=cleaned_data,
        expected_channels=expected_channels
    )

    return cleaned_data, feature_vector, summary_df