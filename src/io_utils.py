from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CHANNELS = 208


FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_measurement_line(line: str) -> list[float]:
    """
    Extract all floating-point numbers from one measurement line.

    Example input line:
        I got: 2.33 0.94 1.99 ... 2.76

    Output:
        [2.33, 0.94, 1.99, ..., 2.76]
    """

    matches = FLOAT_PATTERN.findall(line)
    values = [float(value) for value in matches]

    return values


def is_measurement_line(line: str) -> bool:
    """
    Check whether a line contains one EIT measurement.

    Your raw files use lines like:
        I got: ...
    """

    stripped_line = line.strip()

    return stripped_line.startswith("I")


def load_txt_file_with_report(
    file_path: str | Path,
    expected_channels: int = EXPECTED_CHANNELS
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load one EIT .txt file and return:

        1. data:
            numpy array with shape (N, 208)

        2. report_df:
            dataframe containing invalid or skipped lines

    Only measurement lines with exactly 208 voltage values are accepted.
    Lines with fewer or more values are skipped and logged in the report.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, but got: {file_path}")

    valid_measurements = []
    report_rows = []

    with file_path.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()

    measurement_number = 0

    for line_number, line in enumerate(lines, start=1):
        if not is_measurement_line(line):
            continue

        measurement_number += 1
        values = parse_measurement_line(line)
        value_count = len(values)

        if value_count == expected_channels:
            valid_measurements.append(values)
        else:
            report_rows.append(
                {
                    "file_path": str(file_path),
                    "line_number": line_number,
                    "measurement_number": measurement_number,
                    "n_values_found": value_count,
                    "expected_values": expected_channels,
                    "status": "skipped",
                    "reason": "wrong_number_of_voltage_values",
                }
            )

    if len(valid_measurements) == 0:
        data = np.empty((0, expected_channels), dtype=float)
    else:
        data = np.array(valid_measurements, dtype=float)

    report_df = pd.DataFrame(report_rows)

    return data, report_df


def load_txt_file(
    file_path: str | Path,
    expected_channels: int = EXPECTED_CHANNELS
) -> np.ndarray:
    """
    Load one EIT .txt file and return only the valid measurement matrix.

    Output shape:
        (N, 208)

    Use this function in most cases.
    """

    data, _ = load_txt_file_with_report(
        file_path=file_path,
        expected_channels=expected_channels
    )

    return data


def find_txt_files(root_dir: str | Path) -> list[Path]:
    """
    Find all .txt files inside the raw data folder recursively.

    Example:
        data/raw/Adjacent/circle/conductive/centre.txt
        data/raw/Opposite/rectangle/non-conductive/edge.txt
        data/raw/Skip3/triangle/onlywater.txt

    Returns:
        sorted list of Path objects
    """

    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, but got: {root_dir}")

    txt_files = sorted(root_dir.rglob("*.txt"))

    return txt_files


def ensure_directory(directory_path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Useful before saving CSV files into:
        data/processed/features/
        data/processed/summaries/
    """

    directory_path = Path(directory_path)
    directory_path.mkdir(parents=True, exist_ok=True)

    return directory_path


def save_dataframe_to_csv(
    df: pd.DataFrame,
    output_path: str | Path,
    index: bool = False
) -> None:
    """
    Save a pandas DataFrame to CSV.
    Automatically creates the parent folder if needed.
    """

    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    df.to_csv(output_path, index=index)