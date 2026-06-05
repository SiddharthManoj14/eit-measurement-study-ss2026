from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
SUMMARIES_DIR = PROJECT_ROOT / "data" / "processed" / "summaries"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

ENGINEERED_FEATURES_PATH = FEATURES_DIR / "engineered_features.csv"

MIN_VALID_VOLTAGE = 0.0
MAX_VALID_VOLTAGE = 5.0

EXPECTED_CHANNELS_BY_PATTERN = {
    "Adjacent": 208,
    "Opposite": 192,
    "Skip3": 192,
}

FEATURE_COLUMNS = [
    "bv_min",
    "bv_avg",
    "bv_range",
    "bv_avg_variation",
]

TARGET_COLUMN = "tank_shape"
GROUP_COLUMN = "injection_pattern"
RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5