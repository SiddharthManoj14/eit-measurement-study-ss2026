from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FEATURE_COLUMNS = [
    "bv_min",
    "bv_avg",
    "bv_range",
    "bv_avg_variation",
]

TARGET_COLUMN = "tank_shape"
GROUP_COLUMN = "injection_pattern"


def load_data(feature_csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(feature_csv_path)

    # Baseline is useful for SNR/distinguishability, not normal classification.
    df = df[df["inclusion_type"] != "baseline"].copy()

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN])

    return df


def get_models(random_state: int = 42) -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "SVM RBF": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def get_feature_sets() -> list[tuple[str, list[str]]]:
    return [
        ("all_features", FEATURE_COLUMNS),
        ("bv_range", ["bv_range"]),
        ("bv_avg_variation", ["bv_avg_variation"]),
        ("bv_avg", ["bv_avg"]),
        ("bv_min", ["bv_min"]),
    ]


def make_cv(
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> StratifiedKFold:
    min_class_count = y.value_counts().min()
    n_splits = min(n_splits, min_class_count)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def evaluate_model(
    df: pd.DataFrame,
    model,
    feature_columns: list[str],
    feature_set_name: str | None = None,
) -> dict:
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    cv = make_cv(y)

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
    )

    return {
        "feature_set": feature_set_name or " + ".join(feature_columns),
        "features": " + ".join(feature_columns),
        "n_samples": len(df),
        "cv_folds": cv.get_n_splits(),
        "accuracy": accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
    }


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, model in get_models().items():
        result = evaluate_model(
            df=df,
            model=model,
            feature_columns=FEATURE_COLUMNS,
            feature_set_name="all_features",
        )

        result["model"] = model_name
        rows.append(result)

    return pd.DataFrame(rows).sort_values(
        by="macro_f1",
        ascending=False,
    )


def compare_features(df: pd.DataFrame) -> pd.DataFrame:
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )

    rows = []

    for feature_set_name, feature_columns in get_feature_sets():
        result = evaluate_model(
            df=df,
            model=model,
            feature_columns=feature_columns,
            feature_set_name=feature_set_name,
        )

        result["model"] = "Random Forest"
        rows.append(result)

    return pd.DataFrame(rows).sort_values(
        by="macro_f1",
        ascending=False,
    )


def compare_injection_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary comparison:
    For each injection pattern, evaluate the best selected model
    using all engineered features.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )

    rows = []

    for injection_pattern, group_df in df.groupby(GROUP_COLUMN):
        result = evaluate_model(
            df=group_df,
            model=model,
            feature_columns=FEATURE_COLUMNS,
            feature_set_name="all_features",
        )

        result["injection_pattern"] = injection_pattern
        result["model"] = "Random Forest"
        rows.append(result)

    return pd.DataFrame(rows).sort_values(
        by="macro_f1",
        ascending=False,
    )


def compare_models_by_injection_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mejda point:
    Split the dataset by injection pattern and compare different models
    separately for Adjacent, Opposite, and Skip3.

    Uses all engineered features for each model.
    """

    rows = []

    for injection_pattern, group_df in df.groupby(GROUP_COLUMN):
        for model_name, model in get_models().items():
            result = evaluate_model(
                df=group_df,
                model=model,
                feature_columns=FEATURE_COLUMNS,
                feature_set_name="all_features",
            )

            result["injection_pattern"] = injection_pattern
            result["model"] = model_name
            rows.append(result)

    result_df = pd.DataFrame(rows)

    return result_df.sort_values(
        by=["injection_pattern", "macro_f1"],
        ascending=[True, False],
    )


def compare_features_by_injection_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mejda point:
    Split the dataset by injection pattern and run the same feature comparison
    separately for Adjacent, Opposite, and Skip3.

    Uses Random Forest because it was the best overall model.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )

    rows = []

    for injection_pattern, group_df in df.groupby(GROUP_COLUMN):
        for feature_set_name, feature_columns in get_feature_sets():
            result = evaluate_model(
                df=group_df,
                model=model,
                feature_columns=feature_columns,
                feature_set_name=feature_set_name,
            )

            result["injection_pattern"] = injection_pattern
            result["model"] = "Random Forest"
            rows.append(result)

    result_df = pd.DataFrame(rows)

    return result_df.sort_values(
        by=["injection_pattern", "macro_f1"],
        ascending=[True, False],
    )


def compare_full_logic_by_injection_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete comparison:
    For each injection pattern, evaluate every model with every feature set.

    This gives:
        3 injection patterns x 3 models x 5 feature sets = 45 rows
    """

    rows = []

    for injection_pattern, group_df in df.groupby(GROUP_COLUMN):
        for model_name, model in get_models().items():
            for feature_set_name, feature_columns in get_feature_sets():
                result = evaluate_model(
                    df=group_df,
                    model=model,
                    feature_columns=feature_columns,
                    feature_set_name=feature_set_name,
                )

                result["injection_pattern"] = injection_pattern
                result["model"] = model_name
                rows.append(result)

    result_df = pd.DataFrame(rows)

    return result_df.sort_values(
        by=["injection_pattern", "model", "macro_f1"],
        ascending=[True, True, False],
    )


def save_injection_pattern_subsets(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save three split CSV files:
        Adjacent
        Opposite
        Skip3

    This keeps a physical record of the per-injection datasets used
    for separate classification.
    """

    output_dir = Path(output_dir)
    split_dir = output_dir / "domain_shape_injection_pattern_subsets"
    split_dir.mkdir(parents=True, exist_ok=True)

    for injection_pattern, group_df in df.groupby(GROUP_COLUMN):
        safe_name = (
            str(injection_pattern)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        output_file = split_dir / f"domain_shape_{safe_name}.csv"
        group_df.to_csv(output_file, index=False)

        print(f"Saved injection-pattern subset: {output_file}")


def save_confusion_matrix(df: pd.DataFrame, output_path: str | Path) -> None:
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    cv = make_cv(y)

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
    )

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d")
    ax.set_title("Domain Shape Classification - Random Forest")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_domain_shape_classification(
    feature_csv_path: str | Path,
    results_table_dir: str | Path,
    results_figure_dir: str | Path,
) -> None:
    results_table_dir = Path(results_table_dir)
    results_figure_dir = Path(results_figure_dir)

    results_table_dir.mkdir(parents=True, exist_ok=True)
    results_figure_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(feature_csv_path)

    save_injection_pattern_subsets(
        df=df,
        output_dir=results_table_dir,
    )

    model_results = compare_models(df)
    feature_results = compare_features(df)
    injection_results = compare_injection_patterns(df)

    models_by_injection_results = compare_models_by_injection_pattern(df)
    features_by_injection_results = compare_features_by_injection_pattern(df)
    full_logic_by_injection_results = compare_full_logic_by_injection_pattern(df)

    model_results.to_csv(
        results_table_dir / "domain_shape_model_results.csv",
        index=False,
    )

    feature_results.to_csv(
        results_table_dir / "domain_shape_feature_comparison.csv",
        index=False,
    )

    injection_results.to_csv(
        results_table_dir / "domain_shape_by_injection_pattern.csv",
        index=False,
    )

    models_by_injection_results.to_csv(
        results_table_dir / "domain_shape_models_by_injection_pattern.csv",
        index=False,
    )

    features_by_injection_results.to_csv(
        results_table_dir / "domain_shape_features_by_injection_pattern.csv",
        index=False,
    )

    full_logic_by_injection_results.to_csv(
        results_table_dir / "domain_shape_full_logic_by_injection_pattern.csv",
        index=False,
    )

    save_confusion_matrix(
        df=df,
        output_path=results_figure_dir
        / "domain_shape_confusion_matrix_random_forest.png",
    )

    print("\nDomain-shape classification complete.")

    print("\nModel comparison:")
    print(model_results.to_string(index=False))

    print("\nFeature comparison:")
    print(feature_results.to_string(index=False))

    print("\nInjection-pattern comparison:")
    print(injection_results.to_string(index=False))

    print("\nModel comparison by injection pattern:")
    print(models_by_injection_results.to_string(index=False))

    print("\nFeature comparison by injection pattern:")
    print(features_by_injection_results.to_string(index=False))

    print("\nFull logic by injection pattern:")
    print(full_logic_by_injection_results.to_string(index=False))