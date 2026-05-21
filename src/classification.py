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


def load_data(feature_csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(feature_csv_path)

    # Baseline is useful for SNR/distinguishability, not normal classification
    df = df[df["inclusion_type"] != "baseline"].copy()

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

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


def make_cv(y: pd.Series, n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    min_class_count = y.value_counts().min()
    n_splits = min(n_splits, min_class_count)

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def evaluate_model(df: pd.DataFrame, model, feature_columns: list[str]) -> dict:
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

    feature_sets = [
        ["bv_min"],
        ["bv_avg"],
        ["bv_range"],
        ["bv_avg_variation"],
        FEATURE_COLUMNS,
    ]

    rows = []

    for feature_columns in feature_sets:
        result = evaluate_model(
            df=df,
            model=model,
            feature_columns=feature_columns,
        )

        result["model"] = "Random Forest"
        rows.append(result)

    return pd.DataFrame(rows).sort_values(
        by="macro_f1",
        ascending=False,
    )


def compare_injection_patterns(df: pd.DataFrame) -> pd.DataFrame:
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )

    rows = []

    for injection_pattern, group_df in df.groupby("injection_pattern"):
        result = evaluate_model(
            df=group_df,
            model=model,
            feature_columns=FEATURE_COLUMNS,
        )

        result["injection_pattern"] = injection_pattern
        result["model"] = "Random Forest"
        rows.append(result)

    return pd.DataFrame(rows).sort_values(
        by="macro_f1",
        ascending=False,
    )


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

    model_results = compare_models(df)
    feature_results = compare_features(df)
    injection_results = compare_injection_patterns(df)

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

    save_confusion_matrix(
        df=df,
        output_path=results_figure_dir / "domain_shape_confusion_matrix_random_forest.png",
    )

    print("\nDomain-shape classification complete.")
    print("\nModel comparison:")
    print(model_results.to_string(index=False))

    print("\nFeature comparison:")
    print(feature_results.to_string(index=False))

    print("\nInjection-pattern comparison:")
    print(injection_results.to_string(index=False))