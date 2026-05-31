from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)

    if not path.exists():
        print(f"Skipping missing file: {path}")
        return None

    df = pd.read_csv(path)

    if df.empty:
        print(f"Skipping empty file: {path}")
        return None

    return df


def clean_feature_label(value: str) -> str:
    value = str(value)

    replacements = {
        "bv_min + bv_avg + bv_range + bv_avg_variation": "All features",
        "all_features": "All features",
        "bv_min": "BV min",
        "bv_avg": "BV avg",
        "bv_range": "BV range",
        "bv_avg_variation": "BV avg variation",
    }

    return replacements.get(value, value)


def save_metric_bar_plot(
    df: pd.DataFrame,
    label_column: str,
    metric_column: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: str | Path,
    sort_descending: bool = True,
) -> None:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    plot_df = df.copy()

    plot_df = plot_df.sort_values(
        by=metric_column,
        ascending=not sort_descending,
    )

    plot_df[metric_column] = plot_df[metric_column] * 100.0

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df[label_column], plot_df[metric_column])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 105)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def save_grouped_metric_bar_plot(
    df: pd.DataFrame,
    index_column: str,
    group_column: str,
    metric_column: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    pivot_df = df.pivot_table(
        index=index_column,
        columns=group_column,
        values=metric_column,
        aggfunc="mean",
    )

    pivot_df = pivot_df * 100.0

    ax = pivot_df.plot(kind="bar", figsize=(10, 6))

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 105)

    ax.legend(
        title=group_column.replace("_", " ").title(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")


def plot_model_comparison(
    table_path: str | Path,
    output_dir: str | Path,
) -> None:
    df = read_csv_if_exists(table_path)

    if df is None:
        return

    save_metric_bar_plot(
        df=df,
        label_column="model",
        metric_column="macro_f1",
        title="Domain Shape Classification: Model Comparison",
        x_label="Model",
        y_label="Macro F1 [%]",
        output_path=Path(output_dir) / "domain_shape_model_comparison_macro_f1.png",
    )

    save_metric_bar_plot(
        df=df,
        label_column="model",
        metric_column="accuracy",
        title="Domain Shape Classification: Model Accuracy",
        x_label="Model",
        y_label="Accuracy [%]",
        output_path=Path(output_dir) / "domain_shape_model_comparison_accuracy.png",
    )


def plot_feature_comparison(
    table_path: str | Path,
    output_dir: str | Path,
) -> None:
    df = read_csv_if_exists(table_path)

    if df is None:
        return

    df = df.copy()
    df["feature_label"] = df["features"].apply(clean_feature_label)

    save_metric_bar_plot(
        df=df,
        label_column="feature_label",
        metric_column="macro_f1",
        title="Domain Shape Classification: Feature Comparison",
        x_label="Feature set",
        y_label="Macro F1 [%]",
        output_path=Path(output_dir) / "domain_shape_feature_comparison_macro_f1.png",
    )

    save_metric_bar_plot(
        df=df,
        label_column="feature_label",
        metric_column="accuracy",
        title="Domain Shape Classification: Feature Accuracy",
        x_label="Feature set",
        y_label="Accuracy [%]",
        output_path=Path(output_dir) / "domain_shape_feature_comparison_accuracy.png",
    )


def plot_injection_pattern_comparison(
    table_path: str | Path,
    output_dir: str | Path,
) -> None:
    df = read_csv_if_exists(table_path)

    if df is None:
        return

    save_metric_bar_plot(
        df=df,
        label_column="injection_pattern",
        metric_column="macro_f1",
        title="Domain Shape Classification by Injection Pattern",
        x_label="Injection pattern",
        y_label="Macro F1 [%]",
        output_path=Path(output_dir) / "domain_shape_injection_pattern_macro_f1.png",
    )

    save_metric_bar_plot(
        df=df,
        label_column="injection_pattern",
        metric_column="accuracy",
        title="Domain Shape Classification Accuracy by Injection Pattern",
        x_label="Injection pattern",
        y_label="Accuracy [%]",
        output_path=Path(output_dir) / "domain_shape_injection_pattern_accuracy.png",
    )


def plot_models_by_injection_pattern(
    table_path: str | Path,
    output_dir: str | Path,
) -> None:
    df = read_csv_if_exists(table_path)

    if df is None:
        return

    save_grouped_metric_bar_plot(
        df=df,
        index_column="injection_pattern",
        group_column="model",
        metric_column="macro_f1",
        title="Model Comparison by Injection Pattern",
        x_label="Injection pattern",
        y_label="Macro F1 [%]",
        output_path=Path(output_dir) / "domain_shape_models_by_injection_pattern_macro_f1.png",
    )


def plot_features_by_injection_pattern(
    table_path: str | Path,
    output_dir: str | Path,
) -> None:
    df = read_csv_if_exists(table_path)

    if df is None:
        return

    df = df.copy()

    if "feature_set" in df.columns:
        df["feature_label"] = df["feature_set"].apply(clean_feature_label)
    else:
        df["feature_label"] = df["features"].apply(clean_feature_label)

    save_grouped_metric_bar_plot(
        df=df,
        index_column="feature_label",
        group_column="injection_pattern",
        metric_column="macro_f1",
        title="Feature Comparison by Injection Pattern",
        x_label="Feature set",
        y_label="Macro F1 [%]",
        output_path=Path(output_dir) / "domain_shape_features_by_injection_pattern_macro_f1.png",
    )