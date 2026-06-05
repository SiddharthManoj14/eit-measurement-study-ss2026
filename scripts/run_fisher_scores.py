from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    ENGINEERED_FEATURES_PATH,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
    TARGET_COLUMN,
    GROUP_COLUMN,
    FEATURE_COLUMNS,
)

from src.metrics import compute_fisher_scores



def infer_injection_pattern_from_filename(file_path: Path) -> str:
    name = file_path.stem.lower()

    if "adjacent" in name:
        return "Adjacent"

    if "opposite" in name:
        return "Opposite"

    if "skip" in name:
        return "Skip3"

    return "unknown"

def load_feature_data() -> pd.DataFrame:
    input_file = ENGINEERED_FEATURES_PATH

    if not input_file.exists():
        raise FileNotFoundError(
            f"Expected file not found: {input_file}"
        )

    df = pd.read_csv(input_file)

    if df.empty:
        raise ValueError(f"Input file is empty: {input_file}")

    df["source_csv"] = input_file.name

    return df


def validate_columns(df: pd.DataFrame) -> list[str]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}\n"
            f"Available columns: {list(df.columns)}"
        )

    available_features = [
        feature for feature in FEATURE_COLUMNS
        if feature in df.columns
    ]

    missing_features = [
        feature for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:
        print("Warning: these feature columns are missing:")
        for feature in missing_features:
            print(f"  - {feature}")

    if not available_features:
        raise ValueError(
            "None of the expected feature columns were found.\n"
            f"Expected: {FEATURE_COLUMNS}\n"
            f"Available columns: {list(df.columns)}"
        )

    return available_features


def save_overall_fisher_scores(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    fisher_scores = compute_fisher_scores(
        df=df,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )

    fisher_scores.insert(0, "target", TARGET_COLUMN)
    fisher_scores.insert(1, "scope", "overall")
    fisher_scores.insert(2, "n_samples", len(df))
    fisher_scores.insert(3, "n_classes", df[TARGET_COLUMN].nunique())

    output_file = RESULTS_TABLE_DIR / "fisher_scores_domain_shape_overall.csv"
    fisher_scores.to_csv(output_file, index=False)

    print(f"Saved overall Fisher scores: {output_file}")

    return fisher_scores

""" Compute per-injection Fisher scores """
def save_fisher_scores_by_injection_pattern(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    all_scores = []

    for injection_pattern, group_df in df.groupby(GROUP_COLUMN):
        if group_df[TARGET_COLUMN].nunique() < 2:
            print(
                f"Skipping {injection_pattern}: "
                f"needs at least 2 classes for Fisher score."
            )
            continue

        fisher_scores = compute_fisher_scores(
            df=group_df,
            feature_columns=feature_columns,
            target_column=TARGET_COLUMN,
        )

        fisher_scores.insert(0, "target", TARGET_COLUMN)
        fisher_scores.insert(1, "injection_pattern", injection_pattern)
        fisher_scores.insert(2, "n_samples", len(group_df))
        fisher_scores.insert(3, "n_classes", group_df[TARGET_COLUMN].nunique())

        all_scores.append(fisher_scores)

    if not all_scores:
        raise ValueError("No per-injection Fisher scores could be computed.")

    result = pd.concat(all_scores, ignore_index=True)

    output_file = RESULTS_TABLE_DIR / "fisher_scores_domain_shape_by_injection.csv"
    result.to_csv(output_file, index=False)

    print(f"Saved per-injection Fisher scores: {output_file}")

    return result


def plot_overall_fisher_scores(overall_scores: pd.DataFrame) -> None:
    plot_df = overall_scores.sort_values("fisher_score", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["feature"], plot_df["fisher_score"])

    plt.title("Overall Fisher Scores for Domain Shape Classification")
    plt.xlabel("Fisher score")
    plt.ylabel("Feature")
    plt.tight_layout()

    output_file = RESULTS_FIGURE_DIR / "fisher_scores_domain_shape_overall.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved overall Fisher score plot: {output_file}")


def plot_fisher_scores_by_injection(per_injection_scores: pd.DataFrame) -> None:
    pivot_df = per_injection_scores.pivot_table(
        index="feature",
        columns="injection_pattern",
        values="fisher_score",
        aggfunc="mean",
    )

    pivot_df = pivot_df.loc[
        pivot_df.mean(axis=1).sort_values(ascending=False).index
    ]

    ax = pivot_df.plot(kind="bar", figsize=(10, 6))

    ax.set_title("Fisher Scores by Injection Pattern")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Fisher score")
    ax.legend(title="Injection pattern")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_file = RESULTS_FIGURE_DIR / "fisher_scores_domain_shape_by_injection.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved per-injection Fisher score plot: {output_file}")


def main() -> None:
    RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_feature_data()

    df = df[df["inclusion_type"] != "baseline"].copy()

    print("Loaded feature dataset:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    feature_columns = validate_columns(df)

    df = df.dropna(subset=feature_columns + [TARGET_COLUMN])

    overall_scores = save_overall_fisher_scores(
        df=df,
        feature_columns=feature_columns,
    )

    per_injection_scores = save_fisher_scores_by_injection_pattern(
        df=df,
        feature_columns=feature_columns,
    )

    plot_overall_fisher_scores(overall_scores)
    plot_fisher_scores_by_injection(per_injection_scores)

    print("\nFisher score computation complete.")


if __name__ == "__main__":
    main()