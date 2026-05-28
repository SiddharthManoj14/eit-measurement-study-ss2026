import pandas as pd


def compute_fisher_scores(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute Fisher scores for each feature.

    Fisher score measures class separability:
    high between-class variation and low within-class variation gives a high score.
    """

    scores = []
    overall_means = df[feature_columns].mean()

    for feature in feature_columns:
        numerator = 0.0
        denominator = 0.0

        for class_label, class_df in df.groupby(target_column):
            n_c = len(class_df)

            class_mean = class_df[feature].mean()
            class_variance = class_df[feature].var(ddof=0)

            numerator += n_c * (class_mean - overall_means[feature]) ** 2
            denominator += n_c * class_variance

        fisher_score = numerator / (denominator + eps)

        scores.append(
            {
                "feature": feature,
                "fisher_score": fisher_score,
            }
        )

    result = pd.DataFrame(scores)
    result = result.sort_values("fisher_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result