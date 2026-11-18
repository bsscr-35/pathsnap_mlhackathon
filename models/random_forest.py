from sklearn.ensemble import RandomForestClassifier


def build_model(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
):
    """
    Build a Random Forest classifier.

    Parameters:
        n_estimators: Number of decision trees.
        max_depth: Maximum depth of each tree.
        min_samples_split: Minimum samples required to split.
        min_samples_leaf: Minimum samples at a leaf.
        class_weight: Handles class imbalance.
        random_state: Reproducibility.
        n_jobs: Parallel processing.

    Returns:
        RandomForestClassifier.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )

    return model