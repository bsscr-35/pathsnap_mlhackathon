from xgboost import XGBClassifier


def build_model(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
):
    """
    Build an XGBoost classifier.

    Parameters:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Learning rate.
        subsample: Fraction of training samples per tree.
        colsample_bytree: Fraction of features per tree.
        random_state: Random seed.
        n_jobs: Parallel processing.

    Returns:
        XGBClassifier.
    """

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=n_jobs
    )

    return model