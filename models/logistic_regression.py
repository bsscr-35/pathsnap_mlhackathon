from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_model(
    C=1.0,
    max_iter=2000,
    class_weight=None,
    random_state=42
):
    """
    Build a Logistic Regression classification pipeline.

    Parameters:
        C: Inverse regularization strength.
        max_iter: Maximum number of iterations.
        class_weight: Optional class weighting.
        random_state: Random seed.

    Returns:
        A scikit-learn Pipeline.
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])

    return model