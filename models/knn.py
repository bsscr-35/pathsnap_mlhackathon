from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def build_model(
    n_neighbors=5,
    weights="distance",
    metric="minkowski"
):
    """
    Build a KNN classification pipeline.

    Parameters:
        n_neighbors: Number of nearest neighbors.
        weights: Weighting strategy.
        metric: Distance metric.

    Returns:
        Scikit-learn Pipeline.
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            n_jobs=-1
        ))
    ])

    return model