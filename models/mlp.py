from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def build_model(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42
):
    """
    Build an MLP neural network classifier.

    Parameters:
        hidden_layer_sizes: Number of neurons in hidden layers.
        activation: Activation function.
        solver: Optimisation algorithm.
        alpha: L2 regularisation parameter.
        learning_rate_init: Initial learning rate.
        max_iter: Maximum training iterations.
        random_state: Random seed.

    Returns:
        Scikit-learn Pipeline.
    """

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            random_state=random_state
        ))
    ])

    return model