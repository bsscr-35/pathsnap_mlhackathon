import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from models.preprocessing import engineer_features, build_preprocessor

from models.logistic_regression import build_model as logistic_regression
from models.knn import build_model as knn
from models.random_forest import build_model as random_forest
from models.xgboost_model import build_model as xgboost
from models.mlp import build_model as mlp


MODEL_BUILDERS = {
    "Logistic Regression": logistic_regression,
    "KNN": knn,
    "Random Forest": random_forest,
    "XGBoost": xgboost,
    "MLP": mlp,
}

LABEL_CANDIDATES = [
    "label",
    "candidate_label",
    "is_correct",
    "target",
]


def load_data(path, label_column=None):

    df = pd.read_parquet(path)

    if label_column is None:
        label_column = next(
            (col for col in LABEL_CANDIDATES if col in df.columns),
            None
        )

    if label_column is None or label_column not in df.columns:
        raise ValueError(
            "Candidate labels not found. "
            "Wait for Person C to provide the labelled dataset."
        )

    df = df.dropna(subset=[label_column]).copy()

    y = df[label_column]
    groups = df["Trip_ID"].astype(str)

    X = engineer_features(df)

    excluded = [
        "gps_index",
        "timestamp",
        "arc_id",
        "Trip_ID",
        "Partition",
        label_column,
    ]

    X = X.drop(
        columns=[col for col in excluded if col in X.columns]
    )

    return X, y, groups, label_column


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--folds", type=int, default=5)

    args = parser.parse_args()

    output = Path("results")
    output.mkdir(exist_ok=True)

    X, y, groups, label_column = load_data(
        args.data,
        args.label
    )

    cv = GroupKFold(
        n_splits=min(args.folds, groups.nunique())
    )

    results = []

    for name, builder in MODEL_BUILDERS.items():

        print(f"\nTraining {name}...")

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y, groups), start=1
        ):

            model = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", builder()),
            ])

            start = time.time()

            model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx]
            )

            train_time = time.time() - start

            start = time.time()

            predictions = model.predict(X.iloc[val_idx])

            predict_time = time.time() - start

            accuracy = accuracy_score(
                y.iloc[val_idx],
                predictions
            )

            macro_f1 = f1_score(
                y.iloc[val_idx],
                predictions,
                average="macro"
            )

            probabilities = model.predict_proba(
                X.iloc[val_idx]
            )

            loss = log_loss(
                y.iloc[val_idx],
                probabilities,
                labels=model.named_steps["classifier"].classes_
            )

            results.append({
                "model": name,
                "fold": fold,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "log_loss": loss,
                "training_time": train_time,
                "prediction_time": predict_time,
            })

            print(
                f"Fold {fold}: "
                f"Accuracy={accuracy:.4f}, "
                f"Macro F1={macro_f1:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output / "model_comparison_folds.csv",
        index=False
    )

    summary = results_df.groupby("model").agg({
        "accuracy": "mean",
        "macro_f1": "mean",
        "log_loss": "mean",
        "training_time": "mean",
        "prediction_time": "mean",
    }).sort_values(
        "macro_f1",
        ascending=False
    )

    summary.to_csv(
        output / "model_comparison.csv"
    )

    metadata = {
        "label_column": label_column,
        "training_rows": len(X),
        "number_of_trips": int(groups.nunique()),
        "cross_validation_folds": args.folds,
        "models": list(MODEL_BUILDERS.keys()),
    }

    with open(output / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nMODEL COMPARISON")
    print(summary)

    print("\nResults saved in the results folder.")


if __name__ == "__main__":
    main()