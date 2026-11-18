import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = [
    "distance_m",
    "search_radius_m",
    "gps_speed_mps",
    "gps_heading_sin",
    "gps_heading_cos",
    "segment_length_m",
    "segment_heading_sin",
    "segment_heading_cos",
    "heading_difference_deg",
]

CATEGORICAL_FEATURES = [
    "speed_quality",
]


def engineer_features(df):
    df = df.copy()

    gps_heading = np.deg2rad(df["gps_heading_deg"])
    segment_heading = np.deg2rad(df["segment_heading_deg"])

    df["gps_heading_sin"] = np.sin(gps_heading)
    df["gps_heading_cos"] = np.cos(gps_heading)

    df["segment_heading_sin"] = np.sin(segment_heading)
    df["segment_heading_cos"] = np.cos(segment_heading)

    return df


def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor