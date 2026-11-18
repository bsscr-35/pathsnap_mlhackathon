import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "distance_m", "search_radius_m", "gps_speed_mps", "segment_length_m",
    "heading_difference_deg", "gps_heading_sin", "gps_heading_cos",
    "segment_heading_sin", "segment_heading_cos",
]
CATEGORICAL_FEATURES = ["speed_quality"]


def engineer_features(frame):
    """Raw candidate rows -> model input. Needs gps_heading_deg and segment_heading_deg."""
    z = frame.copy()
    for c in ["gps_heading_deg", "segment_heading_deg"]:
        if c not in z.columns:
            z[c] = np.nan
    gps = np.deg2rad(pd.to_numeric(z["gps_heading_deg"], errors="coerce"))
    seg = np.deg2rad(pd.to_numeric(z["segment_heading_deg"], errors="coerce"))
    z["gps_heading_sin"], z["gps_heading_cos"] = np.sin(gps), np.cos(gps)
    z["segment_heading_sin"], z["segment_heading_cos"] = np.sin(seg), np.cos(seg)
    for c in NUMERIC_FEATURES:
        if c not in z.columns:
            z[c] = np.nan
    if "speed_quality" not in z.columns:
        z["speed_quality"] = "missing"
    z["speed_quality"] = z["speed_quality"].fillna("missing").astype(str)
    return z[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
