"""
PathSnap prediction pipeline.

Uses:
  1. CandidateGenerator to find nearby road segments.
  2. ViterbiDecoder to choose a likely connected sequence.
  3. The .route file only during evaluation.

IMPORTANT:
The current .route file contains a trip-level route, not necessarily a
GPS-point-aligned label sequence. Therefore `accuracy` below is route
membership/coverage (fraction of predictions belonging to the reference
route), NOT point-wise classification accuracy.
"""
import os
from pathlib import Path

import numpy as np

from map_matching.candidates import CandidateGenerator
from map_matching.viterbi import viterbi_decode

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(
    os.environ.get(
        "EVAL_DIR",
        ROOT / "data" / "Student folder" / "student_test",
    )
)

# Keep these consistent with map_matching/hmm_config.yaml.
HMM_CONFIG = {
    "emission_weight": 1.0,
    "transition_weight": 1.0,
    "disconnected_penalty": 12.0,
    "distance_scale": 100.0,
    "max_candidates_per_point": 5,
    "gap_threshold_seconds": 30.0,
}


def _trip_dir(trip_id: str) -> Path:
    trip_dir = EVAL_DIR / trip_id
    if not trip_dir.is_dir():
        raise FileNotFoundError(trip_id)
    return trip_dir


def list_trips():
    if not EVAL_DIR.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {EVAL_DIR}")
    return sorted(path.name for path in EVAL_DIR.iterdir() if path.is_dir())


def _load(trip_id):
    trip_dir = _trip_dir(trip_id)
    track = np.atleast_2d(
        np.loadtxt(trip_dir / f"{trip_id}.track", dtype=float)
    )
    nodes = np.atleast_2d(
        np.loadtxt(trip_dir / f"{trip_id}.nodes", dtype=float)
    )
    arcs = np.atleast_2d(
        np.loadtxt(trip_dir / f"{trip_id}.arcs", dtype=int)
    )

    if track.shape[1] < 3:
        raise ValueError(f"{trip_id}.track must contain lon, lat, timestamp")
    if nodes.shape[1] < 2 or arcs.shape[1] < 2:
        raise ValueError("Invalid nodes/arcs file format")

    return track, nodes, arcs


def _build_prediction(track, nodes_array, arcs_array):
    # Dataset IDs are row indices in the .nodes and .arcs files.
    nodes = {
        node_id: (float(row[0]), float(row[1]))
        for node_id, row in enumerate(nodes_array)
    }
    arcs = {
        arc_id: (int(row[0]), int(row[1]))
        for arc_id, row in enumerate(arcs_array)
    }

    gps_points = [
        (float(row[0]), float(row[1])) for row in track
    ]
    timestamps = [float(row[2]) for row in track]

    generator = CandidateGenerator(
        nodes=nodes,
        arcs=arcs,
        max_candidates=int(HMM_CONFIG["max_candidates_per_point"]),
        sigma=30.0,
    )
    candidates_per_point = generator.generate(gps_points)

    predictions = viterbi_decode(
        candidates_per_point=candidates_per_point,
        arcs=arcs,
        nodes=nodes,
        gps_points=gps_points,
        timestamps=timestamps,
        config=HMM_CONFIG,
    )

    if len(predictions) != len(track):
        raise RuntimeError(
            f"Prediction count {len(predictions)} does not match "
            f"GPS point count {len(track)}"
        )

    # Build the same output contract expected by backend.main and Streamlit.
    segments = []
    for arc_id in predictions:
        if arc_id is None or arc_id not in arcs:
            segments.append([])
            continue
        start_node, end_node = arcs[arc_id]
        start_lon, start_lat = nodes[start_node]
        end_lon, end_lat = nodes[end_node]
        segments.append([
            [float(start_lat), float(start_lon)],
            [float(end_lat), float(end_lon)],
        ])

    return predictions, segments, arcs


def predict_trip(trip_id: str) -> dict:
    track, nodes, arcs_array = _load(trip_id)
    predictions, segments, _ = _build_prediction(track, nodes, arcs_array)

    return {
        "points": [
            [float(row[1]), float(row[0])] for row in track
        ],
        "pred_arcs": predictions,
        "segments": segments,
        "arcs_table": arcs_array,
    }


def evaluate(trip_id: str, result: dict) -> dict:
    """Evaluate predictions against the trip-level reference route."""
    trip_dir = _trip_dir(trip_id)
    route_path = trip_dir / f"{trip_id}.route"
    route = np.atleast_1d(
        np.loadtxt(route_path, dtype=int)
    )
    reference_arcs = set(int(arc_id) for arc_id in route.tolist())

    predictions = result["pred_arcs"]
    arcs_array = result["arcs_table"]

    valid_predictions = [
        int(arc_id) for arc_id in predictions
        if arc_id is not None and 0 <= int(arc_id) < len(arcs_array)
    ]

    # This is route membership/coverage, not GPS-point-aligned accuracy.
    route_coverage = (
        sum(arc_id in reference_arcs for arc_id in valid_predictions)
        / len(predictions)
        if predictions else 0.0
    )

    direct_pairs = 0
    connected_pairs = 0
    for previous_id, current_id in zip(predictions, predictions[1:]):
        if previous_id is None or current_id is None:
            continue
        if not (
            0 <= int(previous_id) < len(arcs_array)
            and 0 <= int(current_id) < len(arcs_array)
        ):
            continue
        direct_pairs += 1
        previous_end = int(arcs_array[int(previous_id), 1])
        current_start = int(arcs_array[int(current_id), 0])
        if previous_end == current_start:
            connected_pairs += 1

    continuity = (
        connected_pairs / direct_pairs if direct_pairs else 0.0
    )

    return {
        # Kept as a 0..1 field because backend.schemas and Streamlit expect it.
        # It currently represents route coverage, not point-wise accuracy.
        "accuracy": round(float(route_coverage), 4),
        "route_continuity": round(float(continuity), 4),
        "n_points": len(predictions),
        "n_unique_predicted_arcs": len(set(valid_predictions)),
        "n_reference_arcs": len(reference_arcs),
    }
