"""
Prediction pipeline used by the API (runs on Laptop 1).

CONTRACT so teammates can plug in their parts:
    predict_trip(trip_id) -> dict with
        points   : list of [lat, lon]               raw GPS points
        pred_arcs: list[int]                        predicted arc_id per GPS point
        segments : list of [[lat,lon],[lat,lon]]    geometry of each predicted arc
        arcs_table: ndarray (server-side only)      used by evaluate(), removed before returning

CURRENT STATE: stand-in baseline = nearest road segment to each GPS point (uses NO labels).
Replace `_nearest_arcs` with:
    Person A -> candidate generation + features
    Person B -> model.predict_proba
    Person C -> viterbi_decode
Ground truth (.route) is read only inside evaluate() and is never returned to the UI.
"""
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Dev: student_test. Final evaluation: set env var EVAL_DIR to the evaluator folder (no code change).
EVAL_DIR = Path(os.environ.get("EVAL_DIR", ROOT / "data" / "Student folder" / "student_test"))


def _trip_dir(trip_id: str) -> Path:
    d = EVAL_DIR / trip_id
    if not d.is_dir():
        raise FileNotFoundError(trip_id)
    return d


def list_trips():
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir())


def _load(trip_id):
    d = _trip_dir(trip_id)
    track = np.loadtxt(d / f"{trip_id}.track")            # lon, lat, timestamp
    nodes = np.loadtxt(d / f"{trip_id}.nodes")            # lon, lat
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=int)   # from_node, to_node
    return track, nodes, arcs


def _to_xy(lon, lat, lat0):
    """Local metric projection (equirectangular), fine at city scale."""
    R = 6371000.0
    return np.radians(lon) * R * np.cos(np.radians(lat0)), np.radians(lat) * R


def _nearest_arcs(track, nodes, arcs):
    lat0 = track[:, 1].mean()
    px, py = _to_xy(track[:, 0], track[:, 1], lat0)
    nx, ny = _to_xy(nodes[:, 0], nodes[:, 1], lat0)
    ax, ay = nx[arcs[:, 0]], ny[arcs[:, 0]]
    bx, by = nx[arcs[:, 1]], ny[arcs[:, 1]]
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    out = np.empty(len(px), dtype=int)
    for i in range(len(px)):
        t = np.clip(((px[i] - ax) * dx + (py[i] - ay) * dy) / L2, 0, 1)
        d2 = (px[i] - (ax + t * dx)) ** 2 + (py[i] - (ay + t * dy)) ** 2
        out[i] = int(np.argmin(d2))
    return out


def predict_trip(trip_id: str) -> dict:
    track, nodes, arcs = _load(trip_id)
    pred = _nearest_arcs(track, nodes, arcs)               # <-- replace with the real pipeline
    segments = [[[float(nodes[arcs[a, 0], 1]), float(nodes[arcs[a, 0], 0])],
                 [float(nodes[arcs[a, 1], 1]), float(nodes[arcs[a, 1], 0])]] for a in pred]
    return {
        "points": [[float(la), float(lo)] for lo, la in track[:, :2]],
        "pred_arcs": pred.tolist(),
        "segments": segments,
        "arcs_table": arcs,
    }


def evaluate(trip_id: str, result: dict) -> dict:
    """Server-side only: reads ground truth and computes metrics."""
    d = _trip_dir(trip_id)
    route = np.loadtxt(d / f"{trip_id}.route", dtype=int, ndmin=1)
    truth = set(route.tolist())
    pred, arcs = result["pred_arcs"], result["arcs_table"]
    on_route = float(np.mean([a in truth for a in pred]))
    connected = lambda a, b: a == b or len({arcs[a, 0], arcs[a, 1]} & {arcs[b, 0], arcs[b, 1]}) > 0
    cont = sum(connected(a, b) for a, b in zip(pred, pred[1:])) / max(len(pred) - 1, 1)
    return {
        "n_points": len(pred),
        "accuracy": round(on_route, 4),          # placeholder definition, see README
        "route_continuity": round(float(cont), 4),
        "n_unique_predicted_arcs": len(set(pred)),
        "n_reference_arcs": len(truth),
    }
