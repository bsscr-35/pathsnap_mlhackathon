
from pathlib import Path


def load_trip(trip_dir):
    """
    Load one PathSnap trip.

    Returns:
        gps_points: [(longitude, latitude), ...]
        timestamps: [time, ...]
        nodes: {node_id: (longitude, latitude)}
        arcs: {arc_id: (start_node, end_node)}
        ground_truth: [arc_id, ...]
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    def read_rows(suffix):
        file_path = trip_dir / f"{trip_id}.{suffix}"
        with file_path.open("r", encoding="utf-8") as file:
            return [
                line.split()
                for line in file
                if line.strip()
            ]

    track_rows = read_rows("track")
    node_rows = read_rows("nodes")
    arc_rows = read_rows("arcs")
    route_rows = read_rows("route")

    gps_points = [
        (float(row[0]), float(row[1]))
        for row in track_rows
    ]

    timestamps = [
        float(row[2])
        for row in track_rows
    ]

    nodes = {
        node_id: (float(row[0]), float(row[1]))
        for node_id, row in enumerate(node_rows)
    }

    arcs = {
        arc_id: (int(row[0]), int(row[1]))
        for arc_id, row in enumerate(arc_rows)
    }

    ground_truth = [
        int(row[0])
        for row in route_rows
    ]

    # Validate the ground-truth route.
    for arc_id in ground_truth:
        if arc_id not in arcs:
            raise ValueError(f"Invalid route arc ID: {arc_id}")

    return {
        "trip_id": trip_id,
        "gps_points": gps_points,
        "timestamps": timestamps,
        "nodes": nodes,
        "arcs": arcs,
        "ground_truth": ground_truth,
    }
