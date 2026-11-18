
"""
PathSnap - Road Segment Candidate Generator
Person C

Finds nearby road segments for each GPS point using SciPy KDTree.
Uses distance-based scores until Person B's classifier is integrated.
"""

import math
import numpy as np
from scipy.spatial import cKDTree


class CandidateGenerator:
    def __init__(self, nodes, arcs, max_candidates=5, sigma=30.0):
        """
        nodes: {node_id: (longitude, latitude)}
        arcs: {arc_id: (start_node, end_node)}
        max_candidates: number of candidates per GPS point
        sigma: distance scale in metres for candidate scores
        """
        self.nodes = nodes
        self.arcs = arcs
        self.max_candidates = max_candidates
        self.sigma = sigma

        if not nodes or not arcs:
            raise ValueError("Nodes and arcs cannot be empty.")

        # Use a local coordinate system measured approximately in metres.
        coordinates = np.array(list(nodes.values()), dtype=float)
        self.reference_lon = float(np.mean(coordinates[:, 0]))
        self.reference_lat = float(np.mean(coordinates[:, 1]))

        self.cos_lat = math.cos(math.radians(self.reference_lat))
        self.node_xy = {
            node_id: self._project(lon, lat)
            for node_id, (lon, lat) in nodes.items()
        }

        self.arc_ids = []
        self.segments = []
        midpoints = []

        for arc_id, (start_node, end_node) in arcs.items():
            if start_node not in self.node_xy or end_node not in self.node_xy:
                continue

            start = self.node_xy[start_node]
            end = self.node_xy[end_node]

            self.arc_ids.append(arc_id)
            self.segments.append((start, end))

            midpoints.append([
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2,
            ])

        if not midpoints:
            raise ValueError("No valid road segments found.")

        self.midpoints = np.array(midpoints)
        self.tree = cKDTree(self.midpoints)

    def _project(self, lon, lat):
        """Convert longitude/latitude into approximate local metres."""
        x = (
            (lon - self.reference_lon)
            * 111320
            * self.cos_lat
        )
        y = (lat - self.reference_lat) * 110540
        return np.array([x, y], dtype=float)

    @staticmethod
    def _point_to_segment_distance(point, start, end):
        """Calculate shortest distance from a point to a line segment."""
        segment = end - start
        length_squared = float(np.dot(segment, segment))

        if length_squared == 0:
            return float(np.linalg.norm(point - start))

        t = float(np.dot(point - start, segment) / length_squared)
        t = max(0.0, min(1.0, t))

        closest = start + t * segment
        return float(np.linalg.norm(point - closest))

    def generate_for_point(self, gps_point):
        """Return the nearest road-segment candidates for one GPS point."""
        point = self._project(*gps_point)

        search_count = min(
            len(self.arc_ids),
            max(self.max_candidates * 4, 20),
        )

        _, indices = self.tree.query(point, k=search_count)
        indices = np.atleast_1d(indices)

        ranked = []

        for index in indices:
            start, end = self.segments[int(index)]

            distance = self._point_to_segment_distance(
                point, start, end
            )

            score = math.exp(
                -0.5 * (distance / self.sigma) ** 2
            )

            ranked.append({
                "arc_id": self.arc_ids[int(index)],
                "score": score,
                "distance_m": distance,
            })

        ranked.sort(key=lambda item: item["distance_m"])

        return ranked[:self.max_candidates]

    def generate(self, gps_points):
        """Generate candidates for every GPS point in a trip."""
        return [
            self.generate_for_point(point)
            for point in gps_points
        ]
