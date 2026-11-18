
"""
PathSnap - HMM and Viterbi Decoder
Person C implementation

Hidden states:
    Candidate road segments for each GPS point.

Emission probability:
    Candidate score from Person B's classifier.
    Distance-based scores can be used as a fallback.

Transition probability:
    Directed road-network distance compared with GPS movement.

Optimisation:
    Multi-target Dijkstra calculates distances to multiple
    candidate nodes in one search.

GPS gap handling:
    Long gaps split the observations into independent
    Viterbi decoding segments.
"""

import math
import heapq


def haversine(lon1, lat1, lon2, lat2):
    """Calculate straight-line distance between coordinates in metres."""

    radius = 6371000

    lon1, lat1, lon2, lat2 = map(
        math.radians,
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


class ViterbiDecoder:

    def __init__(
        self,
        arcs,
        nodes,
        emission_weight=1.0,
        transition_weight=1.0,
        disconnected_penalty=12.0,
        distance_scale=100.0,
    ):

        self.arcs = arcs
        self.nodes = nodes

        self.emission_weight = emission_weight
        self.transition_weight = transition_weight
        self.disconnected_penalty = disconnected_penalty
        self.distance_scale = max(1.0, distance_scale)

        self.graph = {}
        self._distance_cache = {}

        # Build directed road graph.
        for start, end in arcs.values():

            self.graph.setdefault(start, [])

            if start in nodes and end in nodes:

                lon1, lat1 = nodes[start]
                lon2, lat2 = nodes[end]

                length = haversine(
                    lon1, lat1, lon2, lat2
                )

            else:
                length = 1.0

            self.graph[start].append((end, length))

    def shortest_distances(self, source, targets):
        """
        Find shortest directed distances from one source
        to multiple target nodes using Dijkstra's algorithm.
        """

        targets = set(targets)

        results = {}
        unresolved = set()

        for target in targets:

            key = (source, target)

            if key in self._distance_cache:

                results[target] = self._distance_cache[key]

            elif source == target:

                results[target] = 0.0
                self._distance_cache[key] = 0.0

            else:

                unresolved.add(target)

        if not unresolved:
            return results

        distances = {source: 0.0}
        queue = [(0.0, source)]

        while queue and unresolved:

            current_distance, current = heapq.heappop(queue)

            if current_distance > distances.get(
                current, float("inf")
            ):
                continue

            if current in unresolved:

                results[current] = current_distance

                self._distance_cache[
                    (source, current)
                ] = current_distance

                unresolved.remove(current)

            for neighbour, edge_length in self.graph.get(
                current, []
            ):

                new_distance = current_distance + edge_length

                if new_distance < distances.get(
                    neighbour, float("inf")
                ):

                    distances[neighbour] = new_distance

                    heapq.heappush(
                        queue,
                        (new_distance, neighbour)
                    )

        # Cache unreachable targets.
        for target in unresolved:

            results[target] = None

            self._distance_cache[
                (source, target)
            ] = None

        return results

    def shortest_distance(self, source, target):
        """Find shortest directed distance between two nodes."""

        return self.shortest_distances(
            source,
            [target]
        )[target]

    def emission_log_probability(self, candidate):
        """Convert candidate score into a log probability."""

        score = candidate.get("score", 1e-9)

        score = max(
            1e-9,
            min(1.0, float(score))
        )

        return self.emission_weight * math.log(score)

    def transition_log_probability(
        self,
        previous_arc_id,
        current_arc_id,
        gps_distance=None,
        road_distance=None,
    ):
        """Calculate transition score between two road segments."""

        previous_arc = self.arcs.get(previous_arc_id)
        current_arc = self.arcs.get(current_arc_id)

        if previous_arc is None or current_arc is None:

            return -self.disconnected_penalty

        previous_end = previous_arc[1]
        current_start = current_arc[0]

        if road_distance is None:

            road_distance = self.shortest_distance(
                previous_end,
                current_start
            )

        if road_distance is None:

            return -self.disconnected_penalty

        if gps_distance is None:
            gps_distance = 0.0

        difference = abs(
            road_distance - gps_distance
        )

        return -self.transition_weight * (
            difference / self.distance_scale
        )

    def _decode_contiguous(
        self,
        candidates_per_point,
        gps_points=None,
    ):
        """
        Run Viterbi decoding on one continuous GPS sequence.
        """

        if not candidates_per_point:
            return []

        # Handle missing candidate sets.
        valid_points = [
            candidates if candidates else [
                {
                    "arc_id": None,
                    "score": 1e-9
                }
            ]
            for candidates in candidates_per_point
        ]

        # Initial time step.
        first_scores = {}

        for candidate in valid_points[0]:

            arc_id = candidate.get("arc_id")

            first_scores[arc_id] = (
                self.emission_log_probability(candidate)
            )

        scores = [first_scores]
        backpointers = [{}]

        # Process subsequent GPS observations.
        for index in range(1, len(valid_points)):

            previous_states = scores[-1]
            current_candidates = valid_points[index]

            current_scores = {}
            current_backpointers = {}

            gps_distance = None

            if gps_points is not None:

                lon1, lat1 = gps_points[index - 1]
                lon2, lat2 = gps_points[index]

                gps_distance = haversine(
                    lon1,
                    lat1,
                    lon2,
                    lat2
                )

            # Collect target nodes for each previous candidate.
            targets_by_previous = {}

            for previous_arc_id in previous_states:

                if previous_arc_id is None:
                    continue

                previous_arc = self.arcs.get(
                    previous_arc_id
                )

                if previous_arc is not None:

                    targets_by_previous[
                        previous_arc_id
                    ] = previous_arc[1]

            # Run multi-target Dijkstra searches.
            distance_lookup = {}

            for previous_arc_id, source in (
                targets_by_previous.items()
            ):

                target_nodes = []

                for candidate in current_candidates:

                    current_arc_id = candidate.get("arc_id")

                    if current_arc_id is None:
                        continue

                    current_arc = self.arcs.get(
                        current_arc_id
                    )

                    if current_arc is not None:
                        target_nodes.append(current_arc[0])

                distance_lookup[previous_arc_id] = (
                    self.shortest_distances(
                        source,
                        target_nodes
                    )
                )

            # Find best previous state for every current candidate.
            for candidate in current_candidates:

                current_arc_id = candidate.get("arc_id")

                emission = self.emission_log_probability(
                    candidate
                )

                best_score = float("-inf")
                best_previous = None

                for previous_arc_id, previous_score in (
                    previous_states.items()
                ):

                    if (
                        previous_arc_id is None
                        or current_arc_id is None
                    ):

                        transition = (
                            -self.disconnected_penalty
                        )

                    else:

                        current_arc = self.arcs.get(
                            current_arc_id
                        )

                        previous_arc = self.arcs.get(
                            previous_arc_id
                        )

                        if (
                            current_arc is None
                            or previous_arc is None
                        ):

                            road_distance = None

                        else:

                            target = current_arc[0]

                            road_distance = (
                                distance_lookup
                                .get(previous_arc_id, {})
                                .get(target)
                            )

                        transition = (
                            self.transition_log_probability(
                                previous_arc_id,
                                current_arc_id,
                                gps_distance,
                                road_distance
                            )
                        )

                    total = (
                        previous_score
                        + transition
                        + emission
                    )

                    if total > best_score:

                        best_score = total
                        best_previous = previous_arc_id

                current_scores[current_arc_id] = best_score

                current_backpointers[
                    current_arc_id
                ] = best_previous

            scores.append(current_scores)
            backpointers.append(current_backpointers)

        # Backtrack to recover the best path.
        last_state = max(
            scores[-1],
            key=scores[-1].get
        )

        best_path = [last_state]

        for index in range(
            len(valid_points) - 1,
            0,
            -1
        ):

            last_state = backpointers[index][last_state]

            best_path.append(last_state)

        best_path.reverse()

        return best_path

    def decode(
        self,
        candidates_per_point,
        gps_points=None,
        timestamps=None,
        gap_threshold_seconds=30.0,
    ):
        """
        Run timestamp-aware Viterbi decoding.

        Long GPS gaps split the observations into
        independent decoding segments.

        Every original GPS observation receives one prediction.
        """

        if not candidates_per_point:
            return []

        total_points = len(candidates_per_point)

        # Validate input lengths.
        if (
            gps_points is not None
            and len(gps_points) != total_points
        ):

            raise ValueError(
                "GPS points and candidate sequences "
                "must have equal lengths."
            )

        if (
            timestamps is not None
            and len(timestamps) != total_points
        ):

            raise ValueError(
                "Timestamps and candidate sequences "
                "must have equal lengths."
            )

        if gap_threshold_seconds <= 0:

            raise ValueError(
                "Gap threshold must be positive."
            )

        # Identify independent decoding segments.
        segment_starts = [0]

        if timestamps is not None:

            for index in range(1, total_points):

                gap = (
                    timestamps[index]
                    - timestamps[index - 1]
                )

                # Split on long or non-increasing time intervals.
                if (
                    gap <= 0
                    or gap > gap_threshold_seconds
                ):

                    segment_starts.append(index)

        segment_starts.append(total_points)

        final_path = []

        # Decode each segment independently.
        for segment_index in range(
            len(segment_starts) - 1
        ):

            start = segment_starts[segment_index]
            end = segment_starts[segment_index + 1]

            segment_candidates = (
                candidates_per_point[start:end]
            )

            segment_gps = (
                gps_points[start:end]
                if gps_points is not None
                else None
            )

            segment_path = self._decode_contiguous(
                segment_candidates,
                segment_gps
            )

            final_path.extend(segment_path)

        return final_path


def viterbi_decode(
    candidates_per_point,
    arcs,
    nodes,
    gps_points=None,
    config=None,
    timestamps=None,
):
    """
    Public function for team integration.

    Example:

        path = viterbi_decode(
            candidates,
            arcs,
            nodes,
            gps_points,
            config,
            timestamps
        )
    """

    config = config or {}

    decoder = ViterbiDecoder(
        arcs=arcs,
        nodes=nodes,

        emission_weight=config.get(
            "emission_weight",
            1.0
        ),

        transition_weight=config.get(
            "transition_weight",
            1.0
        ),

        disconnected_penalty=config.get(
            "disconnected_penalty",
            12.0
        ),

        distance_scale=config.get(
            "distance_scale",
            100.0
        ),
    )

    return decoder.decode(
        candidates_per_point,
        gps_points,

        timestamps=timestamps,

        gap_threshold_seconds=config.get(
            "gap_threshold_seconds",
            30.0
        ),
    )
