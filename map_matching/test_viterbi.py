
from .viterbi import viterbi_decode


def create_test_network():
    nodes = {
        0: (8.5900, 49.8400),
        1: (8.5910, 49.8400),
        2: (8.5920, 49.8400),
        3: (8.6000, 49.8500),
        4: (8.6010, 49.8500),
    }

    arcs = {
        0: (0, 1),
        1: (1, 2),
        2: (3, 4),
    }

    return nodes, arcs


def test_connected_route():
    nodes, arcs = create_test_network()

    candidates = [
        [
            {"arc_id": 0, "score": 0.9},
            {"arc_id": 2, "score": 0.99},
        ],
        [
            {"arc_id": 1, "score": 0.9},
            {"arc_id": 2, "score": 0.99},
        ],
    ]

    result = viterbi_decode(
        candidates,
        arcs,
        nodes,
        config={"disconnected_penalty": 100.0},
    )

    assert result == [0, 1]
    print("PASS: Connected route selected")


def test_missing_candidates():
    nodes, arcs = create_test_network()

    candidates = [
        [{"arc_id": 0, "score": 0.9}],
        [],
        [{"arc_id": 1, "score": 0.9}],
    ]

    result = viterbi_decode(candidates, arcs, nodes)

    assert len(result) == 3
    assert result[1] is None
    print("PASS: Missing candidates handled")


def test_single_gps_point():
    nodes, arcs = create_test_network()

    candidates = [
        [{"arc_id": 0, "score": 0.9}]
    ]

    result = viterbi_decode(candidates, arcs, nodes)

    assert result == [0]
    print("PASS: Single GPS point handled")


def test_empty_input():
    nodes, arcs = create_test_network()

    result = viterbi_decode([], arcs, nodes)

    assert result == []
    print("PASS: Empty input handled")


if __name__ == "__main__":
    test_connected_route()
    test_missing_candidates()
    test_single_gps_point()
    test_empty_input()

    print("\nAll Viterbi tests passed!")
