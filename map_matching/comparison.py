
"""
PathSnap - HMM Evaluation and Comparison
Person C

Compares:
1. Classifier-only predictions
2. Classifier + HMM predictions

Evaluation metrics:
1. Pointwise accuracy
2. Sequence-level accuracy
3. Direct route continuity
4. Network reachability

Important:
Pointwise accuracy requires GPS-aligned ground-truth labels.
The .route file alone is not automatically aligned with GPS observations.
"""

from collections import deque


def pointwise_accuracy(predictions, ground_truth):
    """Calculate accuracy when both sequences are properly aligned."""

    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Sequence length mismatch: {len(predictions)} predictions "
            f"and {len(ground_truth)} ground-truth labels. "
            "Provide validated GPS-aligned ground truth."
        )

    if not ground_truth:
        return 0.0

    correct = sum(
        predicted == actual
        for predicted, actual in zip(predictions, ground_truth)
    )

    return correct / len(ground_truth) * 100


def sequence_accuracy(predictions, ground_truth):
    """
    Calculate the percentage of complete trips whose predicted
    sequences exactly match their ground-truth sequences.

    Each item in predictions and ground_truth represents
    one complete, aligned trip sequence.
    """

    if len(predictions) != len(ground_truth):
        raise ValueError(
            "Number of predicted and ground-truth "
            "sequences must match."
        )

    if not ground_truth:
        return 0.0

    correct = 0

    for predicted, actual in zip(predictions, ground_truth):

        if len(predicted) != len(actual):
            continue

        if list(predicted) == list(actual):
            correct += 1

    return correct / len(ground_truth) * 100


def route_continuity(predictions, arcs):
    """Percentage of consecutive valid predictions directly connected."""

    valid = 0
    connected = 0

    for previous_id, current_id in zip(
        predictions[:-1], predictions[1:]
    ):
        if previous_id is None or current_id is None:
            continue

        previous_arc = arcs.get(previous_id)
        current_arc = arcs.get(current_id)

        if previous_arc is None or current_arc is None:
            continue

        valid += 1

        if previous_arc[1] == current_arc[0]:
            connected += 1

    return connected / valid * 100 if valid else 0.0


def can_reach_within_hops(start, target, graph, max_hops=5):
    """Check directed network reachability within max_hops arcs."""

    if start == target:
        return True

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, hops = queue.popleft()

        if hops >= max_hops:
            continue

        for neighbour in graph.get(node, []):

            if neighbour == target:
                return True

            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, hops + 1))

    return False


def network_reachability(predictions, arcs, max_hops=5):
    """
    Percentage of consecutive valid predictions whose road segments
    are connected directly or through intermediate directed arcs.

    This is a connectivity diagnostic, not ground-truth accuracy.
    """

    graph = {}

    for start, end in arcs.values():
        graph.setdefault(start, []).append(end)

    valid = 0
    reachable = 0

    for previous_id, current_id in zip(
        predictions[:-1], predictions[1:]
    ):

        if previous_id is None or current_id is None:
            continue

        previous_arc = arcs.get(previous_id)
        current_arc = arcs.get(current_id)

        if previous_arc is None or current_arc is None:
            continue

        valid += 1

        if can_reach_within_hops(
            previous_arc[1],
            current_arc[0],
            graph,
            max_hops,
        ):
            reachable += 1

    return reachable / valid * 100 if valid else 0.0


def evaluate_predictions(predictions, ground_truth, arcs):
    """Calculate accuracy and network continuity diagnostics."""

    return {
        "pointwise_accuracy": pointwise_accuracy(
            predictions, ground_truth
        ),
        "direct_continuity": route_continuity(
            predictions, arcs
        ),
        "network_reachability": network_reachability(
            predictions, arcs
        ),
    }


def compare_predictions(
    classifier_predictions,
    hmm_predictions,
    ground_truth,
    arcs,
):
    """Compare classifier-only and HMM-decoded predictions."""

    classifier_results = evaluate_predictions(
        classifier_predictions,
        ground_truth,
        arcs
    )

    hmm_results = evaluate_predictions(
        hmm_predictions,
        ground_truth,
        arcs
    )

    print("\n========== PathSnap Evaluation ==========")

    for name, results in [
        ("Classifier Only", classifier_results),
        ("Classifier + HMM", hmm_results),
    ]:

        print(f"\n{name}:")

        print(
            f"Pointwise Accuracy: "
            f"{results['pointwise_accuracy']:.2f}%"
        )

        print(
            f"Direct Continuity: "
            f"{results['direct_continuity']:.2f}%"
        )

        print(
            f"Network Reachability: "
            f"{results['network_reachability']:.2f}%"
        )

    print("\n========== Improvement ==========")

    for metric in [
        "pointwise_accuracy",
        "direct_continuity",
        "network_reachability",
    ]:

        difference = (
            hmm_results[metric] - classifier_results[metric]
        )

        print(
            f"{metric}: "
            f"{difference:+.2f} percentage points"
        )

    return {
        "classifier_only": classifier_results,
        "classifier_plus_hmm": hmm_results,
    }
