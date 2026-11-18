
from pathlib import Path
from map_matching.dataset import load_trip

thresholds = [2, 3, 5, 10, 30, 60, 300, 600]

all_gaps = []
trip_max_gaps = []

for trip_dir in sorted(Path("data/train").iterdir()):
    if not trip_dir.is_dir():
        continue

    data = load_trip(trip_dir)
    timestamps = data["timestamps"]

    gaps = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    all_gaps.extend(gaps)

    if gaps:
        trip_max_gaps.append((max(gaps), data["trip_id"]))

all_gaps.sort()

print("Training trips:", len(trip_max_gaps))
print("Total time gaps:", len(all_gaps))

if all_gaps:
    print("Median gap:", all_gaps[len(all_gaps) // 2], "seconds")

    for percentile in [90, 95, 99, 99.9]:
        index = min(
            int((percentile / 100) * (len(all_gaps) - 1)),
            len(all_gaps) - 1
        )
        print(f"{percentile}th percentile:", all_gaps[index])

    print("\nGap counts:")
    for threshold in thresholds:
        count = sum(gap > threshold for gap in all_gaps)
        print(f"Gaps > {threshold} seconds: {count}")

    print("\nLargest 10 gaps:")
    for gap, trip_id in sorted(trip_max_gaps, reverse=True)[:10]:
        print(trip_id, gap, "seconds")
