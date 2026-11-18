
from pathlib import Path
from math import radians, sin, cos, asin, sqrt
from map_matching.dataset import load_trip


def haversine(p, q):
    lon1, lat1 = map(radians, p)
    lon2, lat2 = map(radians, q)

    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2)
        * sin((lon2 - lon1) / 2) ** 2
    )

    return 6371000 * 2 * asin(sqrt(a))


long_gaps = []

for trip_dir in sorted(Path("data/train").iterdir()):
    if not trip_dir.is_dir():
        continue

    data = load_trip(trip_dir)
    timestamps = data["timestamps"]
    points = data["gps_points"]

    for i in range(len(timestamps) - 1):
        gap = timestamps[i + 1] - timestamps[i]

        if gap > 30:
            distance = haversine(points[i], points[i + 1])
            speed = distance / gap if gap > 0 else 0

            long_gaps.append({
                "trip": data["trip_id"],
                "index": i,
                "gap": gap,
                "distance": distance,
                "speed": speed,
            })

long_gaps.sort(key=lambda x: x["gap"], reverse=True)

print("Total gaps > 30 seconds:", len(long_gaps))
print("\nLargest 15 gaps:")

for item in long_gaps[:15]:
    print(
        f"Trip: {item['trip']} | "
        f"Index: {item['index']} | "
        f"Gap: {item['gap']:.1f}s | "
        f"Distance: {item['distance']:.2f}m | "
        f"Speed: {item['speed']:.4f}m/s"
    )
