
from map_matching.dataset import load_trip
from math import radians, sin, cos, asin, sqrt

trips = [
    ("00000011", 1600),
    ("00000069", 843),
    ("00000042", 1891),
    ("00000021", 183),
    ("00000042", 7250),
    ("00000081", 2776),
    ("00000078", 1),
]


def haversine(p, q):
    lon1, lat1 = map(radians, p)
    lon2, lat2 = map(radians, q)

    a = (
        sin((lat2 - lat1) / 2) ** 2
        + cos(lat1) * cos(lat2)
        * sin((lon2 - lon1) / 2) ** 2
    )

    return 6371000 * 2 * asin(sqrt(a))


for trip_id, index in trips:
    data = load_trip("data/train/" + trip_id)

    p = data["gps_points"][index]
    q = data["gps_points"][index + 1]

    gap = data["timestamps"][index + 1] - data["timestamps"][index]
    distance = haversine(p, q)
    speed = distance / gap if gap > 0 else 0

    print("\nTrip:", trip_id)
    print("Time gap:", gap, "seconds")
    print("Distance:", round(distance, 2), "metres")
    print("Average speed:", round(speed, 2), "m/s")
