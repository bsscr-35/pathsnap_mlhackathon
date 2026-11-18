"""Request / response models for the PathSnap API."""
from typing import List
from pydantic import BaseModel


class TripsResponse(BaseModel):
    trips: List[str]


class PredictRequest(BaseModel):
    trip_id: str


class Metrics(BaseModel):
    n_points: int
    accuracy: float
    route_continuity: float
    n_unique_predicted_arcs: int
    n_reference_arcs: int


class PredictResponse(BaseModel):
    trip_id: str
    metrics: Metrics
    latency_ms: float
    points: List[List[float]]              # [lat, lon] raw GPS points
    segments: List[List[List[float]]]      # predicted arc geometry [[lat,lon],[lat,lon]]
    # NOTE: no ground-truth field here on purpose. The reference route never leaves the server.
