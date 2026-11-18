"""FastAPI backend (Laptop 1).
Run from the repo root:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import pipeline
from backend.schemas import PredictRequest, PredictResponse, TripsResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Serving trips from %s (%d trips)", pipeline.EVAL_DIR, len(pipeline.list_trips()))
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trips", response_model=TripsResponse)
def trips():
    ids = pipeline.list_trips()
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    log.info("POST /predict received trip_id=%s", req.trip_id)
    t0 = time.time()
    try:
        result = pipeline.predict_trip(req.trip_id)
    except FileNotFoundError:
        log.warning("Unknown trip_id=%s", req.trip_id)
        raise HTTPException(status_code=404, detail=f"Unknown trip {req.trip_id}")
    metrics = pipeline.evaluate(req.trip_id, result)
    ms = round((time.time() - t0) * 1000, 1)
    log.info("trip=%s points=%d accuracy=%.4f continuity=%.4f time=%sms",
             req.trip_id, metrics["n_points"], metrics["accuracy"], metrics["route_continuity"], ms)
    return {"trip_id": req.trip_id, "metrics": metrics, "latency_ms": ms,
            "points": result["points"], "segments": result["segments"]}
