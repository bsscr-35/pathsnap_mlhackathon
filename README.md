# PathSnap: Distributed Map-Matching & Route Reconstruction

Team project for UE24CS352A (PES University).

## Structure
| Folder | Content |
|---|---|
| `backend/` | FastAPI backend (Laptop 1): `main.py` endpoints, `schemas.py` models, `pipeline.py` prediction pipeline |
| `frontend/` | Streamlit evaluator UI (Laptop 2): `app.py` |
| `data/` | Dataset (NOT committed). Put it at `data/Student folder/{train,student_test}/` |

## Setup
```bash
pip install -r requirements.txt
```

## Run the API (Laptop 1), from the repo root
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
# final evaluation:
# EVAL_DIR=/path/to/evaluator_trips uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Find the IP with `hostname -I`. If the firewall is on: `sudo ufw allow 8000/tcp`.
Check from Laptop 2: `http://<LAPTOP1-IP>:8000/docs`

## Run the UI (Laptop 2)
```bash
PATHSNAP_API=http://<LAPTOP1-IP>:8000 streamlit run frontend/app.py
```
The API URL can also be changed in the UI sidebar. The map needs internet for the OpenStreetMap tiles.

## API
- `GET /trips` returns the trip IDs.
- `POST /predict {"trip_id": "..."}` returns GPS points, predicted route segments and metrics.
- Ground truth (`.route`) is read server-side only in `pipeline.evaluate` and is never returned.

## Plugging in the real pipeline
`backend/pipeline.py` currently uses a nearest-segment baseline (no labels used). Replace
`_nearest_arcs` with candidate generation (A), model probabilities (B) and Viterbi (C).
Keep the return format described at the top of that file.

## Notes
- `accuracy` is currently "fraction of predicted arcs on the reference route". Align it with the team's
  final observation-level Top-1 definition before the report.
