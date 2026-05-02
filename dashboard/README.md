# Training Dashboard

Web-based monitoring UI for live training sessions. Shows metrics, decision distributions,
and lets you tune hyperparameters from the browser while training runs.

## Quick Start

```bash
pip install fastapi uvicorn aiosqlite scikit-learn httpx
python -m dashboard.server
```

Open `http://localhost:8000` in your browser.

## What You Can Do

| View | What It Shows |
|------|--------------|
| **Metrics** | Live loss, reward, and episode return curves per module |
| **Decisions** | EXPLORE/ROLLBACK/INTERRUPT/COMMIT_NEXT distribution histogram |
| **Hyperparams** | Edit lr, entropy_coeff, think_cost, batch_size — applied live via WebSocket |
| **Sessions** | Past training sessions with summaries, compare up to 5 by overlaying loss curves |
| **Embeddings** | 2D PCA scatter of z_t vectors, colored by training step or driving context |
| **Predictions** | Predicted vs actual scatter plot and MSE over time |
| **Weights** | List and download .pt checkpoint files across all modules |

## How It Works

```
Training scripts          Dashboard server              Browser
  (train.py)               (FastAPI + SQLite)           (Vue 3 SPA)
      |                          |                          |
      +-- write JSONL  -------> collector polls files       |
      |                          |                          |
      +-- WebSocket /ws/train    +-- inserts to SQLite      |
      |   (receive param         |                          |
      |    updates)              +-- WebSocket /ws/browser --+
      |                          |   (push metrics,         |
      |                          |    receive commands)      |
```

- **JSONL collector**: polls `training_data/` directories every 1.5s, ingests new metric rows into SQLite
- **WebSocket**: pushes live updates to the browser, forwards param changes to training scripts
- **REST API**: serves historical data, sessions, checkpoints, embeddings, predictions
- **Auth**: optional `DASHBOARD_PASSWORD` env var enables Bearer token auth

## Files

```
dashboard/
  server.py           FastAPI app, lifespan, WebSocket endpoints
  database.py         SQLite schema (5 tables), async query helpers
  models.py           Pydantic models for API types
  auth.py             Password middleware (timing-safe comparison)
  ws_manager.py       Dual WebSocket pool (browser + training clients)
  collector.py        JSONL-to-SQLite ingestion with byte-offset tracking
  routes/
    metrics.py        GET /api/metrics, /api/metrics/latest, /api/metrics/decisions
    params.py         GET/PUT /api/params (hot-reload to training scripts)
    sessions.py       GET /api/sessions, compare overlay
    checkpoints.py    GET /api/checkpoints, file download
    embeddings.py     GET /api/embeddings/pca (server-side PCA)
    predictions.py    GET /api/predictions/scatter, /error
  frontend/           Vue 3 + Vite SPA (built to frontend/dist/)
```
