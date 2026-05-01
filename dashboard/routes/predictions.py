"""Prediction quality visualization endpoint.

D-22a: predicted vs actual scatter plot (z_next_pred vs real z_{t+1}, PCA-reduced).
D-22b: prediction error (MSE) over training steps as a line chart.

All database queries use get_predictions helper from dashboard.database (Plan 01)
instead of raw SQL.
"""
from __future__ import annotations

import logging
import struct

import numpy as np
from fastapi import APIRouter, Request, Query

from dashboard.database import get_predictions

_log = logging.getLogger("dashboard.routes.predictions")

predictions_router = APIRouter(tags=["predictions"])


@predictions_router.get("/predictions/scatter")
async def get_prediction_scatter(
    request: Request,
    session_id: str = Query(..., description="Session ID"),
    max_points: int = Query(500, ge=1, le=2000),
):
    """Return predicted vs actual scatter data (PCA-reduced to 2D).

    D-22a: predicted vs actual scatter plot (z_next_pred vs real z_{t+1}, PCA-reduced).
    Points near the diagonal = good predictions.

    Uses database.py get_predictions helper instead of raw SQL.
    """
    from sklearn.decomposition import PCA

    db = request.app.state.db
    rows = await get_predictions(db, session_id=session_id, limit=max_points)

    if len(rows) < 2:
        return {"points": [], "message": "Need at least 2 prediction records"}

    pred_vectors = []
    real_vectors = []
    steps = []
    mse_values = []

    for row in rows:
        pred_blob = row["z_next_pred"]
        real_blob = row["z_next_real"]
        n_floats = len(pred_blob) // 4
        pred_vectors.append(
            np.array(struct.unpack(f"{n_floats}f", pred_blob), dtype=np.float32)
        )
        real_vectors.append(
            np.array(struct.unpack(f"{n_floats}f", real_blob), dtype=np.float32)
        )
        steps.append(row["step"])
        mse_values.append(row["mse"])

    # PCA on combined vectors (both pred and real together for consistent projection)
    all_vectors = np.vstack([np.array(pred_vectors), np.array(real_vectors)])

    if all_vectors.std() < 1e-8:
        return {"points": [], "message": "All vectors are nearly identical"}

    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(all_vectors)

    n = len(pred_vectors)
    pred_coords = all_coords[:n]
    real_coords = all_coords[n:]

    points = [
        {
            "pred_x": float(pred_coords[i, 0]),
            "pred_y": float(pred_coords[i, 1]),
            "real_x": float(real_coords[i, 0]),
            "real_y": float(real_coords[i, 1]),
            "step": steps[i],
            "mse": mse_values[i],
        }
        for i in range(n)
    ]

    return {
        "points": points,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "session_id": session_id,
    }


@predictions_router.get("/predictions/error")
async def get_prediction_error(
    request: Request,
    session_id: str = Query(..., description="Session ID"),
    limit: int = Query(2000, ge=1, le=10000),
):
    """Return per-step prediction MSE over training steps.

    D-22b: prediction error (MSE) over training steps as a line chart.
    Trend should decrease during successful training.

    Uses database.py get_predictions helper for data retrieval.
    """
    db = request.app.state.db
    rows = await get_predictions(db, session_id=session_id, limit=limit)
    return {
        "data": [{"step": row["step"], "mse": row["mse"]} for row in rows],
        "session_id": session_id,
        "count": len(rows),
    }
