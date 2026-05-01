"""Server-side PCA embedding visualization endpoint.

D-20: PCA computed server-side using scikit-learn.
D-21: Two embedding sub-views:
  (a) cluster evolution -- z_t colored by training step
  (b) state-type clustering -- z_t colored by driving context (straight/turn/braking)

All database queries use get_embeddings helper from dashboard.database (Plan 01)
instead of raw SQL.

T-05-13: max_points capped at 2000; PCA on 2000x128 matrix is fast (<100ms).
"""
from __future__ import annotations

import logging
import struct
from typing import Optional

import numpy as np
from fastapi import APIRouter, Request, Query, HTTPException

from dashboard.database import get_embeddings

_log = logging.getLogger("dashboard.routes.embeddings")

embeddings_router = APIRouter(tags=["embeddings"])


@embeddings_router.get("/embeddings/pca")
async def get_embedding_pca(
    request: Request,
    session_id: str = Query(..., description="Session ID to compute PCA for"),
    max_points: int = Query(500, ge=2, le=2000, description="Max embedding points"),
    color_by: str = Query(
        "step",
        description="Color by: 'step' (cluster evolution D-21a) or 'context' (state-type clustering D-21b)",
    ),
):
    """Compute 2D PCA projection of 128-dim z_t embeddings.

    D-20: PCA computed server-side using scikit-learn.
    D-21: Two embedding sub-views:
      (a) cluster evolution -- z_t colored by training step
      (b) state-type clustering -- z_t colored by driving context (straight/turn/braking)

    Uses database.py get_embeddings helper instead of raw SQL.
    """
    from sklearn.decomposition import PCA

    db = request.app.state.db
    rows = await get_embeddings(db, session_id=session_id, limit=max_points)

    if len(rows) < 2:
        return {
            "points": [],
            "explained_variance": [],
            "message": "Need at least 2 embedding snapshots for PCA",
        }

    # Decode BLOB -> numpy arrays
    vectors = []
    steps = []
    contexts = []
    for row in rows:
        z_blob = row["z_t"]
        n_floats = len(z_blob) // 4  # float32 = 4 bytes
        z_array = np.array(struct.unpack(f"{n_floats}f", z_blob), dtype=np.float32)
        vectors.append(z_array)
        steps.append(row["step"])
        contexts.append(row["driving_context"] or "unknown")

    vectors_np = np.array(vectors)

    # Handle degenerate cases (zero variance in any dimension)
    if vectors_np.std() < 1e-8:
        return {
            "points": [
                {"x": 0.0, "y": 0.0, "step": s, "context": c}
                for s, c in zip(steps, contexts)
            ],
            "explained_variance": [0.0, 0.0],
            "message": "All embeddings are nearly identical (zero variance)",
        }

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors_np)

    points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "step": steps[i],
            "context": contexts[i],
        }
        for i in range(len(rows))
    ]

    return {
        "points": points,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "color_by": color_by,
        "session_id": session_id,
        "total_points": len(points),
    }


@embeddings_router.get("/embeddings/sessions")
async def get_embedding_sessions(request: Request):
    """List sessions that have embedding snapshots available."""
    db = request.app.state.db
    cursor = await db.execute(
        "SELECT DISTINCT e.session_id, s.module, COUNT(*) as snapshot_count "
        "FROM embeddings e JOIN sessions s ON e.session_id = s.session_id "
        "GROUP BY e.session_id ORDER BY e.session_id DESC"
    )
    rows = await cursor.fetchall()
    return {"sessions": [dict(row) for row in rows]}
