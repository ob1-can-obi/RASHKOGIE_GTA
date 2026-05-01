"""REST endpoints for metrics queries.

DASH-02: loss/reward curves data source.
DASH-03: decision histogram data source.
DASH-07: episode_return field included in response.
DASH-08: nodes_expanded and search_depth included.

All database queries use helper functions from dashboard.database (Plan 01)
instead of raw SQL, ensuring single-source-of-truth for SQL queries and
parameterized query safety.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Query

from dashboard.database import get_metrics, get_latest_metrics

_log = logging.getLogger("dashboard.routes.metrics")

metrics_router = APIRouter(tags=["metrics"])


@metrics_router.get("/metrics")
async def list_metrics(
    request: Request,
    session_id: Optional[str] = None,
    module: Optional[str] = None,
    step_from: int = Query(0, ge=0),
    limit: int = Query(2000, ge=1, le=10000),
):
    """Return paginated metrics rows filtered by session and/or module.

    DASH-02: loss/reward curves data source.
    DASH-07: episode_return field included in response.

    Uses database.py get_metrics helper instead of raw SQL.
    When session_id is omitted, queries all sessions.
    """
    db = request.app.state.db
    if session_id:
        rows = await get_metrics(
            db, session_id=session_id, module=module,
            step_from=step_from, limit=limit,
        )
    else:
        # No session_id filter -- query by module or all metrics
        if module:
            cursor = await db.execute(
                "SELECT * FROM metrics WHERE module = ? AND step >= ? "
                "ORDER BY step ASC LIMIT ?",
                (module, step_from, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM metrics WHERE step >= ? "
                "ORDER BY step ASC LIMIT ?",
                (step_from, limit),
            )
        rows = await cursor.fetchall()
    rows_list = [dict(row) for row in rows]
    return {
        "rows": rows_list,
        "count": len(rows_list),
        "step_from": step_from,
        "limit": limit,
    }


@metrics_router.get("/metrics/latest")
async def latest_metrics(
    request: Request,
    module: str = Query(..., description="Module name (e.g., encoder_intuition)"),
    limit: int = Query(100, ge=1, le=5000),
):
    """Return the most recent N metrics rows for a module.

    DASH-02: live chart data source for initial page load.

    Uses database.py get_latest_metrics helper instead of raw SQL.
    """
    db = request.app.state.db
    rows = await get_latest_metrics(db, module=module, limit=limit)
    rows_list = [dict(row) for row in rows]
    return {"rows": rows_list, "module": module}


@metrics_router.get("/metrics/decisions")
async def get_decision_counts(
    request: Request,
    session_id: Optional[str] = None,
    limit: int = Query(500, ge=1, le=5000),
):
    """Return decision distribution counts.

    DASH-03: decision histogram data source.
    DASH-08: nodes_expanded and search_depth included.

    Note: decision_counts does not have a dedicated database.py query helper,
    so this route uses parameterized SQL directly. The query is simple enough
    that a dedicated helper adds no value.
    """
    db = request.app.state.db
    query = "SELECT * FROM decision_counts"
    params: list = []
    if session_id:
        query += " WHERE session_id = ?"
        params.append(session_id)
    query += " ORDER BY step DESC LIMIT ?"
    params.append(limit)
    cursor = await db.execute(query, params)
    rows = await cursor.fetchall()
    rows_list = [dict(row) for row in rows]
    rows_list.reverse()
    return {"rows": rows_list, "count": len(rows_list)}


@metrics_router.get("/pipeline/status")
async def get_pipeline_status(request: Request):
    """Return current training pipeline status from training_status.json."""
    status_path = Path(request.app.state.project_root) / "training_status.json"
    try:
        with open(status_path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"pipeline_run_id": None, "stages": {}, "frozen_modules": []}
