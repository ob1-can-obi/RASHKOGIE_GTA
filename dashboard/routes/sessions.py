"""Session history and comparison endpoints.

DASH-05: session history with per-session metrics summary.
DASH-06: session comparison overlay view.
D-10: max 4-5 overlaid curves for comparison.
D-11: per-module session granularity.

All database queries use helper functions from dashboard.database (Plan 01)
instead of raw SQL -- single-source-of-truth for SQL queries and parameterized
query safety (T-05-14).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request, Query, HTTPException

from dashboard.database import get_sessions, get_metrics

_log = logging.getLogger("dashboard.routes.sessions")

# Stage display names (from coordinator.py lines 38-43)
STAGE_DISPLAY = {
    "encoder_intuition": "Encoder + Intuition",
    "reward_head": "Reward Head",
    "action_planner": "Action Planner",
    "metacontroller": "Metacontroller",
}

sessions_router = APIRouter(tags=["sessions"])


@sessions_router.get("/sessions")
async def list_sessions(
    request: Request,
    module: Optional[str] = None,
):
    """List all training sessions, optionally filtered by module.

    DASH-05: session history with per-session metrics summary.
    D-11: per-module session granularity.

    Uses database.py get_sessions helper instead of raw SQL.
    """
    db = request.app.state.db
    rows = await get_sessions(db, module=module)
    # Convert aiosqlite.Row objects to dicts so we can add display_name
    sessions = [dict(row) for row in rows]
    for s in sessions:
        s["display_name"] = STAGE_DISPLAY.get(s.get("module", ""), s.get("module", ""))
    return {"sessions": sessions, "count": len(sessions)}


# NOTE: compare route is defined BEFORE {session_id} detail route to prevent
# FastAPI from capturing "compare" as a session_id path parameter.
@sessions_router.get("/sessions/compare/overlay")
async def compare_sessions(
    request: Request,
    ids: str = Query(..., description="Comma-separated session IDs (max 5, per D-10)"),
):
    """Return metrics for multiple sessions for overlay comparison.

    DASH-06: session comparison view, max 4-5 overlaid curves (D-10).
    D-11: per-module session granularity.

    Uses database.py get_metrics helper for each session.
    """
    session_ids = [s.strip() for s in ids.split(",") if s.strip()]
    if len(session_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 sessions for comparison (D-10)")

    db = request.app.state.db
    result = {}
    for sid in session_ids:
        rows = await get_metrics(db, session_id=sid, limit=2000)
        # Get session info
        cursor2 = await db.execute("SELECT module FROM sessions WHERE session_id = ?", (sid,))
        session_row = await cursor2.fetchone()
        module = dict(session_row)["module"] if session_row else "unknown"
        result[sid] = {
            "module": module,
            "display_name": STAGE_DISPLAY.get(module, module),
            "data": [dict(r) for r in rows],
        }
    return {"sessions": result, "count": len(result)}


@sessions_router.get("/sessions/{session_id}")
async def get_session_detail(request: Request, session_id: str):
    """Get a single session with its metrics summary.

    DASH-05: per-session metrics summary.

    Uses database.py get_sessions to find the session, then raw SQL for
    the summary aggregation (no dedicated helper for single-session lookup
    with aggregation).
    """
    db = request.app.state.db
    cursor = await db.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    )
    session = await cursor.fetchone()
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    session_dict = dict(session)
    # Add summary metrics
    cursor2 = await db.execute(
        "SELECT COUNT(*) as total_rows, MIN(loss) as min_loss, MAX(loss) as max_loss, "
        "AVG(loss) as avg_loss, MAX(step) as max_step "
        "FROM metrics WHERE session_id = ?",
        (session_id,),
    )
    summary = dict(await cursor2.fetchone())
    session_dict["summary"] = summary
    session_dict["display_name"] = STAGE_DISPLAY.get(
        session_dict.get("module", ""), session_dict.get("module", "")
    )
    return session_dict


@sessions_router.get("/sessions/{session_id}/metrics")
async def get_session_metrics(
    request: Request,
    session_id: str,
    step_from: int = Query(0, ge=0),
    limit: int = Query(2000, ge=1, le=10000),
):
    """Get metrics for a specific session.

    DASH-06: data source for session comparison overlay.

    Uses database.py get_metrics helper instead of raw SQL.
    """
    db = request.app.state.db
    rows = await get_metrics(db, session_id=session_id, step_from=step_from, limit=limit)
    return {"rows": [dict(r) for r in rows], "session_id": session_id, "count": len(rows)}
