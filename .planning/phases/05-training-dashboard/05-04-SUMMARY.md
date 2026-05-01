---
phase: 05-training-dashboard
plan: 04
subsystem: dashboard-api
tags: [rest-api, sessions, checkpoints, embeddings, predictions, pca]
dependency_graph:
  requires: ["05-01 database.py", "05-01 models.py", "05-02 ws_manager.py"]
  provides: ["sessions_router", "checkpoints_router", "embeddings_router", "predictions_router"]
  affects: ["05-05 (Vue frontend consumes these endpoints)", "05-06 (server.py registers these routers)"]
tech_stack:
  added: ["sklearn.decomposition.PCA (server-side embedding reduction)"]
  patterns: ["FastAPI APIRouter per-domain", "aiosqlite.Row -> dict conversion", "struct.unpack BLOB decode", "FileResponse for binary download", "path traversal defense in depth"]
key_files:
  created:
    - dashboard/routes/sessions.py
    - dashboard/routes/checkpoints.py
    - dashboard/routes/embeddings.py
    - dashboard/routes/predictions.py
  modified: []
decisions:
  - "Route ordering: /sessions/compare/overlay defined before /sessions/{session_id} to prevent path parameter capture conflict"
  - "Checkpoint download via HTTP FileResponse (not WebSocket) per RESEARCH.md Open Question 3"
  - "Joint PCA on pred+real vectors for prediction scatter (consistent 2D projection space)"
  - "aiosqlite.Row objects converted to dicts before adding display_name fields"
metrics:
  duration: "151s"
  completed: "2026-05-01T20:58:32Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 4
  files_modified: 0
---

# Phase 05 Plan 04: REST Route Modules (Sessions, Checkpoints, Embeddings, Predictions) Summary

Four FastAPI route modules completing the backend API surface for session management, checkpoint operations, server-side PCA embedding visualization, and prediction quality endpoints -- all using database.py query helpers for single-source-of-truth SQL.

## Task Completion

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create sessions and checkpoints route modules | 6a11d11 | dashboard/routes/sessions.py, dashboard/routes/checkpoints.py |
| 2 | Create embeddings and predictions route modules | 8bc796f | dashboard/routes/embeddings.py, dashboard/routes/predictions.py |

## What Was Built

### Sessions Router (dashboard/routes/sessions.py)
- `GET /sessions` -- list all sessions, filterable by module, with STAGE_DISPLAY names
- `GET /sessions/compare/overlay?ids=a,b,c` -- overlay metrics for up to 5 sessions (D-10 cap)
- `GET /sessions/{session_id}` -- single session with summary metrics (min/max/avg loss, max step)
- `GET /sessions/{session_id}/metrics` -- paginated metrics for a session (step_from, limit)
- All queries use `get_sessions` and `get_metrics` from database.py

### Checkpoints Router (dashboard/routes/checkpoints.py)
- `GET /checkpoints` -- list all .pt files across 4 module checkpoint directories with file size/timestamp
- `GET /checkpoints/{module}/{session_id}/{filename}` -- download checkpoint via FileResponse
- Path traversal protection: reject ".." components AND verify resolved path within expected directory (T-05-12, T-05-15)
- MODULE_CHECKPOINT_DIRS maps 4 modules to their checkpoint paths

### Embeddings Router (dashboard/routes/embeddings.py)
- `GET /embeddings/pca?session_id=X&max_points=500&color_by=step` -- 2D PCA of 128-dim z_t vectors
- `GET /embeddings/sessions` -- list sessions with embedding data available
- BLOB decoding via struct.unpack (float32 x N), PCA via sklearn (D-20)
- Supports color_by="step" (D-21a cluster evolution) and color_by="context" (D-21b state-type)
- Degenerate case handling: zero-variance vectors return zero coordinates with message
- Max points capped at 2000 (T-05-13 DoS protection)

### Predictions Router (dashboard/routes/predictions.py)
- `GET /predictions/scatter?session_id=X` -- predicted vs actual scatter (joint PCA on pred+real)
- `GET /predictions/error?session_id=X` -- per-step MSE over training steps (line chart data)
- Uses `get_predictions` from database.py; no raw SQL
- Joint PCA ensures pred and real coordinates share the same 2D projection space

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed aiosqlite.Row mutation in sessions list endpoint**
- **Found during:** Task 1
- **Issue:** Plan code did `s["display_name"] = ...` on aiosqlite.Row objects, which are immutable (don't support item assignment)
- **Fix:** Added `[dict(row) for row in rows]` conversion before adding display_name
- **Files modified:** dashboard/routes/sessions.py
- **Commit:** 6a11d11

**2. [Rule 1 - Bug] Fixed FastAPI route ordering for sessions compare endpoint**
- **Found during:** Task 1
- **Issue:** Plan defined `/sessions/{session_id}` before `/sessions/compare/overlay`, causing "compare" to be captured as a session_id
- **Fix:** Moved compare_sessions route definition before get_session_detail
- **Files modified:** dashboard/routes/sessions.py
- **Commit:** 6a11d11

## Threat Mitigations Applied

| Threat ID | Component | Mitigation |
|-----------|-----------|------------|
| T-05-12 | checkpoints.py download | Reject ".." in path components + verify resolved path within checkpoint directory |
| T-05-13 | embeddings.py PCA | max_points capped at 2000 via Query(ge=2, le=2000) |
| T-05-14 | sessions.py queries | All queries use database.py parameterized helpers |
| T-05-15 | checkpoints.py download | Only .pt files within known MODULE_CHECKPOINT_DIRS downloadable |

## Verification Results

- All 4 route modules import successfully
- database.py import count: sessions.py=1, embeddings.py=1, predictions.py=1
- Path traversal check present in checkpoints.py (2 occurrences of "..")
- No raw SQL in predictions.py (0 SELECT FROM predictions matches)
- No stubs or placeholder content found

## Self-Check: PASSED

- All 5 files exist on disk (4 route modules + SUMMARY.md)
- Both task commits found in git log (6a11d11, 8bc796f)
