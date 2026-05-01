---
phase: "05-training-dashboard"
plan: "03"
subsystem: "dashboard-api"
tags: [fastapi, rest-api, websocket, metrics, params, hot-reload]
dependency_graph:
  requires: ["05-01", "05-02"]
  provides: ["metrics-rest-api", "params-rest-api", "websocket-endpoints", "collector-wiring"]
  affects: ["dashboard/server.py", "dashboard/routes/metrics.py", "dashboard/routes/params.py"]
tech_stack:
  added: []
  patterns: ["asyncio.Lock for config protection", "aiosqlite.Row to dict conversion", "WebSocket message routing"]
key_files:
  created:
    - dashboard/routes/metrics.py
    - dashboard/routes/params.py
  modified:
    - dashboard/server.py
    - tests/conftest.py
    - tests/test_dashboard.py
decisions:
  - "aiosqlite.Row objects converted to dict in route handlers for JSON serialization"
  - "GET /api/metrics handles optional session_id by falling back to direct SQL when no session filter"
  - "WebSocket /ws/browser handles set_params inline with shared _config_lock from params route"
metrics:
  duration: "3m 52s"
  completed: "2026-05-01"
---

# Phase 05 Plan 03: REST Routes, WebSocket Endpoints, and Server Wiring Summary

Metrics/params REST routes with Pydantic validation, WebSocket endpoints for browser and training script clients, collector background task wired into server lifespan, and 5 previously-skipped tests replaced with real assertions.

## What Was Built

### Task 1: Metrics and Params REST Routes
- **dashboard/routes/metrics.py**: Four endpoints:
  - `GET /api/metrics` - paginated loss/reward/episode_return data (DASH-02, DASH-07), uses `get_metrics` and `get_latest_metrics` from database.py
  - `GET /api/metrics/latest` - last N metrics rows for a module (DASH-02)
  - `GET /api/metrics/decisions` - decision distribution counts with nodes_expanded and search_depth (DASH-03, DASH-08)
  - `GET /api/pipeline/status` - training pipeline status from training_status.json
- **dashboard/routes/params.py**: Two endpoints:
  - `GET /api/params` - returns training_config.json contents with config_version
  - `PUT /api/params` - validates with Pydantic (lr>0, batch_size>=1, entropy_coeff>=0, think_cost>=0), writes to file under asyncio.Lock, broadcasts to training scripts via WebSocket

### Task 2: Server Wiring and WebSocket Endpoints
- **dashboard/server.py**: Full lifespan wiring:
  - WSManager created and stored on `app.state.ws`
  - JSONLCollector started as background task via `asyncio.create_task`
  - `project_root` stored on `app.state` for route access
  - `metrics_router` and `params_router` included at `/api` prefix
  - `/ws/browser` endpoint: auth via `verify_ws_token`, handles `set_params`, `restore_checkpoint`, `ping` messages
  - `/ws/train` endpoint: handles `register` and `ping` messages
  - SPA mount activated (conditional on frontend/dist existence)
- **tests/test_dashboard.py**: 5 skipped tests replaced with real assertions
- **tests/conftest.py**: `test_client` fixture now copies `training_config.json` to tmp_path and overrides `_PROJECT_ROOT`

## Test Results

```
11 passed, 2 skipped (Plan 04 tests) in 0.24s
```

| Test | Status | Covers |
|------|--------|--------|
| test_server_starts | PASS | DASH-01 |
| test_auth_blocks_unauthorized | PASS | DASH-01/D-16 |
| test_auth_blocks_protected_routes | PASS | DASH-01/D-16 |
| test_database_tables_created | PASS | DASH-01/D-01 |
| test_database_wal_mode | PASS | DASH-01 |
| test_ws_metrics_push | PASS | DASH-02 |
| test_decision_counts_ingestion | PASS | DASH-03 |
| test_param_hot_reload | PASS | DASH-04 |
| test_get_params | PASS | DASH-04 |
| test_episode_return_ingestion | PASS | DASH-07 |
| test_nodes_expanded_ingestion | PASS | DASH-08 |
| test_session_history | SKIP | Plan 04 |
| test_session_comparison | SKIP | Plan 04 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] aiosqlite.Row JSON serialization**
- **Found during:** Task 1
- **Issue:** `get_metrics` and `get_latest_metrics` from database.py return `aiosqlite.Row` objects which are not directly JSON serializable by FastAPI
- **Fix:** Added `[dict(row) for row in rows]` conversion in all route handlers before returning
- **Files modified:** dashboard/routes/metrics.py

**2. [Rule 1 - Bug] get_metrics requires session_id but route makes it optional**
- **Found during:** Task 1
- **Issue:** `database.py get_metrics()` takes `session_id` as a required positional parameter, but the plan's route signature makes `session_id` optional. Calling `get_metrics(db, session_id=None, ...)` would query with `WHERE session_id = NULL` returning no results.
- **Fix:** Added conditional: when `session_id` is provided, delegate to `get_metrics` helper; when omitted, use direct parameterized SQL to query by module or all metrics
- **Files modified:** dashboard/routes/metrics.py

## Commits

| Task | Hash | Message |
|------|------|---------|
| 1 | 2d5ab23 | feat(05-03): create metrics and params REST routes |
| 2 | e1ac117 | feat(05-03): wire server lifespan, WebSocket endpoints, and activate tests |

## Threat Mitigation Verification

| Threat ID | Status | Implementation |
|-----------|--------|---------------|
| T-05-08 (Tampering params) | Mitigated | ParamUpdateRequest has Field validators (gt=0, ge=0, ge=1); asyncio.Lock prevents concurrent config writes |
| T-05-09 (Tampering ws_browser) | Mitigated | verify_ws_token called before WebSocket accept; unknown message types silently ignored |
| T-05-10 (Spoofing ws_train) | Accepted | Training script WS has no auth (local network only) per plan disposition |
| T-05-11 (Repudiation params) | Mitigated | Param changes logged with module and changed keys via _log.info |

## Self-Check: PASSED

- All 5 key files exist on disk
- Both task commit hashes (2d5ab23, e1ac117) found in git log
- 11 tests pass, 0 failures
