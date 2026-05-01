---
phase: 05-training-dashboard
plan: 01
subsystem: dashboard-backend
tags: [fastapi, sqlite, auth, pydantic, test-scaffold]
dependency_graph:
  requires: []
  provides: [dashboard-package, sqlite-schema, auth-middleware, test-fixtures]
  affects: [05-02, 05-03, 05-04, 05-05, 05-06]
tech_stack:
  added: [fastapi-0.136, uvicorn-0.46, aiosqlite-0.22, scikit-learn-1.8, pytest-asyncio-0.25]
  patterns: [asynccontextmanager-lifespan, BaseHTTPMiddleware, parameterized-SQL, WAL-mode]
key_files:
  created:
    - dashboard/__init__.py
    - dashboard/database.py
    - dashboard/models.py
    - dashboard/auth.py
    - dashboard/server.py
    - dashboard/routes/__init__.py
    - tests/test_dashboard.py
  modified:
    - tests/conftest.py
decisions:
  - "JSONResponse used instead of raise HTTPException in BaseHTTPMiddleware (Starlette known issue: exceptions in middleware dispatch become 500s)"
  - "pytest_asyncio.fixture decorator used for async fixtures (pytest.fixture with async generators requires it)"
  - "Pre-existing import chain break (reward_head.predict_reward_features) guarded with try/except in conftest.py"
metrics:
  duration: 408s
  completed: "2026-05-01T20:52:10Z"
  tasks_completed: 2
  tasks_total: 2
  files_created: 7
  files_modified: 1
---

# Phase 5 Plan 1: Dashboard Backend Foundation Summary

FastAPI server skeleton with SQLite database (5 tables, 6 indices, WAL mode), Pydantic models for all API types, timing-safe password auth middleware, and test scaffold covering DASH-01 through DASH-08.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 620c5c4 | feat(05-01): create dashboard foundation modules (database, models, auth) |
| 2 | 8bf27b2 | feat(05-01): create FastAPI server, test scaffold, and dashboard fixtures |

## Task Details

### Task 1: Install Python dependencies and create database + models + auth modules

Created four foundation modules:

- **dashboard/__init__.py**: Empty package marker
- **dashboard/database.py**: SQLite schema with 5 CREATE TABLE statements (sessions, metrics, decision_counts, embeddings, predictions), 6 CREATE INDEX statements, WAL journal mode, busy_timeout=5000ms, and 12 async query helper functions all using parameterized queries
- **dashboard/models.py**: 6 Pydantic BaseModel classes (MetricsRow, DecisionCountRow, SessionInfo, ParamUpdate, WSMessage, CheckpointInfo) matching the database schema and WebSocket protocol
- **dashboard/auth.py**: PasswordAuthMiddleware using hmac.compare_digest for timing-safe comparison, /api/health excluded from auth, WebSocket token verification

Installed: fastapi 0.136.1, uvicorn 0.46.0, aiosqlite 0.22.0, scikit-learn 1.8.0, httpx, pytest-asyncio

### Task 2: Create FastAPI server with lifespan and test scaffold

- **dashboard/server.py**: FastAPI app with asynccontextmanager lifespan (init_database on startup, db.close on shutdown), PasswordAuthMiddleware added, /api/health endpoint, CLI entry point binding to 0.0.0.0:8000 per D-15
- **dashboard/routes/__init__.py**: Empty routes package marker for future route modules
- **tests/test_dashboard.py**: 12 tests total -- 5 passing (health check, auth exclusion, auth blocking, db tables, WAL mode) and 7 skipped placeholders for DASH-02 through DASH-08
- **tests/conftest.py**: Added 3 dashboard fixtures (test_db, test_client, test_client_with_auth) with temp database isolation; guarded pre-existing broken import chain

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed BaseHTTPMiddleware exception handling**
- **Found during:** Task 2
- **Issue:** Raising HTTPException inside BaseHTTPMiddleware.dispatch results in 500 Internal Server Error (known Starlette behavior -- exceptions in middleware dispatch bypass FastAPI's exception handler)
- **Fix:** Changed to return JSONResponse(status_code=401, content={"detail": "Unauthorized"}) instead of raise HTTPException(status_code=401)
- **Files modified:** dashboard/auth.py
- **Commit:** 8bf27b2

**2. [Rule 3 - Blocking] Guarded broken import chain in conftest.py**
- **Found during:** Task 2
- **Issue:** Pre-existing error: `from main_model import create_encoder_weights` triggers import of `from reward_head import predict_reward_features` which does not exist in reward_head.py. This prevented ALL tests from loading, including dashboard tests.
- **Fix:** Wrapped MetaMLP and create_encoder_weights imports in try/except with _HAS_MODEL_IMPORTS flag; added pytest.skip() to affected fixtures when imports unavailable
- **Files modified:** tests/conftest.py
- **Commit:** 8bf27b2

**3. [Rule 3 - Blocking] Fixed async fixture decorator**
- **Found during:** Task 2
- **Issue:** pytest-asyncio strict mode requires @pytest_asyncio.fixture for async generator fixtures, not @pytest.fixture
- **Fix:** Changed test_db fixture to use @pytest_asyncio.fixture decorator
- **Files modified:** tests/conftest.py
- **Commit:** 8bf27b2

## Verification Results

```
tests/test_dashboard.py::test_server_starts PASSED
tests/test_dashboard.py::test_auth_blocks_unauthorized PASSED
tests/test_dashboard.py::test_auth_blocks_protected_routes PASSED
tests/test_dashboard.py::test_database_tables_created PASSED
tests/test_dashboard.py::test_database_wal_mode PASSED
tests/test_dashboard.py::test_ws_metrics_push SKIPPED
tests/test_dashboard.py::test_decision_counts_ingestion SKIPPED
tests/test_dashboard.py::test_param_hot_reload SKIPPED
tests/test_dashboard.py::test_session_history SKIPPED
tests/test_dashboard.py::test_session_comparison SKIPPED
tests/test_dashboard.py::test_episode_return_ingestion SKIPPED
tests/test_dashboard.py::test_nodes_expanded_ingestion SKIPPED

5 passed, 7 skipped
```

## Threat Flags

No new threat surface beyond what is documented in the plan's threat model. All mitigations implemented:
- T-05-01: hmac.compare_digest in auth.py
- T-05-02: All SQL uses parameterized queries in database.py
- T-05-03: Timing-safe comparison prevents side-channel leakage
- T-05-04: Accepted (LAN-only, research tool)

## Self-Check: PASSED

All 7 created files verified on disk. Both commit hashes (620c5c4, 8bf27b2) found in git log.
