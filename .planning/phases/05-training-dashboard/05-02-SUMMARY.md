---
phase: 05-training-dashboard
plan: 02
subsystem: dashboard-live-pipeline
tags: [websocket, collector, jsonl, sqlite, live-data]
dependency_graph:
  requires: []
  provides: [ws_manager, jsonl_collector]
  affects: [dashboard/server.py, dashboard routes]
tech_stack:
  added: [fastapi-websocket]
  patterns: [byte-offset-tracking, stale-client-cleanup, row-classification]
key_files:
  created:
    - dashboard/__init__.py
    - dashboard/ws_manager.py
    - dashboard/collector.py
    - dashboard/database.py (stub for parallel development)
  modified: []
decisions:
  - "Separate browser_clients and train_clients lists for distinct push semantics"
  - "Connection caps (20 browser, 10 training) mitigate T-05-06 DoS"
  - "Byte-offset tracking per file avoids re-reading old JSONL data"
  - "JSONDecodeError breaks (not continues) to handle partial writes from concurrent training scripts"
  - "Broadcasts limited to last 10 rows per push to avoid large WebSocket payloads"
  - "database.py stub created for parallel development -- Plan 01 provides real implementation"
metrics:
  duration: 188s
  completed: "2026-05-01T20:49:13Z"
---

# Phase 05 Plan 02: WebSocket Manager and JSONL Collector Summary

WebSocket connection manager with dual client pools (browser/training) and JSONL-to-SQLite collector with byte-offset tracking and partial-write resilience.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create WebSocket connection manager | adcc5fe | dashboard/ws_manager.py, dashboard/__init__.py |
| 2 | Create JSONL collector with byte-offset tracking | d7f4944 | dashboard/collector.py, dashboard/database.py |

## What Was Built

### WSManager (dashboard/ws_manager.py)

- **Dual client pools**: `browser_clients` and `train_clients` tracked separately
- **Broadcast methods**: `broadcast_metrics`, `broadcast_decisions`, `broadcast_session_event`, `broadcast_param_change`
- **Connection lifecycle**: `connect_browser`, `connect_train`, `disconnect_browser`, `disconnect_train`
- **Stale client cleanup**: follows gta_ws_bridge.py pattern -- catches exceptions during send, removes failed clients
- **Config version counter**: monotonic counter for WebSocket message ordering (Pitfall 3)
- **Connection caps**: max 20 browser + 10 training clients (T-05-06 DoS mitigation)
- **Utility methods**: `send_param_ack`, `send_restore_ack`, `forward_restore_to_train`

### JSONLCollector (dashboard/collector.py)

- **Polling loop**: 1.5s interval, scans all 4 module training_data directories
- **Byte-offset tracking**: `file_positions` dict maps file paths to byte offsets, reads only new lines
- **Partial write handling**: on JSONDecodeError, breaks and retries from last_good_pos on next poll (Pitfall 1)
- **Row classification**: by `type` field -- "metric" (default), "decision_counts", "embedding", "prediction"
- **Embedding BLOB conversion**: struct.pack float32 for z_t embeddings
- **Database delegation**: all SQLite writes via database.py helpers (insert_session, insert_metrics_batch, insert_decision_counts, insert_embeddings, insert_predictions) -- no raw SQL (T-05-05)
- **Session auto-creation**: on first JSONL line from a new file, creates session record and broadcasts event
- **Broadcast limiting**: only last 10 rows per push to browser clients

### MODULE_DIRS Mapping

| Module | Directory |
|--------|-----------|
| encoder_intuition | main_model/training_data |
| reward_head | reward_head/training_data |
| action_planner | action_planner/training_data |
| metacontroller | metacontroller/training_data |

## Deviations from Plan

### Auto-added Issues

**1. [Rule 3 - Blocking] Created database.py stub for parallel development**
- **Found during:** Task 2
- **Issue:** Plan 01 (parallel worktree) creates dashboard/database.py, but this worktree needs it for imports
- **Fix:** Created minimal stub with correct function signatures that raises NotImplementedError
- **Files created:** dashboard/database.py
- **Impact:** Stub will be overwritten when Plan 01 merges; all 5 helper function signatures match Plan 01 interface spec

**2. [Rule 2 - Missing] Created dashboard/__init__.py package marker**
- **Found during:** Task 1
- **Issue:** dashboard/ directory needed __init__.py for Python package imports
- **Fix:** Created minimal __init__.py
- **Files created:** dashboard/__init__.py

## Known Stubs

| File | Line | Description | Resolution |
|------|------|-------------|------------|
| dashboard/database.py | all | Stub with NotImplementedError -- placeholder for Plan 01's real implementation | Plan 05-01 merge provides full implementation |

## Verification Results

- `from dashboard.ws_manager import WSManager` -- exits 0
- `from dashboard.collector import JSONLCollector, MODULE_DIRS` -- exits 0
- `grep -c "broadcast_" dashboard/ws_manager.py` -- returns 10 (at least 4 required)
- `grep -c "from dashboard.database import" dashboard/collector.py` -- returns 1
- No raw SQL (self.db.execute / self.db.executemany) in collector.py -- confirmed
- struct.pack present for embedding BLOB conversion -- confirmed
- JSONDecodeError with break (not continue) -- confirmed
- last_good_pos byte tracking -- confirmed

## Self-Check: PASSED

All 4 created files exist on disk. Both task commits (adcc5fe, d7f4944) found in git log. SUMMARY.md created.
