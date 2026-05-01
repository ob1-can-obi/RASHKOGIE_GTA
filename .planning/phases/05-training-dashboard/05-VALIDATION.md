---
phase: 5
slug: training-dashboard
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 5 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | none -- Wave 0 installs |
| **Quick run command** | `python -m pytest tests/test_dashboard.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -x -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_dashboard.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -x -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-T1 | 01 | 1 | DASH-01 | T-05-01, T-05-02, T-05-03 | Auth middleware, parameterized SQL, timing-safe compare | unit | `python -c "from dashboard.database import init_database, SCHEMA_SQL; from dashboard.models import MetricsRow; from dashboard.auth import PasswordAuthMiddleware; print('OK')"` | Plan 01 creates | ⬜ pending |
| 01-T2 | 01 | 1 | DASH-01 | T-05-04 | Server starts on 0.0.0.0, health endpoint | integration | `python -m pytest tests/test_dashboard.py::test_server_starts tests/test_dashboard.py::test_database_tables_created -x -v` | Plan 01 creates | ⬜ pending |
| 02-T1 | 02 | 1 | DASH-02, DASH-03 | T-05-06 | WS connection caps, stale client cleanup | unit | `python -c "from dashboard.ws_manager import WSManager; m = WSManager(); print('OK')"` | Plan 02 creates | ⬜ pending |
| 02-T2 | 02 | 1 | DASH-02, DASH-03, DASH-07, DASH-08 | T-05-05, T-05-07 | Database.py helpers for all SQL, byte-offset tracking | unit | `python -c "from dashboard.collector import JSONLCollector, MODULE_DIRS; print('OK')"` | Plan 02 creates | ⬜ pending |
| 03-T1 | 03 | 2 | DASH-02, DASH-03, DASH-04, DASH-07, DASH-08 | T-05-08 | Database.py helpers for metrics, Pydantic validation for params | unit | `python -c "from dashboard.routes.metrics import metrics_router; from dashboard.routes.params import params_router; print('OK')"` | Plan 03 creates | ⬜ pending |
| 03-T2 | 03 | 2 | DASH-02, DASH-03, DASH-04, DASH-07, DASH-08 | T-05-09, T-05-10, T-05-11 | WS token verification, param broadcast, config lock | integration | `python -m pytest tests/test_dashboard.py -x -v --timeout=30` | Plan 03 creates | ⬜ pending |
| 04-T1 | 04 | 2 | DASH-05, DASH-06 | T-05-12, T-05-14 | Database.py helpers for sessions, path traversal protection | unit | `python -c "from dashboard.routes.sessions import sessions_router; from dashboard.routes.checkpoints import checkpoints_router; print('OK')"` | Plan 04 creates | ⬜ pending |
| 04-T2 | 04 | 2 | DASH-05, DASH-06 | T-05-13, T-05-15 | Database.py helpers for embeddings/predictions, PCA point cap | unit | `python -c "from dashboard.routes.embeddings import embeddings_router; from dashboard.routes.predictions import predictions_router; print('OK')"` | Plan 04 creates | ⬜ pending |
| 05-T1 | 05 | 3 | DASH-01 through DASH-08 | T-05-16, T-05-18 | Auth gate, MAX_DISPLAY_POINTS cap | integration | `cd dashboard/frontend && npm install && ls src/main.js src/router.js src/App.vue && echo 'OK'` | Plan 05 creates | ⬜ pending |
| 05-T2a | 05 | 3 | DASH-02, DASH-03, DASH-07, DASH-08 | T-05-18 | Chart components with correct colors | unit | `ls dashboard/frontend/src/components/charts/*.vue` | Plan 05 creates | ⬜ pending |
| 05-T2b | 05 | 3 | DASH-01 through DASH-08 | T-05-17, T-05-19 | All views, frontend built, server mount active | integration | `cd dashboard/frontend && npm run build && ls dist/index.html` | Plan 05 creates | ⬜ pending |
| 05-CP | 05 | 3 | DASH-01 through DASH-08 | -- | Human visual verification of complete dashboard | manual | Human opens browser to localhost:8000 | N/A | ⬜ pending |
| 06-T1 | 06 | 1 | DASH-02, DASH-07 | T-05-20, T-05-21 | JSONL writes, WS param receiver, additive changes | integration | `python -c "import ast; source=open('main_model/train.py').read(); assert 'write_jsonl' in source and 'DashboardParamReceiver' in source; print('OK')"` | Modifies existing | ⬜ pending |
| 06-T2 | 06 | 1 | DASH-03, DASH-08 | T-05-22 | JSONL decision_counts, WS param receiver, backward-compatible | integration | `python -c "source=open('metacontroller/trainer.py').read(); assert 'decision_counts' in source and 'start_param_receiver' in source; print('OK')"` | Modifies existing | ⬜ pending |

*Status: ⬜ pending -- ✅ green -- ❌ red -- ⚠ flaky*

---

## Wave 0 Requirements

- [x] `tests/test_dashboard.py` -- stubs for DASH-01 through DASH-08 (created by Plan 01 Task 2)
- [x] `tests/conftest.py` -- shared fixtures including test_db, test_client, test_client_with_auth (created by Plan 01 Task 2)

*Existing pytest infrastructure from Phase 4 covers framework installation.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live chart updates in browser | DASH-02 | Requires visual verification of Chart.js rendering | Open dashboard, start training, verify curves update |
| Remote LAN access | DASH-01 | Requires second device on same network | Access dashboard from another device on LAN |
| Checkpoint download from browser | DASH-05 | Requires browser download verification | Click download, verify .pt file received |
| Decision histogram visual layout | DASH-03 | Stacked bar chart visual correctness | Open Decisions view, verify 4-color stacked bars |
| Hyperparameter hot-reload end-to-end | DASH-04 | Requires running training script + dashboard simultaneously | Change param in browser, verify training script receives update |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending execution
