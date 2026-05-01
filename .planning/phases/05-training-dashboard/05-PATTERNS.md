# Phase 5: Training Dashboard - Pattern Map

**Mapped:** 2026-05-01
**Files analyzed:** 28 new files (14 backend Python, 14 frontend JS/Vue)
**Analogs found:** 22 / 28

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `dashboard/__init__.py` | config | -- | `tests/__init__.py` | exact |
| `dashboard/server.py` | controller | request-response + streaming | `gta_stream/gta_ws_bridge.py` | role-match |
| `dashboard/collector.py` | service | file-I/O + event-driven | `reward_head/stats.py` | role-match |
| `dashboard/database.py` | service | CRUD | `training_utils.py` | role-match |
| `dashboard/models.py` | model | transform | `training_utils.py` (status template) | partial |
| `dashboard/ws_manager.py` | service | streaming (WebSocket) | `gta_stream/gta_ws_bridge.py` | exact |
| `dashboard/auth.py` | middleware | request-response | -- (no existing auth) | no-analog |
| `dashboard/routes/__init__.py` | config | -- | `tests/__init__.py` | exact |
| `dashboard/routes/metrics.py` | route | CRUD (read) | `coordinator.py` (load_status) | partial |
| `dashboard/routes/sessions.py` | route | CRUD (read) | `coordinator.py` (cmd_status) | partial |
| `dashboard/routes/params.py` | route | request-response | `training_utils.py` (load_training_config) | role-match |
| `dashboard/routes/checkpoints.py` | route | file-I/O | `main_model/train.py` (save/load checkpoint) | role-match |
| `dashboard/routes/embeddings.py` | route | transform | -- (no PCA analog) | no-analog |
| `dashboard/routes/predictions.py` | route | CRUD (read) | `reward_head/stats.py` (predicted vs actual) | partial |
| `tests/test_dashboard.py` | test | request-response | `tests/test_coordinator.py` | exact |
| `dashboard/frontend/src/main.js` | config | -- | -- (no Vue analog) | no-analog |
| `dashboard/frontend/src/router.js` | config | -- | -- (no Vue analog) | no-analog |
| `dashboard/frontend/src/App.vue` | component | -- | -- (no Vue analog) | no-analog |
| `dashboard/frontend/src/stores/metrics.js` | store | event-driven | -- (no Vue analog) | no-analog |
| `dashboard/frontend/src/stores/params.js` | store | request-response | -- (no Vue analog) | no-analog |
| `dashboard/frontend/src/stores/sessions.js` | store | CRUD | -- (no Vue analog) | no-analog |
| `dashboard/frontend/src/composables/useWebSocket.js` | hook | streaming | `gta_stream/gta_ws_bridge.py` (client pattern) | partial |
| `dashboard/frontend/src/components/Sidebar.vue` | component | -- | -- | no-analog |
| `dashboard/frontend/src/components/AuthGate.vue` | component | -- | -- | no-analog |
| `dashboard/frontend/src/components/charts/*.vue` (7 files) | component | transform | -- | no-analog |
| `dashboard/frontend/src/views/*.vue` (7 files) | component | request-response | -- | no-analog |
| `training_config.json` (MODIFIED) | config | -- | `training_config.json` (self) | exact |
| `main_model/train.py` (MODIFIED) | service | batch | `main_model/train.py` (self) | exact |

## Pattern Assignments

### `dashboard/server.py` (controller, request-response + streaming)

**Analog:** `gta_stream/gta_ws_bridge.py` (lines 1-365)

**Imports pattern** (adapt from `gta_ws_bridge.py` lines 36-48 and `coordinator.py` lines 16-28):
```python
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
```

**Async lifecycle pattern** (adapt from `gta_ws_bridge.py` lines 123-161, `start()` + `stop()`):
The GTA bridge uses `start()` / `stop()` methods with `asyncio.create_task`. Dashboard should use FastAPI's lifespan context manager, which is the same concept:
```python
# gta_ws_bridge.py lines 123-161 -- analogous lifecycle:
async def start(self) -> None:
    self._reader.start()
    self._state_task = asyncio.create_task(self._pump_state(), name="gta-state-pump")
    self._server = await websockets_module.serve(...)

async def stop(self) -> None:
    self._stop.set()
    self._reader.stop()
    if self._state_task is not None:
        self._state_task.cancel()
        try:
            await self._state_task
        except asyncio.CancelledError:
            pass
```
Dashboard adaptation: wrap in `@asynccontextmanager async def lifespan(app)`, create collector task at startup, cancel at shutdown.

**CLI entry point pattern** (from `gta_ws_bridge.py` lines 341-364):
```python
# gta_ws_bridge.py lines 341-364
def main() -> None:
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    try:
        asyncio.run(_run_bridge(args))
    except KeyboardInterrupt:
        pass
```
Dashboard adaptation: use `uvicorn.run(app, host=args.host, port=args.port)` instead of raw asyncio.

---

### `dashboard/ws_manager.py` (service, streaming)

**Analog:** `gta_stream/gta_ws_bridge.py` (lines 103-308)

**Client tracking pattern** (lines 116-119):
```python
# gta_ws_bridge.py lines 116-119
self._clients: Set[Any] = set()
self._latest_state: Optional[Dict[str, Any]] = None
self._state_seq = 0
self._stop = asyncio.Event()
```
Dashboard adaptation: separate `browser_clients` set and `train_clients` set.

**Broadcast pattern** (lines 293-301):
```python
# gta_ws_bridge.py lines 293-301
async def _broadcast(self, message: str) -> None:
    stale = []
    for client in list(self._clients):
        try:
            await client.send(message)
        except Exception:
            stale.append(client)
    for client in stale:
        self._clients.discard(client)
```
Dashboard adaptation: copy this pattern for both `broadcast_metrics()` (to browser clients) and `broadcast_param_change()` (to training script clients). Use the same stale-client cleanup.

**Client connect/disconnect pattern** (lines 190-212):
```python
# gta_ws_bridge.py lines 190-212
async def _handle_client(self, websocket) -> None:
    client_id = secrets.token_hex(4)
    self._clients.add(websocket)
    self._log.info("Client connected | id=%s | clients=%d", client_id, len(self._clients))
    try:
        await self._send_json(websocket, {"type": "hello", "client_id": client_id, ...})
        async for raw in websocket:
            await self._handle_message(websocket, raw)
    except Exception as exc:
        self._log.info("Client disconnected | id=%s | reason=%s", client_id, exc)
    finally:
        self._clients.discard(websocket)
```

**Message dispatch pattern** (lines 214-291):
```python
# gta_ws_bridge.py lines 214-291
async def _handle_message(self, websocket, raw: str) -> None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await self._send_error(websocket, "invalid JSON")
        return
    if not isinstance(data, dict):
        await self._send_error(websocket, "message must be a JSON object")
        return
    msg_type = str(data.get("type", "")).strip().lower()
    if msg_type == "ping":
        await self._send_json(websocket, {"type": "pong", ...})
        return
    # ... further type-based dispatch
    await self._send_error(websocket, f"unknown message type: {msg_type or '<missing>'}")
```
Dashboard adaptation: handle `set_params`, `restore_checkpoint` from browser clients; handle `register` from training script clients.

**JSON serialization pattern** (lines 303-307):
```python
# gta_ws_bridge.py lines 303-307
async def _send_json(self, websocket, payload: Dict[str, Any]) -> None:
    await websocket.send(json.dumps(payload, separators=(",", ":")))

async def _send_error(self, websocket, message: str) -> None:
    await self._send_json(websocket, {"type": "error", "message": message})
```

---

### `dashboard/collector.py` (service, file-I/O + event-driven)

**Analog:** `reward_head/stats.py` (lines 1-206)

**JSONL reading pattern** (lines 36-45):
```python
# reward_head/stats.py lines 36-45
rows = []
for path in sorted(data_dir.glob("*.jsonl")):
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_file"] = path.name
            row["_line"] = line_number
            rows.append(row)
```
Dashboard adaptation: add byte-offset tracking per file (`f.seek(offset)` / `f.tell()`) so the collector only reads new lines. Add `try/except json.JSONDecodeError` with break on partial line (from `main_model/train.py` load_data pattern).

**Also from `main_model/train.py` (lines 62-92) -- robust JSONL parsing:**
```python
# main_model/train.py lines 77-91
for path in sorted(data_dir.glob("*.jsonl")):
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                logging.warning(
                    "Skipping malformed line %d in %s", line_number, path.name
                )
```
Dashboard adaptation: same try/except but `break` instead of `continue` on `JSONDecodeError` (partial write from concurrent training script -- retry next poll).

**Background loop pattern** (from `gta_ws_bridge.py` lines 166-188):
```python
# gta_ws_bridge.py lines 166-188
async def _pump_state(self) -> None:
    while not self._stop.is_set():
        try:
            state = await asyncio.to_thread(self._reader.get, 1.0)
        except queue.Empty:
            continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log.warning("State pump error: %s", exc)
            await asyncio.sleep(1.0)
            continue
        # ... process and broadcast
```
Dashboard adaptation: replace `to_thread` with direct file I/O (JSONL reads are fast), use `await asyncio.sleep(self.poll_interval)` at end of each loop.

**Module directory scanning pattern** (from `coordinator.py` lines 36-57):
```python
# coordinator.py lines 36-57
STAGE_ORDER = ["encoder_intuition", "reward_head", "action_planner", "metacontroller"]
STAGE_SCRIPTS = {
    "encoder_intuition": "python main_model/train.py",
    "reward_head": "python reward_head/train.py",
    "action_planner": "python action_planner/train.py",
    "metacontroller": "(uses existing frame_loop.py -- run GTA agent)",
}
```
Dashboard adaptation: use same stage names to know which `training_data/` directories to scan (`main_model/training_data/`, `reward_head/training_data/`, `action_planner/training_data/`).

---

### `dashboard/database.py` (service, CRUD)

**Analog:** `training_utils.py` (lines 1-244)

**File path resolution pattern** (lines 106-107):
```python
# training_utils.py lines 106-107
_PROJECT_ROOT = Path(__file__).resolve().parent
```
Dashboard adaptation: `_DASHBOARD_DIR = Path(__file__).resolve().parent` for locating the SQLite database file.

**JSON config read pattern** (lines 109-137):
```python
# training_utils.py lines 109-137
def load_training_config(config_path=None):
    if config_path is None:
        config_path = _PROJECT_ROOT / "training_config.json"
    else:
        config_path = Path(config_path)
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Malformed training_config.json: {e.msg}", e.doc, e.pos
        )
    return config
```
Dashboard adaptation: database.py should follow the same pattern for path defaulting and error handling, but use aiosqlite instead of sync file I/O.

**Status read-modify-write pattern** (lines 180-243):
```python
# training_utils.py lines 180-243
def update_training_status(stage_name, status, metric=None, ...):
    # Read existing or create from template
    if status_path.exists():
        try:
            with open(status_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = json.loads(json.dumps(_INITIAL_STATUS_TEMPLATE))
    else:
        data = json.loads(json.dumps(_INITIAL_STATUS_TEMPLATE))
    # ... modify
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
```
Dashboard adaptation: the params route uses this same read-modify-write pattern for `training_config.json` hot-reload. Wrap in `asyncio.Lock` for concurrent access.

---

### `dashboard/routes/params.py` (route, request-response)

**Analog:** `training_utils.py` (lines 109-137 for reading, lines 180-243 for writing)

**Config read pattern** (from `training_utils.py` lines 109-137):
```python
# training_utils.py lines 109-137
def load_training_config(config_path=None):
    if config_path is None:
        config_path = _PROJECT_ROOT / "training_config.json"
    else:
        config_path = Path(config_path)
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(...)
    return config
```
Dashboard adaptation: wrap in a FastAPI GET endpoint. For PUT, add write-back (same pattern as `update_training_status`).

**Config structure** (from `training_config.json`):
```json
{
    "encoder_intuition": {
        "lr": 3e-4,
        "batch_size": 8,
        "max_grad_norm": 0.5,
        "eval_every_n_steps": 100,
        "convergence": {
            "metric": "mse",
            "threshold": 0.05,
            "patience": 10,
            "mode": "min"
        }
    }
}
```
Dashboard adaptation: the hyperparameter control panel reads this structure. Hot-reloadable keys per D-07: `lr`, `entropy_coeff`, `think_cost`, `batch_size`, plus `convergence.threshold`, `convergence.patience`.

---

### `dashboard/routes/checkpoints.py` (route, file-I/O)

**Analog:** `main_model/train.py` (lines 99-199)

**Checkpoint directory structure** (lines 119-148):
```python
# main_model/train.py lines 119-148
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = Path(checkpoint_dir) / f"session_{session_id}"
session_dir.mkdir(parents=True, exist_ok=True)
torch.save(encoder_state, session_dir / "encoder_weights.pt")
torch.save({"model_state_dict": intuition_mlp.state_dict()}, session_dir / "intuition_mlp.pt")
torch.save({"model_state_dict": token_embed.state_dict()}, session_dir / "token_embed.pt")
torch.save({"optimizer_state_dict": optimizer.state_dict(), "step_count": step_count}, session_dir / "optimizer.pt")
```
Dashboard adaptation: listing checkpoints means scanning `<module>/checkpoints/session_*/` for `.pt` files with stat metadata. Download endpoint streams the `.pt` file via HTTP (not WebSocket). Restore sends a WebSocket command to the training script.

**Also from `metacontroller/trainer.py` (lines 887-991) -- TrainingState checkpoint:**
```python
# metacontroller/trainer.py lines 914-933
session_dir = Path(checkpoint_dir) / f"session_{session_id}"
session_dir.mkdir(parents=True, exist_ok=True)
meta_path = session_dir / "meta_mlp.pt"
torch.save({
    "model_state_dict": meta_mlp.state_dict(),
    "optimizer_state_dict": self.optimizer_meta.state_dict(),
    "step_count": self.step_count,
    ...
}, meta_path)
```
Dashboard uses same `session_<id>` naming convention to display and manage checkpoints.

---

### `dashboard/routes/metrics.py` (route, CRUD read)

**Analog:** `coordinator.py` (lines 242-296)

**Status reading and display pattern** (lines 242-296):
```python
# coordinator.py lines 242-296
def cmd_status(status):
    run_id = status.get("pipeline_run_id", "none")
    for stage_name in STAGE_ORDER:
        stage = status.get("stages", {}).get(stage_name, {})
        display = STAGE_DISPLAY.get(stage_name, stage_name)
        st = stage.get("status", "pending")
        metric_val = stage.get("final_metric")
        # ... format and display
```
Dashboard adaptation: REST endpoint queries SQLite instead of reading JSON file. Returns paginated metrics by session/module/step range.

---

### `dashboard/routes/sessions.py` (route, CRUD read)

**Analog:** `coordinator.py` (lines 34-57 for stage mapping, lines 242-296 for display)

**Stage display names** (lines 38-43):
```python
# coordinator.py lines 38-43
STAGE_DISPLAY = {
    "encoder_intuition": "Encoder + Intuition",
    "reward_head": "Reward Head",
    "action_planner": "Action Planner",
    "metacontroller": "Metacontroller",
}
```
Dashboard adaptation: reuse these display names in the sessions REST endpoint response.

---

### `tests/test_dashboard.py` (test, request-response)

**Analog:** `tests/test_coordinator.py` (lines 1-80)

**Test file structure** (lines 1-35):
```python
# tests/test_coordinator.py lines 1-35
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coordinator import (
    load_status,
    save_status,
    cmd_status,
    ...
)
```
Dashboard adaptation: add `httpx`, `pytest-asyncio` imports. Use `httpx.AsyncClient` with FastAPI's `TestClient` pattern for async endpoint testing.

**Helper factory pattern** (lines 42-77):
```python
# tests/test_coordinator.py lines 42-77
def _make_status(overrides=None, frozen=None):
    """Create a fresh pipeline status dict with optional stage overrides."""
    status = {
        "pipeline_run_id": "20260501_143022",
        "stages": {
            "encoder_intuition": {"status": "pending", ...},
            ...
        },
        "frozen_modules": frozen or [],
    }
    if overrides:
        for stage_name, stage_overrides in overrides.items():
            ...
    return status
```
Dashboard adaptation: create `_make_test_db()` helper that creates an in-memory SQLite database with test data. Create `_make_test_jsonl()` helper that writes synthetic JSONL files.

**conftest.py fixture pattern** (from `tests/conftest.py` lines 1-19):
```python
# tests/conftest.py lines 1-19
import sys
from pathlib import Path
import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
for d in (METACONTROLLER_DIR, ...):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
```
Dashboard adaptation: add dashboard-specific fixtures (mock DB, test JSONL files, FastAPI test client) following same conftest pattern.

---

### `dashboard/models.py` (model, transform)

**Analog:** `training_utils.py` (lines 144-177 for status template)

**Data template pattern** (lines 144-177):
```python
# training_utils.py lines 144-177
_INITIAL_STATUS_TEMPLATE = {
    "pipeline_run_id": None,
    "stages": {
        "encoder_intuition": {
            "status": "pending",
            "started_at": None,
            "converged_at": None,
            "final_metric": None,
            "total_steps": None,
            "checkpoint": None,
        },
        ...
    },
    "frozen_modules": [],
}
```
Dashboard adaptation: Pydantic models replace raw dicts. Define `MetricsRow`, `SessionInfo`, `ParamUpdate`, `WebSocketMessage` etc. as Pydantic BaseModel classes.

---

### `main_model/train.py` (MODIFIED -- add embedding snapshots + WS client)

**Self-analog:** `main_model/train.py` (lines 332-414)

**Training loop where snapshots should be added** (lines 370-378):
```python
# main_model/train.py lines 370-378
# Logging
if step_count % 10 == 0:
    logging.info(
        "step=%d loss=%.6f grad_norm=%.4f clipped=%s",
        step_count,
        total_loss.item(),
        grad_norm.item(),
        clipped,
    )
```
Dashboard adaptation: add JSONL metric writes here (step, loss, grad_norm, clipped, lr). Add embedding snapshot every ~500 steps (serialize `z_t.detach().numpy().tolist()` to JSONL). Add WebSocket ParamReceiver from research patterns.

---

## Shared Patterns

### Logging Setup
**Source:** `gta_stream/gta_ws_bridge.py` lines 74-100
**Apply to:** `dashboard/server.py`, `dashboard/collector.py`
```python
# gta_ws_bridge.py lines 74-100
def _build_logger(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gta_ws_bridge.log")
    logger = logging.getLogger("gta5_ws_bridge")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
```

### JSONL Parsing (Robust)
**Source:** `main_model/train.py` lines 62-92
**Apply to:** `dashboard/collector.py`
```python
# main_model/train.py lines 77-91
for path in sorted(data_dir.glob("*.jsonl")):
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed line %d in %s", line_number, path.name)
```

### Project Root Resolution
**Source:** `training_utils.py` line 106, `coordinator.py` lines 23-24
**Apply to:** All dashboard backend files
```python
# training_utils.py line 106
_PROJECT_ROOT = Path(__file__).resolve().parent

# coordinator.py lines 23-24
_ROOT = Path(__file__).resolve().parent
```

### Training Config Read/Write
**Source:** `training_utils.py` lines 109-137 (read), lines 180-243 (write)
**Apply to:** `dashboard/routes/params.py`, `dashboard/server.py`
```python
# training_utils.py lines 109-137
def load_training_config(config_path=None):
    if config_path is None:
        config_path = _PROJECT_ROOT / "training_config.json"
    else:
        config_path = Path(config_path)
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Malformed training_config.json: {e.msg}", e.doc, e.pos)
    return config
```

### Checkpoint Session Directory Convention
**Source:** `main_model/train.py` lines 119-121, `metacontroller/trainer.py` lines 914-918
**Apply to:** `dashboard/routes/checkpoints.py`
```python
# Shared pattern: session_id = timestamp, session_dir = checkpoints/session_<id>/
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
session_dir = Path(checkpoint_dir) / f"session_{session_id}"
session_dir.mkdir(parents=True, exist_ok=True)
```

### Stage Names and Display Names
**Source:** `coordinator.py` lines 36-57
**Apply to:** `dashboard/collector.py`, `dashboard/routes/sessions.py`, `dashboard/routes/metrics.py`
```python
# coordinator.py lines 36-57
STAGE_ORDER = ["encoder_intuition", "reward_head", "action_planner", "metacontroller"]
STAGE_DISPLAY = {
    "encoder_intuition": "Encoder + Intuition",
    "reward_head": "Reward Head",
    "action_planner": "Action Planner",
    "metacontroller": "Metacontroller",
}
```

### Test File Structure
**Source:** `tests/conftest.py` lines 1-19, `tests/test_coordinator.py` lines 1-35
**Apply to:** `tests/test_dashboard.py`, `tests/conftest.py` (additions)
```python
# tests/conftest.py lines 1-19
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for d in (...):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
```

---

## No Analog Found

Files with no close match in the codebase (planner should use RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `dashboard/auth.py` | middleware | request-response | No authentication middleware exists in the project. Use RESEARCH.md Pattern 2 (password auth with `hmac.compare_digest`) and RESEARCH.md Security section. |
| `dashboard/routes/embeddings.py` | route | transform | No PCA or dimensionality reduction code exists. Use RESEARCH.md "Server-Side PCA" code example with scikit-learn. |
| `dashboard/frontend/src/main.js` | config | -- | No Vue.js code exists in the project. Use RESEARCH.md "Vue SPA Main Structure" code example. |
| `dashboard/frontend/src/router.js` | config | -- | No Vue Router code exists. Use RESEARCH.md code example (7 sidebar routes). |
| `dashboard/frontend/src/App.vue` | component | -- | No Vue SFC exists. Use RESEARCH.md for structure; sidebar + router-view layout. |
| `dashboard/frontend/src/stores/*.js` | store | event-driven | No Pinia stores exist. Use RESEARCH.md patterns for reactive data from WebSocket. |
| `dashboard/frontend/src/composables/useWebSocket.js` | hook | streaming | Partial analog in `gta_ws_bridge.py` (client handling), but no JS WebSocket client code exists. Use RESEARCH.md WebSocket message protocol. |
| `dashboard/frontend/src/components/Sidebar.vue` | component | -- | No Vue component analog. Build from sidebar nav spec in D-13. |
| `dashboard/frontend/src/components/AuthGate.vue` | component | -- | No auth UI exists. Simple password prompt component. |
| `dashboard/frontend/src/components/charts/*.vue` (7 files) | component | transform | No chart components exist. Use RESEARCH.md "Reactive Chart Update Pattern" with vue-chartjs. |
| `dashboard/frontend/src/views/*.vue` (7 files) | component | request-response | No Vue views exist. Each view composes chart components + store data. |

---

## Metadata

**Analog search scope:** Project root (`/Users/jishnuraviprolu/Desktop/RASHKOGIE_GTA/`), including `gta_stream/`, `training_utils.py`, `coordinator.py`, `reward_head/`, `main_model/`, `metacontroller/`, `tests/`
**Files scanned:** 12 existing source files read in detail
**Pattern extraction date:** 2026-05-01
