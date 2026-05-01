# Phase 5: Training Dashboard - Research

**Researched:** 2026-05-01
**Domain:** FastAPI + Vue SPA real-time training dashboard with WebSocket, SQLite, Chart.js
**Confidence:** HIGH

## Summary

Phase 5 builds a FastAPI web server with a Vue 3 SPA frontend that provides live training metrics visualization, hyperparameter hot-reload, session history comparison, embedding PCA views, prediction quality plots, and checkpoint management. The dashboard bridges existing JSONL training output into a SQLite database via a background asyncio collector task, pushes updates to the browser over WebSocket, and receives parameter changes back through the same WebSocket channel. Training scripts also connect as WebSocket clients to receive hot-reload commands.

The project already has a well-established WebSocket pattern (`gta_ws_bridge.py` using the `websockets` library), JSONL data patterns across all training scripts, and a `training_config.json` / `training_status.json` contract from Phase 4. The dashboard reuses `load_training_config()` and `update_training_status()` from `training_utils.py` and adds write-back capability for hot-reload persistence. The frontend is a Vue 3 SPA built with Vite, using vue-chartjs (Chart.js 4.x wrapper) for all charting, with Pinia for state management and Vue Router for sidebar navigation.

**Primary recommendation:** Build the backend as a single `dashboard/server.py` FastAPI application with aiosqlite for async SQLite access, a background collector task launched via FastAPI's lifespan context manager, and WebSocket endpoints for both browser clients (push metrics) and training script clients (push parameter changes). Build the frontend as a separate `dashboard/frontend/` Vite+Vue 3 project that compiles to `dashboard/frontend/dist/` and is served by FastAPI's `StaticFiles(html=True)`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: SQLite database as the central metrics store
- D-02: Collector background asyncio task polls JSONL every 1-2s, ingests into SQLite
- D-03: Training scripts keep writing JSONL (no train.py modifications for output)
- D-04: WebSocket for live updates (not SSE)
- D-05: JSONL schema extended with embedding snapshots and decision distribution counts
- D-06: WebSocket command channel for hyperparameter hot-reload
- D-07: Hot-reloadable params: lr, entropy_coeff, think_cost, batch_size + convergence thresholds
- D-08: Dashboard writes changes to training_config.json for persistence
- D-09: SQLite stores session history
- D-10: Session comparison supports 4-5 overlaid curves
- D-11: Per-module session granularity
- D-12: Vue SPA with sidebar navigation
- D-13: 7 sidebar sections (Metrics, Decisions, Hyperparams, Sessions, Embeddings, Predictions, Weights)
- D-14: Chart.js via vue-chartjs
- D-15: FastAPI binds to 0.0.0.0
- D-16: Simple password auth via env var
- D-17: Checkpoints in per-session subdirectories
- D-18: Dashboard shows checkpoint list with metadata
- D-19: Download/restore checkpoints from browser via WebSocket
- D-20: PCA computed server-side (scikit-learn)
- D-21: Two embedding sub-views (cluster evolution, state-type clustering)
- D-22: Two prediction quality views (scatter plot, error over steps)

### Claude's Discretion
- SQLite schema design (tables, indices, relationships)
- WebSocket message protocol format (JSON structure for commands and updates)
- Vue component structure and state management (Pinia vs Vuex vs composables)
- Build tooling setup (Vite)
- Exact Chart.js chart configurations and styling
- How training scripts discover and connect to the dashboard WebSocket
- How driving context labels (straight/turn/braking) are derived from raw state for embedding coloring
- Collector error handling (what happens if JSONL is being written while collector reads)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DASH-01 | FastAPI web server serving training dashboard on localhost | FastAPI 0.136.x + uvicorn 0.46.x, lifespan pattern for background tasks, StaticFiles(html=True) for Vue SPA |
| DASH-02 | Live loss curves and reward curves | WebSocket push from collector -> browser, vue-chartjs reactive data watchers, SQLite metrics table |
| DASH-03 | Decision distribution histogram (EXPLORE/ROLLBACK/INTERRUPT/COMMIT_NEXT ratios over time) | JSONL schema extension (D-05), bar/stacked chart in Chart.js, SQLite decision_counts table |
| DASH-04 | Hyperparameter control panel (tune lr, entropy coeff, think_cost, batch size from browser) | WebSocket command channel (D-06), training_config.json write-back (D-08), training script WS client |
| DASH-05 | Training session history with per-session metrics summary | SQLite sessions table with session_id, module, start/end, final metrics |
| DASH-06 | Session comparison view (overlay loss curves from different runs) | Multiple dataset overlay in vue-chartjs Line chart, max 4-5 curves (D-10) |
| DASH-07 | Episode return tracking (primary health metric, not loss) | Separate episode_returns table or column in metrics, dedicated chart component |
| DASH-08 | Nodes expanded per token and search depth distribution | JSONL schema includes nodes_expanded and search_depth per metacontroller batch, histogram chart |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| JSONL ingestion / collector | API / Backend | -- | File I/O, SQLite writes, async polling -- pure server concern |
| Metrics storage | Database (SQLite) | -- | Central data store for all training metrics |
| Live data push | API / Backend (WebSocket) | Browser (WS client) | Server pushes new metrics; browser renders |
| Hyperparameter hot-reload | API / Backend (WebSocket) | Browser (form UI) | Browser sends command, server writes config + forwards to training scripts |
| PCA computation | API / Backend | -- | scikit-learn on 128-dim vectors -- too heavy for browser |
| Chart rendering | Browser / Client | -- | Chart.js + vue-chartjs -- client-side rendering of pushed data |
| Session history / comparison | API / Backend (REST) | Database (SQLite) | REST endpoints query SQLite, browser renders |
| Checkpoint management | API / Backend | Frontend (download button) | Server reads checkpoint files, streams to browser for download |
| Authentication | API / Backend | -- | Simple password check on WebSocket connect and HTTP endpoints |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | 0.136.1 | HTTP + WebSocket server | [VERIFIED: pip index] Standard Python async web framework, native WebSocket support |
| uvicorn | 0.46.0 | ASGI server | [VERIFIED: pip index] Default ASGI server for FastAPI |
| aiosqlite | 0.22.1 | Async SQLite access | [VERIFIED: pip index] asyncio bridge to sqlite3, WAL mode compatible |
| pydantic | 2.13.3 | Data validation / models | [VERIFIED: pip index] FastAPI's native validation layer |
| scikit-learn | 1.8.0 | PCA for embedding visualization | [VERIFIED: pip index] Server-side dimensionality reduction per D-20 |
| websockets | 16.0 | WebSocket protocol (already installed) | [VERIFIED: venv] Already in project for gta_ws_bridge.py |
| Vue | 3.5.33 | Frontend SPA framework | [VERIFIED: npm registry] Per D-12, reactive UI with composition API |
| Vite | 8.0.10 | Frontend build tool | [VERIFIED: npm registry] Fast dev server + production build for Vue |
| vue-chartjs | 5.3.3 | Chart.js wrapper for Vue 3 | [VERIFIED: npm registry] Per D-14, reactive chart updates on data change |
| chart.js | 4.5.1 | Charting library | [VERIFIED: npm registry] Loss curves, histograms, scatter plots per D-14 |
| vue-router | 5.0.6 | Client-side routing | [VERIFIED: npm registry] Sidebar navigation per D-13 |
| pinia | 3.0.4 | Vue state management | [VERIFIED: npm registry] Lightweight, Composition API native store |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| @vitejs/plugin-vue | 6.0.6 | Vite Vue 3 plugin | [VERIFIED: npm registry] Required for Vite + Vue 3 SFC compilation |
| pytest | 9.0.3 | Test framework | [VERIFIED: venv] Already installed, test dashboard backend endpoints |
| httpx | latest | Async HTTP test client | For testing FastAPI endpoints with pytest |
| pytest-asyncio | latest | Async test support | For testing async FastAPI/aiosqlite code |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| aiosqlite | sqlite3 (sync) | aiosqlite avoids blocking the event loop; sync sqlite3 would need run_in_executor wrapping |
| Pinia | Vuex 4 | Pinia is lighter, has better TypeScript/Composition API support, and is the officially recommended store for Vue 3 |
| chartjs-plugin-streaming | Manual reactive updates | The streaming plugin does NOT support Chart.js 4.x natively; vue-chartjs reactive data watchers handle 1-2s push intervals perfectly |
| SSE (Server-Sent Events) | WebSocket (chosen D-04) | WebSocket is bidirectional, needed for both metric push AND command channel; SSE would require a separate mechanism for commands |

**Installation (Python backend):**
```bash
pip install "fastapi>=0.136.0" "uvicorn[standard]>=0.46.0" "aiosqlite>=0.22.0" "pydantic>=2.13.0" "scikit-learn>=1.8.0"
```

**Installation (Vue frontend):**
```bash
cd dashboard/frontend
npm create vite@latest . -- --template vue
npm install vue-chartjs chart.js vue-router pinia
```

## Architecture Patterns

### System Architecture Diagram

```
                    Training Scripts (train.py)
                         |
           [write JSONL files to training_data/]
                         |
                         v
    +--------------------------------------------------+
    |              FastAPI Server (dashboard/)          |
    |                                                  |
    |  +-----------+     +----------+     +---------+  |
    |  | Collector |---->| SQLite   |<----| REST    |  |
    |  | (asyncio  |     | (WAL     |     | API     |  |
    |  |  bg task) |     |  mode)   |     | routes  |  |
    |  +-----------+     +----------+     +---------+  |
    |       |                                 |        |
    |       v                                 v        |
    |  +-----------+                   +----------+    |
    |  | WS Hub    |<--- push -------->| WS Hub   |    |
    |  | (browser  |     metrics       | (train   |    |
    |  |  clients) |                   |  clients)|    |
    |  +-----------+                   +----------+    |
    |       ^                               |          |
    |       |  param change command         |          |
    |       +----> write training_config -> +          |
    |              .json + forward cmd                 |
    +--------------------------------------------------+
         |                            |
         v                            v
    +-----------+              +-----------+
    | Browser   |              | train.py  |
    | (Vue SPA) |              | (WS       |
    | Chart.js  |              |  client)  |
    +-----------+              +-----------+
```

Data flow:
1. Training scripts write JSONL to `<module>/training_data/` (no changes to train.py output per D-03)
2. Collector background task polls JSONL every 1-2s, ingests new lines into SQLite
3. After ingestion, collector publishes new metrics to browser WebSocket clients
4. Browser renders charts via vue-chartjs reactive data
5. User changes hyperparameter in browser -> WebSocket command to server
6. Server writes to training_config.json (D-08) AND forwards command to training script WS clients (D-06)
7. Training scripts receive command, update live parameters without restart

### Recommended Project Structure

```
dashboard/
├── __init__.py
├── server.py           # FastAPI app, lifespan, mount points
├── collector.py        # JSONL -> SQLite ingestion background task
├── database.py         # SQLite schema, connection, queries (aiosqlite)
├── models.py           # Pydantic models for API and WebSocket messages
├── ws_manager.py       # WebSocket connection manager (browser + train clients)
├── routes/
│   ├── __init__.py
│   ├── metrics.py      # REST: GET /api/metrics, /api/metrics/latest
│   ├── sessions.py     # REST: GET /api/sessions, /api/sessions/{id}
│   ├── params.py       # REST: GET/PUT /api/params (hot-reload config)
│   ├── checkpoints.py  # REST: GET /api/checkpoints, download, restore
│   ├── embeddings.py   # REST: GET /api/embeddings/pca (server-side PCA)
│   └── predictions.py  # REST: GET /api/predictions (scatter + error data)
├── auth.py             # Simple password middleware (D-16)
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── App.vue
        ├── main.js
        ├── router.js        # 7 sidebar routes per D-13
        ├── stores/
        │   ├── metrics.js   # Pinia store: live metrics from WS
        │   ├── params.js    # Pinia store: hyperparameter state
        │   └── sessions.js  # Pinia store: session history
        ├── composables/
        │   └── useWebSocket.js  # WebSocket connection + reconnect
        ├── components/
        │   ├── Sidebar.vue
        │   ├── AuthGate.vue
        │   └── charts/
        │       ├── LossChart.vue
        │       ├── RewardChart.vue
        │       ├── DecisionHistogram.vue
        │       ├── EpisodeReturnChart.vue
        │       ├── NodesExpandedChart.vue
        │       ├── EmbeddingScatter.vue
        │       └── PredictionScatter.vue
        └── views/
            ├── MetricsView.vue      # DASH-02, DASH-07
            ├── DecisionsView.vue    # DASH-03, DASH-08
            ├── HyperparamsView.vue  # DASH-04
            ├── SessionsView.vue    # DASH-05, DASH-06
            ├── EmbeddingsView.vue  # D-21
            ├── PredictionsView.vue # D-22
            └── WeightsView.vue     # D-18, D-19
```

### Pattern 1: FastAPI Lifespan for Background Collector

**What:** Use FastAPI's lifespan async context manager to start the JSONL collector and WebSocket hub as background asyncio tasks.
**When to use:** Always -- the collector must run as long as the server is alive.

```python
# Source: https://fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init database, start collector
    db = await init_database()
    collector_task = asyncio.create_task(run_collector(db))
    app.state.db = db
    app.state.collector = collector_task
    yield
    # Shutdown: stop collector, close database
    collector_task.cancel()
    try:
        await collector_task
    except asyncio.CancelledError:
        pass
    await db.close()

app = FastAPI(lifespan=lifespan)
```

### Pattern 2: WebSocket Connection Manager with Auth

**What:** Manage browser and training-script WebSocket connections separately, with password auth on connect.
**When to use:** All WebSocket endpoints.

```python
# Source: https://fastapi.tiangolo.com/advanced/websockets/
from fastapi import WebSocket, WebSocketDisconnect, WebSocketException, status

class WSManager:
    def __init__(self):
        self.browser_clients: list[WebSocket] = []
        self.train_clients: list[WebSocket] = []

    async def connect_browser(self, ws: WebSocket, token: str):
        if token != os.environ.get("DASHBOARD_PASSWORD", ""):
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
        await ws.accept()
        self.browser_clients.append(ws)

    async def connect_train(self, ws: WebSocket):
        await ws.accept()
        self.train_clients.append(ws)

    async def broadcast_metrics(self, data: dict):
        """Push new metrics to all browser clients."""
        msg = json.dumps({"type": "metrics", "data": data})
        stale = []
        for ws in self.browser_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.browser_clients.remove(ws)

    async def broadcast_param_change(self, params: dict):
        """Forward param changes to all training script clients."""
        msg = json.dumps({"type": "param_update", "params": params})
        stale = []
        for ws in self.train_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.train_clients.remove(ws)
```

### Pattern 3: JSONL Collector with File Position Tracking

**What:** Background task that polls JSONL files, tracks last-read byte offset per file, and ingests only new lines.
**When to use:** The collector loop running every 1-2 seconds.

```python
# [ASSUMED] -- pattern based on training data convention in project
import json
import asyncio
from pathlib import Path

class JSONLCollector:
    def __init__(self, db, ws_manager, poll_interval=1.5):
        self.db = db
        self.ws = ws_manager
        self.poll_interval = poll_interval
        self.file_positions: dict[str, int] = {}  # path -> byte offset

    async def run(self):
        while True:
            try:
                await self._poll_all_modules()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logging.warning("Collector error: %s", e)
            await asyncio.sleep(self.poll_interval)

    async def _poll_all_modules(self):
        # Scan known training data directories
        for module_dir in ["main_model", "reward_head", "action_planner"]:
            data_dir = Path(module_dir) / "training_data"
            if not data_dir.exists():
                continue
            for jsonl_path in sorted(data_dir.glob("*.jsonl")):
                new_rows = self._read_new_lines(jsonl_path)
                if new_rows:
                    await self.db.insert_metrics(module_dir, new_rows)
                    await self.ws.broadcast_metrics({
                        "module": module_dir,
                        "rows": new_rows[-10:]  # last 10 for live chart
                    })

    def _read_new_lines(self, path: Path) -> list[dict]:
        """Read only new lines since last poll, handling partial writes."""
        key = str(path)
        offset = self.file_positions.get(key, 0)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Partial write -- stop here, retry next poll
                    break
            self.file_positions[key] = f.tell()
        return rows
```

### Pattern 4: Serving Vue SPA from FastAPI

**What:** Mount the compiled Vue dist/ as StaticFiles with `html=True` to handle client-side routing.
**When to use:** After API routes are defined, mount SPA as last route.

```python
# Source: https://fastapi.tiangolo.com/tutorial/static-files/
# IMPORTANT: mount AFTER all API routes to avoid shadowing
from fastapi.staticfiles import StaticFiles

# API routes first
app.include_router(metrics_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
# ... other API routes

# SPA mount LAST -- catches all non-API routes
app.mount("/", StaticFiles(directory="dashboard/frontend/dist", html=True), name="spa")
```

### Pattern 5: SQLite WAL Mode with aiosqlite

**What:** Enable WAL (Write-Ahead Logging) mode for concurrent reads during collector writes.
**When to use:** Database initialization.

```python
# Source: https://aiosqlite.omnilib.dev/ [CITED]
import aiosqlite

async def init_database(db_path="dashboard/training.db"):
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    # Create tables...
    await db.executescript(SCHEMA_SQL)
    await db.commit()
    return db
```

### Anti-Patterns to Avoid

- **Polling SQLite from the browser:** Never expose raw SQL queries to the frontend. All data goes through REST endpoints or WebSocket push.
- **Blocking the event loop with sync sqlite3:** Always use aiosqlite. The collector runs in the same async event loop as FastAPI; sync sqlite3 would freeze all request handling during writes.
- **Mutating training_config.json without locking:** Multiple browser clients could send param changes simultaneously. Use a simple asyncio.Lock around the read-modify-write cycle.
- **Large WebSocket payloads:** Don't send the entire metrics history on every push. Send only new/delta data; the frontend accumulates state in Pinia.
- **Chart.js re-creation on every data push:** vue-chartjs supports reactive data updates -- mutate the chart data arrays, don't recreate the chart component. Use `ref()` for chart data and update in-place.
- **Putting API routes after StaticFiles mount:** FastAPI evaluates routes in order; if `StaticFiles(html=True)` is mounted at `/` first, it will catch `/api/...` requests. Always mount API routes before the SPA catch-all.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Async SQLite access | Manual run_in_executor wrapping | aiosqlite | Handles thread safety, cursor management, connection lifecycle |
| PCA dimensionality reduction | Custom matrix math | scikit-learn PCA | Numerically stable, handles edge cases (zero variance, degenerate data) |
| WebSocket protocol | Raw socket handling | FastAPI WebSocket + websockets | Connection lifecycle, ping/pong, graceful disconnect, error handling |
| Chart rendering | Custom canvas/SVG drawing | Chart.js + vue-chartjs | Axes, legends, tooltips, zoom, responsive layout all built-in |
| Vue state management | Custom event bus / provide/inject for global state | Pinia | DevTools support, type safety, SSR-ready, official recommendation |
| Password hashing/comparison | string == comparison | hmac.compare_digest | Timing-safe comparison prevents timing attacks on auth token |
| JSONL partial line handling | Assume complete lines | try/except JSONDecodeError + break | Training scripts flush JSONL mid-line; partial reads are expected |
| File change detection | inotify/watchdog | Byte-offset tracking in collector | Cross-platform, simpler than filesystem watchers, works on Windows (target platform) |

**Key insight:** The dashboard is a monitoring/control overlay on existing training infrastructure. It should never modify the core training data flow (JSONL files, training_status.json structure, checkpoint format). All extensions (WebSocket client in train.py, embedding snapshots, decision counts) are additive, not replacing existing patterns.

## Common Pitfalls

### Pitfall 1: Partial JSONL Reads from Concurrent Writers
**What goes wrong:** Collector reads a JSONL file while a training script is mid-write, getting a truncated JSON line like `{"step": 100, "loss": 0.0`.
**Why it happens:** Training scripts flush to JSONL without exclusive file locks. The collector polls every 1-2s and may catch a write in progress.
**How to avoid:** In the collector, wrap each `json.loads()` in a try/except. On `JSONDecodeError`, stop processing that file and retry from the same position next poll. The next poll will read the complete line.
**Warning signs:** `JSONDecodeError` warnings in collector logs. If they happen on every single poll, something else is wrong.

### Pitfall 2: SQLite Write Contention
**What goes wrong:** Multiple concurrent writers (collector + REST API writing params) cause `database is locked` errors.
**Why it happens:** SQLite allows only one writer at a time. Default busy timeout is 0 (immediate failure).
**How to avoid:** Enable WAL mode (`PRAGMA journal_mode=WAL`) and set a busy timeout (`PRAGMA busy_timeout=5000`). WAL allows concurrent readers while one writer is active. Use a single aiosqlite connection (not a pool) for the entire FastAPI process -- aiosqlite serializes writes through its internal thread.
**Warning signs:** `OperationalError: database is locked` in logs.

### Pitfall 3: WebSocket Message Ordering
**What goes wrong:** Browser receives a param_update acknowledgment before the corresponding metrics update, leading to UI showing stale data alongside "saved" confirmation.
**Why it happens:** WebSocket messages are ordered per-connection but the param write and the next collector push are independent events.
**How to avoid:** Include a `config_version` counter in both param updates and metric pushes. The frontend can show "applying..." until a metrics push arrives with the new config_version.
**Warning signs:** UI shows contradictory state (e.g., lr=0.001 in the control panel but lr=0.003 in the metrics header).

### Pitfall 4: Vue SPA Routing vs. FastAPI API Routes
**What goes wrong:** Navigating to `/sessions` in the browser returns a 404 from FastAPI instead of the Vue SPA.
**Why it happens:** `StaticFiles(html=True)` is mounted at `/` but API routes at `/api/` shadow it, or the mount order is wrong.
**How to avoid:** All API routes use `/api/` prefix. WebSocket routes use `/ws/`. Mount `StaticFiles(html=True)` at `/` as the LAST route. Vue Router handles all non-API paths.
**Warning signs:** 404 errors on browser refresh when not on the root URL.

### Pitfall 5: Chart.js Memory Leak on Rapid Updates
**What goes wrong:** Browser memory grows unboundedly as metrics accumulate over hours-long training sessions.
**Why it happens:** Chart.js stores all data points in memory. With 1-2s updates over 8+ hours, that is 15,000-30,000 points per chart.
**How to avoid:** Cap the displayed window to the last N points (e.g., 2000) in the Pinia store. Show a "full history" option that queries the REST API with pagination. The SQLite database stores everything; the browser only holds the working window.
**Warning signs:** Browser tab memory growing past 500MB during long sessions.

### Pitfall 6: Training Script WebSocket Client Blocking the Training Loop
**What goes wrong:** The training script's WebSocket client blocks the training loop while waiting for or processing messages.
**Why it happens:** Training scripts use synchronous PyTorch operations. A naive WebSocket client in the same thread would block.
**How to avoid:** Run the WebSocket client in a separate daemon thread. Use `threading.Thread(daemon=True)` with a simple polling loop: `websocket.recv()` with timeout, check a `threading.Event` for shutdown. The main training loop checks a shared `dict` for updated params at the start of each batch.
**Warning signs:** Training throughput drops significantly when dashboard is connected.

### Pitfall 7: Stale Frontend Build
**What goes wrong:** Code changes to Vue components don't appear in the browser.
**Why it happens:** FastAPI serves the compiled `dist/` folder, which is a snapshot from `npm run build`. Development changes are not reflected until rebuild.
**How to avoid:** During development, run Vite dev server on port 5173 with a proxy to FastAPI on port 8000. In production, `npm run build` and restart FastAPI. Add a build script / makefile target.
**Warning signs:** Changes to `.vue` files have no effect in the browser.

## Code Examples

### SQLite Schema Design

```sql
-- [ASSUMED] -- schema designed for the 7 dashboard sections
-- training.db

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,     -- e.g., "20260501_120000"
    module TEXT NOT NULL,                -- e.g., "encoder_intuition"
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT DEFAULT 'running',       -- running, converged, stopped
    final_metric REAL,
    total_steps INTEGER,
    checkpoint_path TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    module TEXT NOT NULL,
    step INTEGER NOT NULL,
    epoch INTEGER,
    loss REAL,
    reward REAL,
    episode_return REAL,               -- DASH-07: primary health metric
    grad_norm REAL,
    clipped INTEGER,                   -- 0 or 1
    lr REAL,                           -- current learning rate
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS decision_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    explore INTEGER DEFAULT 0,         -- DASH-03
    rollback INTEGER DEFAULT 0,
    interrupt INTEGER DEFAULT 0,
    commit_next INTEGER DEFAULT 0,
    nodes_expanded INTEGER,            -- DASH-08
    search_depth INTEGER,              -- DASH-08
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    z_t BLOB NOT NULL,                 -- 128-dim float32 vector (512 bytes)
    driving_context TEXT,              -- "straight" | "turn" | "braking"
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    z_next_pred BLOB NOT NULL,         -- predicted next-state (128-dim)
    z_next_real BLOB NOT NULL,         -- actual next-state (128-dim)
    mse REAL NOT NULL,                 -- per-sample prediction MSE
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Indices for common queries
CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id, step);
CREATE INDEX IF NOT EXISTS idx_metrics_module ON metrics(module, step);
CREATE INDEX IF NOT EXISTS idx_decision_session ON decision_counts(session_id, step);
CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings(session_id, step);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id, step);
CREATE INDEX IF NOT EXISTS idx_sessions_module ON sessions(module);
```

### WebSocket Message Protocol

```python
# [ASSUMED] -- protocol designed for the 7 dashboard sections

# Server -> Browser messages:
{
    "type": "metrics_update",
    "module": "encoder_intuition",
    "data": [
        {"step": 100, "loss": 0.045, "reward": null, "episode_return": null,
         "grad_norm": 0.32, "clipped": false, "lr": 0.0003}
    ]
}

{
    "type": "decision_update",
    "data": [
        {"step": 100, "explore": 12, "rollback": 3, "interrupt": 1,
         "commit_next": 8, "nodes_expanded": 5, "search_depth": 3}
    ]
}

{
    "type": "param_ack",
    "params": {"lr": 0.0001, "entropy_coeff": 0.03},
    "config_version": 42
}

{
    "type": "session_started",
    "session_id": "20260501_120000",
    "module": "encoder_intuition"
}

# Browser -> Server messages:
{
    "type": "set_params",
    "params": {"lr": 0.0001, "module": "encoder_intuition"}
}

{
    "type": "restore_checkpoint",
    "session_id": "20260501_120000",
    "module": "encoder_intuition"
}

# Server -> Training Script messages (via separate WS endpoint):
{
    "type": "param_update",
    "params": {"lr": 0.0001, "entropy_coeff": 0.03, "think_cost": 0.01, "batch_size": 8}
}

{
    "type": "restore_checkpoint",
    "checkpoint_path": "main_model/checkpoints/session_20260501_120000"
}
```

### Vue SPA Main Structure

```javascript
// Source: vue-chartjs.org/guide/ [CITED]
// dashboard/frontend/src/main.js
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'

const routes = [
  { path: '/', redirect: '/metrics' },
  { path: '/metrics', component: () => import('./views/MetricsView.vue') },
  { path: '/decisions', component: () => import('./views/DecisionsView.vue') },
  { path: '/hyperparams', component: () => import('./views/HyperparamsView.vue') },
  { path: '/sessions', component: () => import('./views/SessionsView.vue') },
  { path: '/embeddings', component: () => import('./views/EmbeddingsView.vue') },
  { path: '/predictions', component: () => import('./views/PredictionsView.vue') },
  { path: '/weights', component: () => import('./views/WeightsView.vue') },
]

const router = createRouter({ history: createWebHistory(), routes })
const pinia = createPinia()

createApp(App).use(pinia).use(router).mount('#app')
```

### Reactive Chart Update Pattern

```javascript
// Source: vue-chartjs.org/guide/ [CITED]
// dashboard/frontend/src/components/charts/LossChart.vue
<template>
  <Line :data="chartData" :options="chartOptions" />
</template>

<script setup>
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import { Chart, registerables } from 'chart.js'
import { useMetricsStore } from '../stores/metrics'

Chart.register(...registerables)

const store = useMetricsStore()

const chartData = computed(() => ({
  labels: store.steps,
  datasets: [{
    label: 'Loss',
    data: store.losses,
    borderColor: '#7c3aed',
    tension: 0.1,
    pointRadius: 0,
  }]
}))

const chartOptions = {
  responsive: true,
  animation: { duration: 0 },  // disable animation for live updates
  scales: {
    x: { title: { display: true, text: 'Step' } },
    y: { title: { display: true, text: 'Loss' }, beginAtZero: false }
  }
}
</script>
```

### Training Script WebSocket Client (Hot-Reload Receiver)

```python
# [ASSUMED] -- pattern for training scripts to receive param updates
# This is a minimal addition to existing train.py scripts

import threading
import json
import websockets.sync.client  # websockets 16.0 sync client

class ParamReceiver:
    """Background thread that listens for param updates from dashboard."""

    def __init__(self, dashboard_url="ws://localhost:8000/ws/train"):
        self.url = dashboard_url
        self.params = {}       # shared dict, read by training loop
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                with websockets.sync.client.connect(self.url) as ws:
                    ws.send(json.dumps({"type": "register", "module": "encoder_intuition"}))
                    while not self._stop.is_set():
                        try:
                            msg = ws.recv(timeout=1.0)
                            data = json.loads(msg)
                            if data.get("type") == "param_update":
                                self.params.update(data["params"])
                        except TimeoutError:
                            continue
            except Exception:
                if not self._stop.is_set():
                    import time
                    time.sleep(5)  # reconnect backoff

# Usage in training loop:
# receiver = ParamReceiver()
# receiver.start()
# ... in training loop:
# if receiver.params:
#     config.update(receiver.params)
#     receiver.params.clear()
```

### Server-Side PCA for Embedding Visualization

```python
# Source: scikit-learn PCA documentation [CITED: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html]
import numpy as np
from sklearn.decomposition import PCA

async def compute_embedding_pca(db, session_id, max_points=500):
    """Compute 2D PCA projection of 128-dim z_t embeddings."""
    rows = await db.execute_fetchall(
        "SELECT step, z_t, driving_context FROM embeddings "
        "WHERE session_id = ? ORDER BY step LIMIT ?",
        (session_id, max_points)
    )
    if len(rows) < 2:
        return {"points": [], "explained_variance": []}

    vectors = np.array([
        np.frombuffer(row[1], dtype=np.float32) for row in rows
    ])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "step": rows[i][0],
            "context": rows[i][2],
        }
        for i in range(len(rows))
    ]
    return {
        "points": points,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }
```

### Simple Password Auth Middleware

```python
# [ASSUMED] -- minimal auth per D-16
import os
import hmac
from fastapi import Request, HTTPException, WebSocket, WebSocketException, status
from starlette.middleware.base import BaseHTTPMiddleware

DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")

class PasswordAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for static files and health check
        if request.url.path.startswith("/api/"):
            token = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if not DASHBOARD_PASSWORD:
                pass  # No password set = open access
            elif not hmac.compare_digest(token, DASHBOARD_PASSWORD):
                raise HTTPException(status_code=401, detail="Unauthorized")
        return await call_next(request)

# WebSocket auth via query parameter
async def verify_ws_token(token: str):
    if DASHBOARD_PASSWORD and not hmac.compare_digest(token, DASHBOARD_PASSWORD):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `@app.on_event("startup")` | `lifespan` async context manager | FastAPI 0.93+ (2023) | Must use lifespan for startup/shutdown logic; old events deprecated |
| Vuex | Pinia | Vue 3 (2022) | Pinia is official Vue 3 store; simpler API, better TypeScript support |
| Chart.js 2.x + vue-chartjs 3.x | Chart.js 4.x + vue-chartjs 5.x | 2023 | Tree-shakeable, ESM-only imports, register pattern required |
| chartjs-plugin-streaming 1.x | Reactive data watchers | Chart.js 4 (2023) | Plugin incompatible with Chart.js 4; manual reactive updates preferred |
| websockets async API only | websockets 16.0 with sync client API | websockets 13+ (2024) | `websockets.sync.client` available for training scripts (no asyncio needed) |

**Deprecated/outdated:**
- `@app.on_event("startup")` / `@app.on_event("shutdown")` -- replaced by `lifespan` context manager
- Vuex 4 -- replaced by Pinia as official recommendation
- `chartjs-plugin-streaming` -- incompatible with Chart.js 4; use reactive data updates
- Vue Options API for new components -- Composition API with `<script setup>` is standard

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | SQLite schema design (5 tables: sessions, metrics, decision_counts, embeddings, predictions) | Code Examples | Schema may need additional tables or different column types; low risk since schema is in Claude's discretion |
| A2 | WebSocket message protocol format (JSON with type field) | Code Examples | Protocol is in Claude's discretion; any reasonable JSON format works |
| A3 | Training script WebSocket client uses `websockets.sync.client` in daemon thread | Code Examples | websockets 16.0 sync client API may differ slightly; need to verify import path |
| A4 | Driving context labels derived from speed + steering angle thresholds | Architecture Patterns | Exact derivation logic not specified; "braking" = brake > 0.3, "turn" = abs(steer) > 0.2, else "straight" |
| A5 | Collector polls `main_model/training_data/`, `reward_head/training_data/`, `action_planner/training_data/` | Architecture Patterns | JSONL files may be in different subdirectories; collector should be configurable |
| A6 | Frontend build output goes to `dashboard/frontend/dist/` | Architecture Patterns | Vite default output is `dist/`; may need `vite.config.js` outDir adjustment |

## Open Questions (RESOLVED)

1. **JSONL schema extension for embedding snapshots (D-05)** (RESOLVED)
   - What we know: z_t is [1, 128] float32 from encoder, snapshots every ~500 steps
   - What's unclear: Should the embedding be serialized as a JSON array of 128 floats in JSONL (readable but large), or as base64-encoded binary (compact but opaque)?
   - Recommendation: JSON array of floats in JSONL (human-readable, consistent with existing JSONL patterns), convert to binary blob when inserting into SQLite for storage efficiency
   - **Resolution:** JSON array of 128 floats in JSONL. Collector (Plan 02) converts to binary BLOB via struct.pack on SQLite insert. Plan 06 Task 1 implements the write in main_model/train.py using z_t.detach().cpu().squeeze().tolist().
2. **Metacontroller JSONL training metrics source** (RESOLVED)
   - What we know: Metacontroller trains via RL in the frame loop (trainer.py), not via a standalone train.py with JSONL output
   - What's unclear: How does the collector get metacontroller metrics (episode returns, decision distributions, nodes expanded)?
   - Recommendation: Add a JSONL writer to the frame loop / trainer that logs per-episode summaries to `metacontroller/training_data/`. This is a minimal, additive modification to the training loop (not changing output format, just adding a new file)
   - **Resolution:** Plan 06 Task 2 modifies TrainingState.__init__ to accept jsonl_dir parameter. Writes decision_counts and metric JSONL rows in update_metapolicy_batch(). Callers pass jsonl_dir to enable output.
3. **Checkpoint download via WebSocket vs HTTP** (RESOLVED)
   - What we know: D-19 says download/restore from browser via WebSocket
   - What's unclear: Streaming large .pt files (50-200MB) over WebSocket is awkward; HTTP with Content-Disposition is the standard pattern for file downloads
   - Recommendation: Use HTTP endpoint for checkpoint download (`GET /api/checkpoints/{session_id}/{file}`), WebSocket only for the restore command. D-19 intent is "from the browser", not necessarily "via WebSocket protocol for the binary transfer"
   - **Resolution:** Plan 04 implements HTTP FileResponse at GET /api/checkpoints/{module}/{session_id}/{filename}. WebSocket used only for restore_checkpoint command.
## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | Backend | Yes | 3.12.9 | -- |
| FastAPI | Server | No (not installed) | -- | Install: `pip install fastapi` |
| uvicorn | ASGI server | No (not installed) | -- | Install: `pip install uvicorn[standard]` |
| aiosqlite | Async SQLite | No (not installed) | -- | Install: `pip install aiosqlite` |
| pydantic | Data validation | No (not installed) | -- | Install: `pip install pydantic` (comes with FastAPI) |
| scikit-learn | PCA embeddings | No (not installed) | -- | Install: `pip install scikit-learn` |
| websockets | WebSocket protocol | Yes | 16.0 | -- |
| torch | Checkpoint inspection | Yes | 2.11.0 | -- |
| pytest | Testing | Yes | 9.0.3 | -- |
| Node.js | Vue frontend build | Yes | 25.7.0 | -- |
| npm | Package management | Yes | 11.10.1 | -- |

**Missing dependencies with no fallback:**
- FastAPI, uvicorn, aiosqlite, pydantic, scikit-learn must be installed. All are pip-installable.

**Missing dependencies with fallback:**
- None -- all missing deps have straightforward `pip install` solutions.

**Installation command:**
```bash
pip install "fastapi>=0.136.0" "uvicorn[standard]>=0.46.0" "aiosqlite>=0.22.0" "scikit-learn>=1.8.0"
```
Note: pydantic is installed as a FastAPI dependency automatically.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | `tests/conftest.py` (exists, needs dashboard fixtures) |
| Quick run command | `python -m pytest tests/test_dashboard.py -x` |
| Full suite command | `python -m pytest tests/ -x` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DASH-01 | FastAPI serves dashboard on localhost | integration | `python -m pytest tests/test_dashboard.py::test_server_starts -x` | Wave 0 |
| DASH-02 | Live loss/reward curves via WebSocket | integration | `python -m pytest tests/test_dashboard.py::test_ws_metrics_push -x` | Wave 0 |
| DASH-03 | Decision distribution histogram updates | integration | `python -m pytest tests/test_dashboard.py::test_decision_counts_ingestion -x` | Wave 0 |
| DASH-04 | Hyperparameter changes take effect | integration | `python -m pytest tests/test_dashboard.py::test_param_hot_reload -x` | Wave 0 |
| DASH-05 | Session history with metrics summary | unit | `python -m pytest tests/test_dashboard.py::test_session_history -x` | Wave 0 |
| DASH-06 | Session comparison overlay | unit | `python -m pytest tests/test_dashboard.py::test_session_comparison -x` | Wave 0 |
| DASH-07 | Episode return tracking | unit | `python -m pytest tests/test_dashboard.py::test_episode_return_ingestion -x` | Wave 0 |
| DASH-08 | Nodes expanded + search depth tracking | unit | `python -m pytest tests/test_dashboard.py::test_nodes_expanded_ingestion -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_dashboard.py -x`
- **Per wave merge:** `python -m pytest tests/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_dashboard.py` -- covers DASH-01 through DASH-08 backend validation
- [ ] `tests/conftest.py` additions -- dashboard-specific fixtures (mock db, test JSONL files, test client)
- [ ] Framework install: `pip install httpx pytest-asyncio` -- for async FastAPI test client

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | Yes | Simple password via env var DASHBOARD_PASSWORD (D-16), hmac.compare_digest for timing-safe comparison |
| V3 Session Management | No | No user sessions -- single shared password, stateless per-request auth |
| V4 Access Control | Yes | All API endpoints behind password middleware; training script WS endpoint has no auth (local network only) |
| V5 Input Validation | Yes | Pydantic models validate all REST/WebSocket inputs; param ranges validated server-side |
| V6 Cryptography | No | No encryption of data at rest; LAN-only access per project constraints |

### Known Threat Patterns for FastAPI + WebSocket + SQLite

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| SQL injection in metrics queries | Tampering | Parameterized queries via aiosqlite (never f-string SQL) |
| WebSocket message injection | Tampering | Pydantic validation of all incoming WS messages; reject unknown types |
| Timing attack on password comparison | Information Disclosure | hmac.compare_digest instead of == comparison |
| Path traversal in checkpoint download | Information Disclosure | Validate checkpoint paths are within allowed directories; reject `..` components |
| Unbounded WebSocket connections (DoS) | Denial of Service | Cap max connections in WSManager; reject after limit |
| Large file upload via WS (DoS) | Denial of Service | Set max_size on WebSocket; reject messages exceeding threshold |
| Training config injection | Tampering | Validate param values against allowed ranges (e.g., lr > 0, batch_size > 0) before writing to training_config.json |

## Sources

### Primary (HIGH confidence)
- [pip index] -- FastAPI 0.136.1, uvicorn 0.46.0, aiosqlite 0.22.1, scikit-learn 1.8.0, pydantic 2.13.3 version verification
- [npm registry] -- Vue 3.5.33, Vite 8.0.10, vue-chartjs 5.3.3, chart.js 4.5.1, vue-router 5.0.6, pinia 3.0.4 version verification
- [Project venv] -- websockets 16.0, pytest 9.0.3, torch 2.11.0, Python 3.12.9 already installed
- [FastAPI official docs: /advanced/websockets/] -- WebSocket endpoint patterns, ConnectionManager
- [FastAPI official docs: /advanced/events/] -- Lifespan context manager pattern
- [FastAPI official docs: /tutorial/static-files/] -- StaticFiles mount with html=True

### Secondary (MEDIUM confidence)
- [vue-chartjs.org/guide/] -- Reactive chart data updates, Chart.register pattern
- [aiosqlite.omnilib.dev] -- aiosqlite connection and WAL mode usage
- [GitHub: nagix/chartjs-plugin-streaming] -- Confirmed NOT compatible with Chart.js 4.x
- [npm: @nckrtl/chartjs-plugin-streaming] -- Chart.js 4 compatible fork exists but unnecessary for 1-2s polling

### Tertiary (LOW confidence)
- WebSearch results for JSONL concurrent reading patterns -- community patterns, not official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all versions verified against registries, compatibility confirmed
- Architecture: HIGH -- patterns from official FastAPI/Vue docs, existing project patterns (gta_ws_bridge.py) validate approach
- Pitfalls: HIGH -- based on known SQLite concurrency constraints, WebSocket lifecycle issues, and Chart.js memory characteristics
- Security: MEDIUM -- ASVS L1 controls identified, but simple password auth is minimal (appropriate for LAN-only research tool per D-16)

**Research date:** 2026-05-01
**Valid until:** 2026-06-01 (30 days -- stable ecosystem, no expected breaking changes)
