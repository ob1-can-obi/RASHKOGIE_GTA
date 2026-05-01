# Phase 5: Training Dashboard - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-01
**Phase:** 05-training-dashboard
**Areas discussed:** Live data pipeline, Hyperparameter hot-reload, Session storage & comparison, Frontend approach, Weight storage, Embedding/prediction viz

---

## Live Data Pipeline

### Metrics store

| Option | Description | Selected |
|--------|-------------|----------|
| File polling | Training scripts keep writing JSONL. FastAPI watches files and pushes via SSE. No changes to train.py. | |
| In-process API push | Training scripts POST metrics to FastAPI. Tighter coupling, lower latency. | |
| Shared SQLite | Training scripts write to SQLite DB. Dashboard queries it. Richer queries. | ✓ |

**User's choice:** Shared SQLite
**Notes:** None

### Ingestion method

| Option | Description | Selected |
|--------|-------------|----------|
| Direct writes from train.py | Each train.py writes to SQLite alongside JSONL. Couples training to DB schema. | |
| Collector process | Background process watches JSONL and ingests into SQLite. Train scripts unchanged. | ✓ |

**User's choice:** Collector process
**Notes:** None

### Push method

| Option | Description | Selected |
|--------|-------------|----------|
| SSE | Server-Sent Events. FastAPI polls SQLite, pushes to browser. One-directional. | |
| WebSocket | Bidirectional connection. More complex but enables future features. | ✓ |

**User's choice:** WebSocket
**Notes:** None

### Poll frequency

| Option | Description | Selected |
|--------|-------------|----------|
| 1-2 seconds | Near-real-time without excessive I/O. | ✓ |
| Sub-second (250ms) | Very responsive but may cause I/O contention. | |

**User's choice:** 1-2 seconds
**Notes:** None

### Collector hosting

| Option | Description | Selected |
|--------|-------------|----------|
| Inside FastAPI | Background asyncio task. One process to start. | ✓ |
| Standalone script | Separate process. More isolated but two processes to manage. | |

**User's choice:** Inside FastAPI
**Notes:** None

### JSONL schema extension

| Option | Description | Selected |
|--------|-------------|----------|
| Add embedding snapshots | Periodically dump z_t vectors for PCA viz. | |
| Add decision distributions | Log EXPLORE/ROLLBACK/INTERRUPT/COMMIT_NEXT counts per batch. | |
| Both embeddings + decisions | Full observability for all dashboard requirements. | ✓ |

**User's choice:** Both embeddings + decisions
**Notes:** None

### Embedding snapshot frequency

| Option | Description | Selected |
|--------|-------------|----------|
| Every N steps (e.g., 500) | ~50 vectors per 25K-step run. Enough for PCA trends. | ✓ |
| Per epoch boundary | Fewer points, aligned with training boundaries. | |

**User's choice:** Every N steps (e.g., 500)
**Notes:** None

---

## Hyperparameter Hot-Reload

### Delivery mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| Config file polling | Dashboard writes training_config.json. Scripts re-read periodically. | |
| WebSocket command | Dashboard sends changes over WS. Instant effect. | ✓ |
| Signal-based | SIGUSR1 triggers config reload. Unix-only. | |

**User's choice:** WebSocket command
**Notes:** None

### Reloadable parameters

| Option | Description | Selected |
|--------|-------------|----------|
| Core four (DASH-04) | lr, entropy_coeff, think_cost, batch_size | |
| Core four + convergence | Add convergence thresholds and patience | ✓ |
| Everything in training_config.json | All parameters tunable | |

**User's choice:** Core four + convergence
**Notes:** None

### WebSocket direction

| Option | Description | Selected |
|--------|-------------|----------|
| Train scripts connect to dashboard | Dashboard is WS server. Scripts are clients. | ✓ |
| Dashboard connects to train scripts | Each script runs its own WS server. | |

**User's choice:** You decide — Claude chose "Train scripts connect to dashboard" (simpler architecture)
**Notes:** None

---

## Session Storage & Comparison

### Session definition

| Option | Description | Selected |
|--------|-------------|----------|
| Per module run | Each train.py invocation is its own session. | ✓ |
| Full pipeline run | Entire pipeline is one session. | |
| Both levels | Pipeline contains module runs, zoom in/out. | |

**User's choice:** Per module run
**Notes:** User initially didn't understand the question. Clarified with plain-language explanation of what a "session" means in context.

### Overlay count

| Option | Description | Selected |
|--------|-------------|----------|
| 2 sessions side-by-side | Simple, clean. | |
| Up to 4-5 overlaid | Multiple curves on same chart. | ✓ |

**User's choice:** Up to 4-5 overlaid
**Notes:** None

---

## Frontend Approach

### Technology

| Option | Description | Selected |
|--------|-------------|----------|
| Vanilla HTML + JS + Chart.js | Single HTML file, no build step. | |
| Jinja2 templates + Chart.js | Server-rendered, template reuse. | |
| React/Vue SPA | Full framework, more powerful. | ✓ |

**User's choice:** React/Vue SPA
**Notes:** None

### Framework

| Option | Description | Selected |
|--------|-------------|----------|
| React | Larger ecosystem, more charting libraries. | |
| Vue | Simpler, vue-chartjs available. | ✓ |

**User's choice:** Vue
**Notes:** None

### Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Tabbed single-page | Tabs across top, switch views. | |
| Sidebar navigation | Sidebar with sections, main content area. | ✓ |

**User's choice:** Sidebar navigation
**Notes:** None

### Remote access

| Option | Description | Selected |
|--------|-------------|----------|
| LAN access (bind 0.0.0.0) | Any device on local network can access. | ✓ |
| Tailscale/ZeroTier VPN | Private VPN mesh, secure. | |
| SSH tunnel | Forward port via SSH. | |

**User's choice:** LAN access (bind 0.0.0.0)
**Notes:** User asked about remote operation — training on one machine, dashboard on another.

### Authentication

| Option | Description | Selected |
|--------|-------------|----------|
| No auth | Home/lab network, open access. | |
| Simple password | Single shared password via env var. | ✓ |

**User's choice:** Simple password
**Notes:** None

---

## Weight Storage

### Dashboard capabilities

| Option | Description | Selected |
|--------|-------------|----------|
| View & download | List checkpoints, download from browser. | |
| View, download & restore | Plus roll back training to previous checkpoint. | ✓ |
| View, download & compare | Plus side-by-side metric comparison. | |

**User's choice:** View, download & restore
**Notes:** User asked for clarification on what "restore" means. Explained as loading a previous checkpoint back into a running training session.

### Storage location

| Option | Description | Selected |
|--------|-------------|----------|
| Per-session subdirectories | training_data/checkpoints/session_<id>/module_name.pt | ✓ |
| Flat per-module directory | module_dir/checkpoints/step_N.pt | |

**User's choice:** Per-session subdirectories
**Notes:** None

---

## Embedding & Prediction Visualization

### PCA embedding views

| Option | Description | Selected |
|--------|-------------|----------|
| Cluster evolution over training | z_t colored by step number. | |
| State-type clustering | z_t colored by driving context. | |
| Both views as tabs | Evolution + state-type separation. | ✓ |

**User's choice:** Both views as tabs
**Notes:** None

### Prediction quality views

| Option | Description | Selected |
|--------|-------------|----------|
| Predicted vs actual scatter | z_next_pred vs real z_{t+1} in 2D. | |
| Prediction error over time | MSE line chart over steps. | |
| Both scatter + error curve | Two complementary views. | ✓ |

**User's choice:** Both scatter + error curve
**Notes:** None

### PCA computation

| Option | Description | Selected |
|--------|-------------|----------|
| Server-side | scikit-learn on server, send 2D to browser. | ✓ |
| Browser-side | Raw 128-dim to browser, JS PCA. | |

**User's choice:** Server-side
**Notes:** None

---

## Claude's Discretion

- SQLite schema design (tables, indices, relationships)
- WebSocket message protocol format
- Vue component structure and state management
- Build tooling setup (Vite)
- Chart.js configurations and styling
- Training script WebSocket discovery/connection mechanism
- Driving context label derivation for embedding coloring
- Collector error handling

## Deferred Ideas

None — discussion stayed within phase scope
