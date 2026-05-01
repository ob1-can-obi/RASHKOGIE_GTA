# Phase 5: Training Dashboard - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a FastAPI web server with a Vue SPA frontend that provides live training metrics, decision distribution histograms, hyperparameter controls, session history with comparison, weight/checkpoint management, and embedding/prediction visualizations — all accessible remotely over LAN during any training session.

</domain>

<decisions>
## Implementation Decisions

### Live data pipeline
- **D-01:** SQLite database as the central metrics store — all training metrics flow through SQLite, not raw JSONL reads
- **D-02:** Collector process runs inside the FastAPI server as a background asyncio task — polls JSONL files every 1-2 seconds and ingests new rows into SQLite
- **D-03:** Training scripts continue writing JSONL as they do now — no modifications to train.py for metric output. Collector bridges JSONL → SQLite
- **D-04:** WebSocket pushes updates from server to browser — bidirectional connection, not SSE
- **D-05:** JSONL schema extended: add embedding snapshots (z_t [128] vectors) every ~500 steps, and metacontroller decision distribution counts (EXPLORE/ROLLBACK/INTERRUPT/COMMIT_NEXT) per batch

### Hyperparameter hot-reload
- **D-06:** WebSocket command channel — dashboard sends param changes over WebSocket to training scripts. Training scripts connect to the dashboard's WebSocket server as clients on startup
- **D-07:** Hot-reloadable parameters: lr, entropy_coeff, think_cost, batch_size (DASH-04 core four) PLUS convergence thresholds and patience from training_config.json
- **D-08:** Dashboard also writes changes to training_config.json for persistence — WebSocket for immediate effect, file for restart resilience

### Session storage & comparison
- **D-09:** Same SQLite database stores session history — each train.py invocation is one session with a unique ID, module name, start time, end time, final metrics
- **D-10:** Session comparison supports up to 4-5 overlaid loss curves on the same chart
- **D-11:** Per-module session granularity — compare "encoder run #1 vs encoder run #3", not whole pipeline runs

### Frontend approach
- **D-12:** Vue SPA frontend with sidebar navigation layout
- **D-13:** Sidebar sections: Metrics (live loss/reward curves), Decisions (distribution histogram), Hyperparams (control panel), Sessions (history + comparison), Embeddings (PCA viz), Predictions (quality viz), Weights (checkpoint management)
- **D-14:** Chart.js for charting (via vue-chartjs)

### Remote access
- **D-15:** FastAPI binds to 0.0.0.0 — accessible from any device on the local network
- **D-16:** Simple password authentication via environment variable (e.g., DASHBOARD_PASSWORD). No user accounts — single shared password

### Weight storage & management
- **D-17:** Checkpoint files stored in per-session subdirectories: training_data/checkpoints/session_<id>/module_name.pt
- **D-18:** Dashboard shows checkpoint list per module per session with file size, timestamp, and metrics at save time
- **D-19:** Download checkpoints from browser and restore (roll back training to a previous checkpoint via WebSocket command)

### Embedding & prediction visualization
- **D-20:** PCA computed server-side using scikit-learn on 128-dim z_t vectors, 2D coordinates sent to browser
- **D-21:** Two embedding sub-views: (a) cluster evolution over training — z_t colored by training step, (b) state-type clustering — z_t colored by driving context (straight, turn, braking)
- **D-22:** Two prediction quality views: (a) predicted vs actual scatter plot (z_next_pred vs real z_{t+1}, PCA-reduced), (b) prediction error (MSE) over training steps as a line chart

### Claude's Discretion
- SQLite schema design (tables, indices, relationships)
- WebSocket message protocol format (JSON structure for commands and updates)
- Vue component structure and state management (Pinia vs Vuex vs composables)
- Build tooling setup (Vite)
- Exact Chart.js chart configurations and styling
- How training scripts discover and connect to the dashboard WebSocket
- How driving context labels (straight/turn/braking) are derived from raw state for embedding coloring
- Collector error handling (what happens if JSONL is being written while collector reads)

</decisions>

<specifics>
## Specific Ideas

- The dashboard is the control center for the entire training pipeline — not just a viewer but an operator interface where you can see what's happening, tune parameters live, roll back to good checkpoints, and understand what the model has learned via embeddings
- Remote access is key — train on the Windows/GPU machine, monitor and control from any device on the LAN (laptop, phone)
- Embedding PCA should show the encoder learning to separate different driving situations over time — this is a direct measure of whether z_t representations are useful
- Prediction scatter plot near the diagonal means the intuition head is accurately forecasting next states — this validates the joint encoder+intuition training from Phase 4

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project context
- `.planning/PROJECT.md` — Architecture overview, training order, module dependencies
- `.planning/REQUIREMENTS.md` — DASH-01 through DASH-08 requirement definitions
- `.planning/ROADMAP.md` — Phase 5 success criteria (5 criteria that must be TRUE)

### Prior phase context (training infrastructure this dashboard monitors)
- `.planning/phases/04-module-training-pipelines/04-CONTEXT.md` — Training pipeline decisions, JSONL format, training_config.json, training_status.json, coordinator design
- `.planning/phases/01-training-correctness/01-CONTEXT.md` — Penalty signals, entropy annealing, duration normalization (parameters that appear in hyperparameter panel)

### Source files to build on
- `training_utils.py` — `load_training_config()`, `update_training_status()`, `ConvergenceDetector` — reuse for dashboard backend
- `training_config.json` — Central config that dashboard reads and writes for hot-reload
- `training_status.json` — Pipeline state that dashboard displays
- `coordinator.py` — Stage ordering, dependency tracking, display names — reference for dashboard pipeline view
- `reward_head/stats.py` — Existing stats/graph generation from JSONL — reference for metric processing patterns
- `main_model/train.py` — Encoder+intuition training script (writes JSONL, needs embedding snapshots added)
- `reward_head/train.py` — Reward head training script (writes JSONL)
- `action_planner/train.py` — Action planner training script (writes JSONL)
- `metacontroller/trainer.py` — TrainingState class, batch update (metacontroller RL metrics source)
- `gta_stream/gta_ws_bridge.py` — Existing WebSocket bridge pattern (reference for WS implementation)

### Existing training data structure
- `tokenizer/training_data/captures/` — JSONL session file naming pattern
- `reward_head/training_data/` — Per-module training data layout

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `load_training_config()` (training_utils.py): reads training_config.json — dashboard backend reuses this, adds write-back for hot-reload
- `update_training_status()` (training_utils.py): updates training_status.json — dashboard reads this for pipeline overview
- `gta_ws_bridge.py`: existing WebSocket implementation using websockets library — reference for dashboard WS server
- `reward_head/stats.py`: JSONL parsing and metric extraction — reference for collector's ingestion logic
- `ConvergenceDetector` (training_utils.py): convergence logic — dashboard can display convergence progress

### Established Patterns
- JSONL format with session-timestamped filenames (tokenizer/training_data/captures/)
- Stateless weight pattern: modules use `output, mlp = module(input, mlp=None)` — checkpoint restore must handle this
- Per-module training_data/ subdirectories
- Adam optimizer with gradient clipping (max_norm=0.5)

### Integration Points
- Train.py scripts need: (a) WebSocket client to receive hot-reload commands, (b) embedding snapshot writes every ~500 steps, (c) decision distribution logging per batch
- Checkpoint save currently in TrainingState — needs to also save to per-session subdirectory
- training_config.json is the shared contract between dashboard and training scripts

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-training-dashboard*
*Context gathered: 2026-05-01*
