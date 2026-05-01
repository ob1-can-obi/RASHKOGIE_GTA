---
phase: 05-training-dashboard
plan: 06
subsystem: training-scripts
tags: [jsonl, websocket, hot-reload, metrics, embeddings, decision-counts]
dependency_graph:
  requires: []
  provides: [jsonl-metric-rows, jsonl-embedding-snapshots, jsonl-decision-counts, ws-hot-reload-client]
  affects: [main_model/train.py, metacontroller/trainer.py]
tech_stack:
  added: [websockets.sync.client]
  patterns: [daemon-thread-ws-client, jsonl-line-writer, driving-context-classification]
key_files:
  created: []
  modified:
    - main_model/train.py
    - metacontroller/trainer.py
decisions:
  - "lr validated > 0 before applying hot-reload (T-05-20 mitigation)"
  - "DashboardParamReceiver as standalone class in train.py vs inline in TrainingState for trainer.py -- different patterns appropriate to each file's structure"
  - "Driving context classification thresholds (steering > 0.3 = turn, brake > 0.3 = braking) chosen as reasonable defaults"
metrics:
  duration: 284s
  completed: "2026-05-01T20:50:33Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 5 Plan 6: Training Script JSONL + WebSocket Modifications Summary

JSONL metric, embedding, and decision count writes added to both training scripts, with background WebSocket clients for dashboard hot-reload of hyperparameters.

## What Was Done

### Task 1: main_model/train.py JSONL + WS Client (e9a7cc5)

Added three additive capabilities to the encoder/intuition training script:

1. **JSONL metric writer** -- `write_jsonl()` function emits `type:metric` rows every 10 steps with step, loss, grad_norm, clipped, lr, epoch, timestamp. File opened at session start in data_dir, flushed after every write.

2. **Embedding snapshot writer** -- Emits `type:embedding` rows every 500 steps containing the 128-float z_t vector and a driving context classification (straight/turn/braking). Added `_derive_driving_context()` helper function that classifies based on steering and brake input thresholds.

3. **DashboardParamReceiver class** -- Background daemon thread WebSocket client that connects to `/ws/train`, registers as `encoder_intuition`, and listens for `param_update` messages. Applies lr directly to optimizer (validated > 0), queues batch_size/entropy_coeff/think_cost for training loop to read. Retries every 10s if dashboard unavailable. Cleanup at both convergence and max-epochs return paths.

### Task 2: metacontroller/trainer.py JSONL + WS Client (bd772c9)

Added four additive capabilities to the TrainingState class:

1. **JSONL writer** -- New `jsonl_dir` parameter on `__init__` (defaults to None for backward compatibility). `_write_jsonl()` method writes compact JSON lines with flush.

2. **Decision counts emission** -- After each `update_metapolicy_batch`, aggregates decision distributions across the batch (explore/rollback/interrupt/commit_next counts using decision mapping 0-3) and emits `type:decision_counts` JSONL rows with nodes_expanded and search_depth.

3. **Per-batch metric emission** -- Emits `type:metric` JSONL rows with loss, reward (mean episode return via `compute_token_return`), grad_norm, clipped, lr.

4. **WS param receiver** -- `start_param_receiver()` method starts a daemon thread WS client that connects to `/ws/train`, registers as `metacontroller`, and applies param updates. `get_pending_params()` retrieves and clears queued params. `close()` method for cleanup. Pending batch_size applied at start of `update_metapolicy_batch`.

## Commits

| Task | Commit | Files | Description |
|------|--------|-------|-------------|
| 1 | e9a7cc5 | main_model/train.py | JSONL metric/embedding writes + WS client |
| 2 | bd772c9 | metacontroller/trainer.py | JSONL decision_counts/metric writes + WS client |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Security] Added lr > 0 validation in DashboardParamReceiver._apply_params**
- **Found during:** Task 1
- **Issue:** T-05-20 threat model requires validation of param values before applying. The plan pseudocode did not include the `> 0` guard for lr.
- **Fix:** Added `if new_lr > 0:` guard before applying lr to optimizer param groups. Same guard applied in Task 2's WS receiver.
- **Files modified:** main_model/train.py, metacontroller/trainer.py
- **Commits:** e9a7cc5, bd772c9

## Decisions Made

1. **lr > 0 validation**: Added per T-05-20 threat model mitigation -- prevents setting lr to zero or negative which would break training.
2. **Separate class vs inline pattern**: main_model/train.py uses a standalone `DashboardParamReceiver` class (cleaner for a function-based training script), while metacontroller/trainer.py uses inline methods on `TrainingState` (natural fit since TrainingState already owns the optimizer).
3. **Driving context thresholds**: steering > 0.3 for "turn", brake > 0.3 for "braking" -- reasonable defaults that can be tuned later.

## Known Stubs

None -- all data sources are wired to real training loop variables (z_t, total_loss, grad_norm, etc.).

## Self-Check: PASSED

- All modified files exist on disk
- All task commits verified in git log (e9a7cc5, bd772c9)
- SUMMARY.md created at expected path
