# Roadmap: RASHKOGIE GTA

## Overview

This is a brownfield project — all module code exists. The remaining work is: fixing three compounding training bugs that currently produce invalid gradients, adding batch training infrastructure to reduce REINFORCE variance, upgrading module architectures to support the 237-dim metacontroller input, implementing the strict staged training pipeline (encoder → intuition → reward → action planner → metacontroller), and building a FastAPI dashboard for live training visibility. Phases are ordered by hard dependency: nothing converges until the training loop is correct, nothing trains stably without batch infrastructure, and the dashboard only shows useful data after the pipeline is wired.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Training Correctness** - Fix the three compounding bugs that corrupt every gradient update
- [x] **Phase 2: Batch Training and Checkpointing** - Add trajectory buffer, Adam optimizer, and per-module checkpoint system
- [x] **Phase 3: Architecture Upgrades** - Deepen MLPs, add encoder attention block, and add LayerNorm throughout
- [ ] **Phase 4: Module Training Pipelines** - Implement the strict staged training chain with convergence-triggered freezes
- [ ] **Phase 5: Training Dashboard** - Build the FastAPI live monitoring and hyperparameter control UI

## Phase Details

### Phase 1: Training Correctness
**Goal**: The metacontroller training loop produces valid exploration and a coherent reward signal
**Depends on**: Nothing (first phase)
**Requirements**: TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06
**Success Criteria** (what must be TRUE):
  1. Metacontroller samples all four decisions (EXPLORE/ROLLBACK/INTERRUPT/COMMIT_NEXT) during a training run — not a single repeated argmax
  2. Policy entropy does not collapse to near-zero within the first 100 gradient updates
  3. A token that expires without a commit produces a large negative reward in the trainer log
  4. A COMMIT_NEXT with zero nodes expanded produces a distinct negative penalty in the trainer log
  5. Advantage values across a batch are zero-mean and unit-variance before the policy gradient step
**Plans:** 3 plans

Plans:
- [x] 01-01-PLAN.md — Categorical sampling with training mode flag (TRAIN-01)
- [x] 01-02-PLAN.md — Duration normalization, not-ready penalty, lazy-commit penalty (TRAIN-03, TRAIN-04, TRAIN-06)
- [x] 01-03-PLAN.md — Entropy regularization and advantage normalization (TRAIN-02, TRAIN-05)

### Phase 2: Batch Training and Checkpointing
**Goal**: Training sessions accumulate trajectory batches, survive interruption, and resume without cold-starting the optimizer
**Depends on**: Phase 1
**Requirements**: BATCH-01, BATCH-02, BATCH-03, BATCH-04, BATCH-05, BATCH-06
**Success Criteria** (what must be TRUE):
  1. Trainer accumulates complete metalevel trajectories and only updates after every 8th trajectory flushes the buffer
  2. Adam optimizer state (momentum, variance) persists across buffer flushes — not re-initialized on each update
  3. Gradient norm is clipped and the clip event is logged when a large reward spike would otherwise cause a catastrophic update
  4. Stopping and restarting training resumes from the last saved checkpoint with no loss of optimizer state
  5. Each module saves a separate .pt checkpoint file per session that can be loaded independently
**Plans:** 3 plans

Plans:
- [x] 02-01-PLAN.md — TrainingState class with buffer, Adam optimizers, batch update, gradient clipping (BATCH-01, BATCH-02, BATCH-03, BATCH-04)
- [x] 02-02-PLAN.md — Per-module checkpoint save/load on TrainingState (BATCH-05, BATCH-06)
- [x] 02-03-PLAN.md — Integration into train_step and drive_token for end-to-end batch training flow (all BATCH-*)

### Phase 3: Architecture Upgrades
**Goal**: All modules have sufficient capacity for their inputs and all hidden representations are normalized
**Depends on**: Phase 2
**Requirements**: ARCH-01, ARCH-02, ARCH-03, ARCH-04
**Success Criteria** (what must be TRUE):
  1. MetaMLP has 3 hidden layers (256-256-128) with a skip connection from input to layer 2 and LayerNorm at every hidden layer
  2. Encoder processes input through 2 stacked attention blocks, each followed by LayerNorm
  3. Action planner uses a 2-layer MLP — not a single hidden layer
  4. All modules serialize to disk and reload cleanly via torch.save/load with no shape mismatches
**Plans:** 3 plans

Plans:
- [x] 03-01-PLAN.md — MetaMLP class with skip connection and LayerNorm, META_INPUT_DIM constant (ARCH-01, ARCH-04)
- [x] 03-02-PLAN.md — Encoder 2-block attention with LayerNorm, action planner 2-layer MLP (ARCH-02, ARCH-03)
- [x] 03-03-PLAN.md — Validation test suite for all architecture upgrades (ARCH-01, ARCH-02, ARCH-03, ARCH-04)

### Phase 4: Module Training Pipelines
**Goal**: The complete encoder → intuition → reward → action planner → metacontroller training chain runs in strict order with convergence-triggered freezes
**Depends on**: Phase 3
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, PIPE-06, PIPE-07
**Success Criteria** (what must be TRUE):
  1. Intuition head training loop runs automated during gameplay and logs MSE on z_next_pred vs real z_{t+1}
  2. Reward head training loop runs automated during gameplay and logs MSE on r_edge vs realized return
  3. Action planner trains via imitation learning from player driving captures and reaches top-1 accuracy above 60%
  4. Reaching the configured convergence threshold for intuition head and reward head automatically freezes those modules — gradients are blocked in all downstream consumers
  5. Metacontroller RL training only begins after intuition head and reward head are frozen — the reward target does not move during metacontroller policy updates
  6. A full end-to-end training run can be started, progressed through all module stages, and evaluated as a loop: capture data → train in order → run agent → evaluate
**Plans:** 6 plans

Plans:
- [ ] 04-01-PLAN.md — Shared infrastructure: training_config.json, training_status.json, training_utils.py, training_data dirs, test scaffold (PIPE-01, PIPE-05, PIPE-06)
- [ ] 04-02-PLAN.md — Encoder + intuition head joint training script (PIPE-02)
- [ ] 04-03-PLAN.md — Reward head offline training script with freeze on convergence (PIPE-03)
- [ ] 04-04-PLAN.md — Action planner imitation learning script (PIPE-04)
- [ ] 04-05-PLAN.md — Coordinator CLI and end-to-end pipeline integration (PIPE-07)
- [ ] 04-06-PLAN.md — Data capture scripts and training_data documentation (PIPE-01, PIPE-02, PIPE-03, PIPE-04)

### Phase 5: Training Dashboard
**Goal**: Live training metrics, decision distributions, and hyperparameter controls are accessible from a browser during any training session
**Depends on**: Phase 4
**Requirements**: DASH-01, DASH-02, DASH-03, DASH-04, DASH-05, DASH-06, DASH-07, DASH-08
**Success Criteria** (what must be TRUE):
  1. Navigating to localhost in a browser shows live-updating loss and reward curves for all modules without page reload
  2. The decision distribution histogram updates in real time and shows the ratio of EXPLORE/ROLLBACK/INTERRUPT/COMMIT_NEXT
  3. Changing entropy_coeff, think_cost, lr, or batch_size in the browser control panel takes effect in the running training session without restart
  4. Past training sessions are listed with per-session metric summaries and can be compared by overlaying their loss curves
  5. Episode return and nodes-expanded-per-token are tracked as primary health metrics distinct from raw loss values
**UI hint**: yes
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Training Correctness | 3/3 | Complete | 2026-05-01 |
| 2. Batch Training and Checkpointing | 3/3 | Complete | 2026-05-01 |
| 3. Architecture Upgrades | 3/3 | Complete | 2026-05-01 |
| 4. Module Training Pipelines | 0/6 | Not started | - |
| 5. Training Dashboard | 0/TBD | Not started | - |
