# Requirements: RASHKOGIE GTA

**Defined:** 2026-04-30
**Core Value:** The metacontroller must learn to search intelligently — using the tree to find better actions than the planner's top-1, while staying ready before the current token ends.

## v1 Requirements

### Training Correctness

- [x] **TRAIN-01**: Metacontroller uses categorical sampling from decision logits during training (argmax only at inference)
- [x] **TRAIN-02**: Entropy regularization added to metacontroller loss with annealing schedule (0.05 → 0.005)
- [x] **TRAIN-03**: Large negative reward when metacontroller fails to commit before current token ends
- [x] **TRAIN-04**: Penalty for lazy commits (immediate COMMIT_NEXT without meaningful search depth)
- [x] **TRAIN-05**: Advantage normalization across metalevel trajectory batches
- [x] **TRAIN-06**: Duration-normalized returns to correct BPE variable-length token bias

### Batch Training Infrastructure

- [x] **BATCH-01**: Trajectory replay buffer (deque, ~10K capacity) collecting full metalevel trajectories
- [x] **BATCH-02**: Batch updates every N=8 trajectories instead of single-sample online updates
- [x] **BATCH-03**: Adam optimizer (lr=3e-4, eps=1e-5) replacing manual SGD for all modules
- [x] **BATCH-04**: Gradient clipping (max_norm=0.5) on all policy gradient updates
- [x] **BATCH-05**: Per-module checkpoint saving (one .pt per module per session)
- [x] **BATCH-06**: Checkpoint loading and resume for interrupted training sessions

### Architecture Upgrades

- [x] **ARCH-01**: Metacontroller MLP upgraded to 3 layers (256-256-128-4) with skip connection and LayerNorm
- [x] **ARCH-02**: Encoder attention upgraded to 2 blocks with LayerNorm (keep 4 heads, head_dim=16)
- [x] **ARCH-03**: Action planner upgraded to 2-layer MLP
- [x] **ARCH-04**: Input dimension pinned as a constant (not recomputed dynamically on every call)

### Module Training Pipelines

- [x] **PIPE-01**: Central training_data folder structure for all modules
- [x] **PIPE-02**: Intuition head standalone training loop (automated during gameplay, MSE on z_next vs real z_{t+1})
- [x] **PIPE-03**: Reward head standalone training loop (automated during gameplay, MSE on r_edge vs realized return)
- [ ] **PIPE-04**: Action planner training loop (imitation learning from player driving captures)
- [x] **PIPE-05**: Module freeze mechanism — freeze intuition head + reward head when converged, block gradients
- [x] **PIPE-06**: Convergence detection with configurable thresholds per module
- [ ] **PIPE-07**: Full end-to-end training chain: encoder → intuition → reward → freeze → action planner → metacontroller RL

### Training Dashboard

- [ ] **DASH-01**: FastAPI web server serving training dashboard on localhost
- [ ] **DASH-02**: Live loss curves and reward curves (SSE streaming + Chart.js)
- [ ] **DASH-03**: Decision distribution histogram (EXPLORE/ROLLBACK/INTERRUPT/COMMIT_NEXT ratios over time)
- [ ] **DASH-04**: Hyperparameter control panel (tune lr, entropy coeff, think_cost, batch size from browser)
- [ ] **DASH-05**: Training session history with per-session metrics summary
- [ ] **DASH-06**: Session comparison view (overlay loss curves from different runs)
- [ ] **DASH-07**: Episode return tracking (primary health metric, not loss)
- [ ] **DASH-08**: Nodes expanded per token and search depth distribution

## v2 Requirements

### Advanced Research Metrics

- **VOC-01**: Value of Cognition estimation per search step (meaningful only after stable metacontroller)
- **VOC-02**: Search efficiency metric (quality improvement per node expanded)
- **VOC-03**: Counterfactual analysis (would planner top-1 have been better than searched result?)

### Advanced Training

- **ADV-01**: Actor-critic value head for metacontroller (reduces REINFORCE variance)
- **ADV-02**: Curriculum learning (easy routes → hard routes)
- **ADV-03**: Multi-route evaluation benchmark

## Out of Scope

| Feature | Reason |
|---------|--------|
| TensorBoard / W&B integration | Custom dashboard chosen for full control over hyperparameter UI |
| Prioritized experience replay | Overkill for trajectory-level buffer; adds complexity without clear benefit |
| Cloud training | Same Windows PC runs GTA + training |
| Image/pixel perception | State comes from SHVDN game data, not screenshots |
| Multi-agent scenarios | Single vehicle research project |
| Pretrained model integration | All modules trained from scratch on GTA data |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRAIN-01 | Phase 1 | Complete |
| TRAIN-02 | Phase 1 | Complete |
| TRAIN-03 | Phase 1 | Complete |
| TRAIN-04 | Phase 1 | Complete |
| TRAIN-05 | Phase 1 | Complete |
| TRAIN-06 | Phase 1 | Complete |
| BATCH-01 | Phase 2 | Complete |
| BATCH-02 | Phase 2 | Complete |
| BATCH-03 | Phase 2 | Complete |
| BATCH-04 | Phase 2 | Complete |
| BATCH-05 | Phase 2 | Complete |
| BATCH-06 | Phase 2 | Complete |
| ARCH-01 | Phase 3 | Complete |
| ARCH-02 | Phase 3 | Complete |
| ARCH-03 | Phase 3 | Complete |
| ARCH-04 | Phase 3 | Complete |
| PIPE-01 | Phase 4 | Complete |
| PIPE-02 | Phase 4 | Complete |
| PIPE-03 | Phase 4 | Complete |
| PIPE-04 | Phase 4 | Pending |
| PIPE-05 | Phase 4 | Complete |
| PIPE-06 | Phase 4 | Complete |
| PIPE-07 | Phase 4 | Pending |
| DASH-01 | Phase 5 | Pending |
| DASH-02 | Phase 5 | Pending |
| DASH-03 | Phase 5 | Pending |
| DASH-04 | Phase 5 | Pending |
| DASH-05 | Phase 5 | Pending |
| DASH-06 | Phase 5 | Pending |
| DASH-07 | Phase 5 | Pending |
| DASH-08 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-30*
*Last updated: 2026-04-30 after initial definition*
