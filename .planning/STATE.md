---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 5 context gathered
last_updated: "2026-05-01T20:27:08.177Z"
last_activity: 2026-05-01 -- Phase 05 wave 1 complete (3/6 plans)
progress:
  total_phases: 5
  completed_phases: 4
  total_plans: 21
  completed_plans: 18
  percent: 86
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-30)

**Core value:** The metacontroller must learn to search intelligently — using the tree to find better actions than the planner's top-1, while staying ready before the current token ends.
**Current focus:** Phase 5 (Training Dashboard) — Wave 1 complete, executing Wave 2

## Current Position

Phase: 5 of 5 (Training Dashboard) — Executing
Plan: 3 of 6 in current phase (Wave 1 complete)
Status: Executing Wave 2
Last activity: 2026-05-01 -- Phase 05 wave 1 complete

Progress: [█████████████████░░░] 86%

## Performance Metrics

**Velocity:**

- Total plans completed: 15
- Average duration: 214s
- Total execution time: ~54 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 556s | 185s |
| 02 | 3 | 735s | 245s |
| 03 | 3 | 579s | 193s |
| 04 | 6 | 1343s | 224s |

**Recent Trend:**

- Last 5 plans: 04-02 (242s), 04-03 (210s), 04-04 (220s), 04-06 (225s), 04-05 (219s)
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- 01-01: training=False default preserves backward compatibility -- no caller changes needed
- 01-01: Categorical(logits=) used instead of softmax+Categorical(probs=) for numerical stability
- 01-02: NOT_READY_C=0.1, LAZY_K=0.05, MIN_SEARCH_NODES=2 -- moderate penalty constants that guide without dominating
- 01-02: Duration normalization applied BEFORE penalty injection to avoid double-scaling (Pitfall 5)
- 01-02: Not-ready penalty REPLACES final reward; lazy penalty ADDS to it -- different severity levels
- 01-03: DEFAULT_ENTROPY_ANNEAL_STEPS=5000 chosen as moderate schedule for linear entropy decay
- 01-03: entropy_loss = -entropy_coeff * entropy_sum: negative sign maximizes entropy (prevents collapse)
- 01-03: Advantage normalization skipped for single-step trajectories (std undefined for n=1)
- 01-03: Epsilon 1e-8 guards division in advantage normalization for all-same advantages
- 02-01: trajectories_since_update counter tracks batch trigger separately from buffer length (avoids off-by-one)
- 02-01: Cross-batch advantage normalization pools all steps from all trajectories before normalizing
- 02-01: reward_mlp and rf_predictor share a single Adam optimizer matching existing manual SGD grouping
- 02-01: Batch-mean divides by total_steps (metapolicy) or batch count (reward) for consistent gradient scale
- 02-02: reward_mlp and rf_predictor get SEPARATE .pt files; shared optimizer saves as optimizer_reward.pt
- 02-02: Buffer NOT restored on load -- starts empty on resume per RESEARCH.md resolved decision
- 02-02: All torch.load calls use weights_only=True and map_location='cpu' (Pitfall 4 and 6)
- 02-03: training_state=None default preserves 100% backward compatibility for all callers
- 02-03: Batch update in drive_token triggers BOTH meta and reward updates together, then saves checkpoint
- 02-03: In batch mode, drive_token skips legacy train_reward_head (reward update handled in batch)
- 02-03: drive_token return dict includes batch_ready, meta_batch_result, reward_batch_result
- Roadmap: Three bugs (argmax, no entropy, no not-ready penalty) must all be fixed in Phase 1 before any training is valid
- Roadmap: Batch infrastructure (Phase 2) is a prerequisite for stable training — online single-sample REINFORCE has unbounded variance
- Roadmap: Architecture upgrades (Phase 3) must precede the training pipeline (Phase 4) so the upgraded modules are what gets trained
- 03-01: MetaMLP uses nn.Module subclass (not nn.Sequential) because skip connections require additive composition from two computation graph branches
- 03-01: META_INPUT_DIM=237 hardcoded with breakdown comment rather than computed from imported constants (TOP_K/TOKEN_EMBED_DIM not module-level)
- 03-01: hidden_dim parameter removed from metacontroller() since MetaMLP has fixed internal layer sizes
- 03-02: Block 2 query_dim=64 from ctx1 output (not 192 from ego/scene/route concat)
- 03-02: Residual connection only on block 2 (block 1 has dim mismatch: 192 query vs 64 output)
- 03-02: Action planner uses hidden_dim*2=256 for first hidden layer, matching wider-then-narrow pattern
- 04-01: ConvergenceDetector uses mode param ("min"/"max") for MSE vs accuracy metrics
- 04-01: freeze_module uses hasattr(value, 'parameters') to skip int/float in encoder weight dicts
- 04-01: update_training_status falls back to initial template if JSON is malformed (T-04-16)
- 04-01: load_training_config wraps JSONDecodeError with descriptive message (T-04-01)
- 04-02: importlib used in tests for train.py import since main_model/ is not a Python package
- 04-02: Batch-mean loss divides by len(batch) for consistent gradient scale
- 04-02: Max epochs reached without convergence still saves checkpoint and updates status
- 04-03: Encoder checkpoint loading supports both session directories and direct .pt files
- 04-03: Batch-mean loss divides by len(batch) for consistent gradient scale (same as main_model/train.py)
- 04-03: Max epochs reached without convergence still saves checkpoint and updates status
- 04-04: importlib used in tests for train.py import since action_planner/ is not a Python package
- 04-04: Batch-mean loss divides by len(batch) for consistent gradient scale (same as other train.py scripts)
- 04-04: Max epochs reached without convergence still saves checkpoint and updates status
- 04-04: Intuition checkpoint loading supports both session directories and direct .pt files
- 04-04: token_id=0 (idle) used as prev_token for intuition head during action planner training
- 04-06: compute_reward returns (reward, components) tuple -- unpack correctly in capture_states.py
- 04-06: token_start_state tracked separately from prev_state for accurate reward head state_before
- 04-06: Session timestamps generated via datetime.now().strftime() -- never from user input (T-04-18)
- 04-05: STAGE_ORDER = [encoder_intuition, reward_head, action_planner, metacontroller] matching D-03
- 04-05: Freeze choices limited to encoder_intuition and reward_head (action_planner and metacontroller do not freeze)
- 04-05: Stale detection checks both checkpoint mtime and started_at timestamp when no checkpoint exists
- 04-05: _get_frozen_stage_names maps individual module names back to stage names for dependency checking

### Pending Todos

None yet.

### Blockers/Concerns

- RESOLVED: Encoder pretraining objective decided — next-state prediction, joint with intuition head (D-01, D-02 in 04-CONTEXT.md)
- RESOLVED: Convergence approach decided — dual criteria (threshold + patience), configurable via training_config.json (D-11, D-13 in 04-CONTEXT.md)
- Convergence threshold VALUES (MSE < 0.05, etc.) are still heuristics — will need empirical calibration after first training sessions

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | VOC-01: Value of Cognition estimate per frame | Deferred | Init |
| v2 | VOC-02: Search efficiency metric | Deferred | Init |
| v2 | VOC-03: Counterfactual analysis | Deferred | Init |
| v2 | ADV-01: Actor-critic value head | Deferred | Init |
| v2 | ADV-02: Curriculum learning | Deferred | Init |
| v2 | ADV-03: Multi-route evaluation benchmark | Deferred | Init |

## Session Continuity

Last session: 2026-05-01T18:00:00Z
Stopped at: Phase 5 context gathered
Resume file: .planning/phases/05-training-dashboard/05-CONTEXT.md
