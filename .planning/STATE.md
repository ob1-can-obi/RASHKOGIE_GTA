# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-30)

**Core value:** The metacontroller must learn to search intelligently — using the tree to find better actions than the planner's top-1, while staying ready before the current token ends.
**Current focus:** Phase 2 — Batch Training and Checkpointing

## Current Position

Phase: 2 of 5 (Batch Training and Checkpointing)
Plan: 3 of 3 in current phase
Status: Phase Complete
Last activity: 2026-05-01 — Plan 02-03 complete (Integration wiring: train_step + drive_token)

Progress: [████████░░] 40%

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 215s
- Total execution time: ~22 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 556s | 185s |
| 02 | 3 | 735s | 245s |

**Recent Trend:**
- Last 5 plans: 01-02 (203s), 01-03 (188s), 02-01 (285s), 02-02 (224s), 02-03 (226s)
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

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 4 research flag: Encoder pretraining objective is not yet defined (reconstruction vs. contrastive vs. joint) — must be resolved before Phase 4 planning
- Phase 4 research flag: Convergence thresholds (MSE < 0.05, MSE < 0.1, accuracy > 60%) are heuristics that may need empirical calibration after first training sessions

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

Last session: 2026-05-01
Stopped at: Completed 02-03-PLAN.md (Phase 2 complete)
Resume file: None
