# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-30)

**Core value:** The metacontroller must learn to search intelligently — using the tree to find better actions than the planner's top-1, while staying ready before the current token ends.
**Current focus:** Phase 1 — Training Correctness

## Current Position

Phase: 1 of 5 (Training Correctness) -- COMPLETE
Plan: 3 of 3 in current phase
Status: Phase Complete
Last activity: 2026-05-01 — Completed 01-03 (Entropy & Advantage Fixes)

Progress: [████░░░░░░] 21%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 185s
- Total execution time: ~9 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | 556s | 185s |

**Recent Trend:**
- Last 5 plans: 01-01 (165s), 01-02 (203s), 01-03 (188s)
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
Stopped at: Completed 01-03-PLAN.md (Entropy & Advantage Fixes) -- Phase 1 Complete
Resume file: .planning/phases/01-training-correctness/01-03-SUMMARY.md
