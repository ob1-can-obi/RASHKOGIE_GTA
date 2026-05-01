---
phase: 01-training-correctness
plan: 02
subsystem: training
tags: [reward-shaping, duration-normalization, penalty-injection, metacontroller, reinforcement-learning]

# Dependency graph
requires:
  - phase: 01-training-correctness/01
    provides: "Categorical sampling in metacontroller (TRAIN-01)"
provides:
  - "Duration-normalized token returns via sqrt(k) in compute_token_return"
  - "Not-ready penalty (TRAIN-03) for fallback commits in compute_metalevel_advantages"
  - "Lazy-commit penalty (TRAIN-04) for insufficient search in compute_metalevel_advantages"
  - "Fallback detection (is_fallback flag) wired through frame_loop.py to trainer.py"
  - "10 new validation tests for TRAIN-03, TRAIN-04, TRAIN-06"
affects: [01-training-correctness/03, 02-batch-infrastructure]

# Tech tracking
tech-stack:
  added: [math.sqrt]
  patterns: [penalty-injection-after-normalization, duration-proportional-penalties, fallback-flag-propagation]

key-files:
  created: []
  modified:
    - metacontroller/trainer.py
    - metacontroller/frame_loop.py
    - tests/test_training_correctness.py

key-decisions:
  - "NOT_READY_C=0.1 and LAZY_K=0.05 chosen as penalty multipliers -- moderate enough to guide without dominating the reward signal"
  - "MIN_SEARCH_NODES=2 threshold for lazy-commit detection -- low bar ensures only truly zero-search commits are penalized"
  - "Duration normalization applied BEFORE penalty injection in the call chain (Pitfall 5) to avoid double-scaling"
  - "Not-ready penalty REPLACES final reward (meta_rewards[-1] = penalty) while lazy penalty ADDS to it (meta_rewards[-1] += penalty)"

patterns-established:
  - "Penalty injection pattern: penalties applied in compute_metalevel_advantages after normalization in compute_token_return"
  - "Fallback flag pattern: is_fallback boolean propagated from frame_loop drive_token through train_step to credit assignment"
  - "Duration-proportional penalty scaling: penalties scale with token_duration_frames so short tokens get near-zero penalties"

requirements-completed: [TRAIN-03, TRAIN-04, TRAIN-06]

# Metrics
duration: 3min
completed: 2026-05-01
---

# Phase 1 Plan 02: Reward Signal Fixes Summary

**Duration-normalized returns via sqrt(k), not-ready penalty for fallback commits, and lazy-commit penalty for zero-search commits with 10 validation tests**

## Performance

- **Duration:** 3 min (203s)
- **Started:** 2026-05-01T01:44:47Z
- **Completed:** 2026-05-01T01:48:10Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- TRAIN-06: compute_token_return now divides by sqrt(num_frames) to normalize variable-length BPE token returns
- TRAIN-03: Fallback commits (metacontroller failed to decide before deadline) get -NOT_READY_C * token_duration_frames penalty replacing the realized return
- TRAIN-04: Lazy commits (nodes_expanded < MIN_SEARCH_NODES) get duration-proportional penalty added to realized return
- Fallback detection wired from frame_loop.py drive_token through trainer.py train_step with is_fallback flag
- 10 new tests covering normalization edge cases, penalty scaling, threshold boundaries, and penalty priority

## Task Commits

Each task was committed atomically:

1. **Task 1: Add duration normalization, penalty injection, and fallback detection** - `563a629` (feat)
2. **Task 2: Add TRAIN-03, TRAIN-04, TRAIN-06 validation tests** - `2d15d04` (test)

## Files Created/Modified
- `metacontroller/trainer.py` - Added import math, NOT_READY_C/LAZY_K/MIN_SEARCH_NODES constants, sqrt(k) normalization in compute_token_return, penalty injection in compute_metalevel_advantages, new params in train_step, is_fallback in return dict
- `metacontroller/frame_loop.py` - Added fallback detection (is_fallback flag), passed is_fallback/nodes_expanded/token_duration_frames to train_step, added is_fallback to drive_token return dict
- `tests/test_training_correctness.py` - Added 10 new test functions: 4 for TRAIN-06, 2 for TRAIN-03, 3 for TRAIN-04, 1 for penalty priority

## Decisions Made
- NOT_READY_C=0.1: Moderate penalty multiplier -- 10-frame token gets -1.0 penalty, large enough to discourage timeouts but not catastrophic
- LAZY_K=0.05: Half the not-ready multiplier -- lazy commits are bad but less bad than complete failure to commit
- MIN_SEARCH_NODES=2: Very low threshold -- only truly zero-or-one-node searches are penalized, any real search effort passes
- Not-ready penalty replaces final reward entirely (the commit was forced, no credit for outcome), while lazy penalty adds to it (the agent did commit, just without sufficient search)
- Duration normalization in compute_token_return happens before penalty injection in compute_metalevel_advantages, preventing double-scaling (Pitfall 5)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TRAIN-01, TRAIN-03, TRAIN-04, TRAIN-06 all have passing tests
- Remaining for Phase 1: TRAIN-02 (entropy bonus) and TRAIN-05 (advantage normalization) in Plan 01-03
- All reward signal infrastructure is in place for the entropy/advantage plan to build on

## Self-Check: PASSED

- All 3 modified files exist on disk
- Commit 563a629 (Task 1) verified in git log
- Commit 2d15d04 (Task 2) verified in git log
- All 12 tests pass (pytest exit 0)

---
*Phase: 01-training-correctness*
*Completed: 2026-05-01*
