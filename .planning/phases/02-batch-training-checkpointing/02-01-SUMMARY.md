---
phase: 02-batch-training-checkpointing
plan: 01
subsystem: training
tags: [pytorch, adam, gradient-clipping, replay-buffer, deque, reinforce]

# Dependency graph
requires:
  - phase: 01-training-correctness
    provides: "compute_token_return, compute_metalevel_advantages, update_metapolicy, get_entropy_coeff"
provides:
  - "TrainingState class with buffer, Adam optimizers, batch update, gradient clipping"
  - "add_trajectory / get_batch / update_metapolicy_batch / update_reward_batch methods"
  - "DEFAULT_BATCH_SIZE, DEFAULT_BUFFER_CAPACITY, DEFAULT_LR, DEFAULT_EPS, DEFAULT_MAX_GRAD_NORM constants"
  - "Test fixtures: mock_reward_mlp, mock_rf_predictor, mock_training_state, _make_trajectory_dict"
affects: [02-02, 02-03, frame_loop integration]

# Tech tracking
tech-stack:
  added: [collections.deque, torch.optim.Adam, torch.nn.utils.clip_grad_norm_]
  patterns: [TrainingState class owning optimizers and buffer, batch-averaged REINFORCE, cross-batch advantage normalization]

key-files:
  created:
    - tests/test_batch_training.py
  modified:
    - metacontroller/trainer.py
    - tests/conftest.py

key-decisions:
  - "trajectories_since_update counter tracks batch trigger separately from buffer length (avoids off-by-one per Pitfall 2)"
  - "Cross-batch advantage normalization pools all steps from all trajectories before normalizing"
  - "reward_mlp and rf_predictor share a single Adam optimizer matching existing manual SGD grouping"
  - "Batch-mean divides by total steps (metapolicy) or batch count (reward) for consistent gradient scale"

patterns-established:
  - "TrainingState pattern: stateful class owning buffer + optimizers, created once at session start"
  - "Batch update pattern: zero_grad -> loop accumulate -> backward -> clip_grad_norm_ -> step"
  - "Gradient clip event reporting: return dict includes grad_norm and clipped boolean"

requirements-completed: [BATCH-01, BATCH-02, BATCH-03, BATCH-04]

# Metrics
duration: 5min
completed: 2026-05-01
---

# Phase 2 Plan 01: Batch Training Infrastructure Summary

**TrainingState class with deque replay buffer, Adam optimizers replacing manual SGD, batch-averaged REINFORCE with cross-batch advantage normalization, and gradient clipping at max_norm=0.5**

## Performance

- **Duration:** 5 min (285s)
- **Started:** 2026-05-01T02:43:10Z
- **Completed:** 2026-05-01T02:48:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- TrainingState class in trainer.py with buffer (deque maxlen=10000), Adam optimizers (meta + reward groups), batch update logic, and gradient clipping
- 11 unit tests covering BATCH-01 through BATCH-04 requirements all passing
- Full test suite (31 tests: 20 Phase 1 + 11 Phase 2) green with no regressions
- Existing functions (update_metapolicy, train_reward_head) preserved unchanged for backward compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: Add TrainingState class** - `b163f08` (feat)
2. **Task 2: Add test fixtures and unit tests** - `08e1d68` (test)

## Files Created/Modified
- `metacontroller/trainer.py` - Added TrainingState class with buffer, Adam optimizers, update_metapolicy_batch, update_reward_batch; added imports (deque, Adam, clip_grad_norm_); added constants (DEFAULT_BATCH_SIZE=8, DEFAULT_LR=3e-4, etc.)
- `tests/conftest.py` - Added mock_reward_mlp, mock_rf_predictor, mock_training_state fixtures and _make_trajectory_dict helper
- `tests/test_batch_training.py` - 11 unit tests: buffer accumulation/eviction, batch trigger, Adam weight updates, Adam state persistence, gradient clipping with clip event reporting, constant verification

## Decisions Made
- trajectories_since_update counter tracks batch trigger separately from buffer length (avoids off-by-one per Pitfall 2 from RESEARCH.md)
- Cross-batch advantage normalization pools all steps from all trajectories before normalizing (resolves Open Question 3 from RESEARCH.md)
- reward_mlp and rf_predictor share a single Adam optimizer matching existing manual SGD grouping (Pitfall 5 from RESEARCH.md)
- Batch-mean divides by total_steps for metapolicy (consistent per-step gradient scale) and by len(batch) for reward head (matching per-trajectory loss structure)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed conftest import path in test file**
- **Found during:** Task 2 (test file creation)
- **Issue:** Plan specified `from conftest import _make_trajectory_dict` but pytest's conftest.py is not on the regular Python import path
- **Fix:** Added TESTS_DIR to sys.path in test_batch_training.py, following the same pattern used for METACONTROLLER_DIR
- **Files modified:** tests/test_batch_training.py
- **Verification:** All 11 tests pass after fix
- **Committed in:** 08e1d68 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for test imports to work. No scope creep.

## Issues Encountered
None beyond the import path deviation documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- TrainingState class ready for integration with frame_loop.py (Plan 02-02)
- Checkpoint save/load methods to be added in Plan 02-02
- Adam optimizer state persists across batches, ready for checkpoint serialization

## Self-Check: PASSED

- All files exist (metacontroller/trainer.py, tests/test_batch_training.py, tests/conftest.py, 02-01-SUMMARY.md)
- All commits verified (b163f08, 08e1d68)
- TrainingState class present (1 occurrence)
- 11 test functions present in test_batch_training.py
- Full test suite: 31 passed, 0 failed

---
*Phase: 02-batch-training-checkpointing*
*Completed: 2026-05-01*
