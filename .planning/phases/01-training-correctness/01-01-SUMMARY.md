---
phase: 01-training-correctness
plan: 01
subsystem: metacontroller
tags: [training, sampling, categorical, bugfix]
dependency_graph:
  requires: []
  provides: [training-flag, categorical-sampling, test-infrastructure]
  affects: [metacontroller, search-tree]
tech_stack:
  added: [torch.distributions.Categorical]
  patterns: [conditional-sampling, flag-threading]
key_files:
  created:
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_training_correctness.py
  modified:
    - metacontroller/metacontroller.py
    - metacontroller/search_tree.py
decisions:
  - "training=False default preserves backward compatibility -- no caller changes needed"
  - "Categorical(logits=) used instead of softmax+Categorical(probs=) for numerical stability"
metrics:
  duration: 165s
  completed: 2026-05-01T01:41:56Z
  tasks: 2/2
  files_created: 3
  files_modified: 2
---

# Phase 1 Plan 1: Categorical Sampling Fix Summary

Replaced metacontroller's hardcoded argmax with Categorical sampling during training, threaded the training flag through SearchState/search_init/search_step, and created test infrastructure with TRAIN-01 validation tests.

## What Was Done

### Task 1: Add training mode flag and Categorical sampling (755c16e)

**metacontroller/metacontroller.py:**
- Added `from torch.distributions import Categorical` import
- Added `training=False` parameter to `metacontroller()` function signature
- Replaced line 166 argmax with conditional: `Categorical(logits=).sample()` in training mode, `argmax(dim=-1)` in inference mode

**metacontroller/search_tree.py:**
- Added `training=False` parameter to `SearchState.__init__()` and stored as `self.training`
- Added `training=False` parameter to `search_init()` and passed through to `SearchState` constructor
- Added `training=state.training` to the `metacontroller()` call in `search_step()`

### Task 2: Create test infrastructure and TRAIN-01 validation tests (2a2d10c)

**tests/conftest.py:**
- `mock_meta_mlp`: Small MLP matching metacontroller's lazy-init pattern (Linear->ReLU->Linear(4))
- `mock_meta_trajectory`: 3-step synthetic trajectory matching search_tree format
- `mock_rollout`: 5-frame rollout matching executor format
- `mock_rollout_long`: 20-frame rollout for duration normalization tests

**tests/test_training_correctness.py:**
- `test_categorical_sampling`: Validates training=True produces varied decisions (>=2 unique over 500 samples), inference mode is deterministic (1 unique over 10 samples), all decisions are valid (0-3)
- `test_training_flag_default_is_false`: Validates default behavior (no training kwarg) is deterministic, matching original pre-fix behavior

## Test Results

```
tests/test_training_correctness.py::test_categorical_sampling PASSED
tests/test_training_correctness.py::test_training_flag_default_is_false PASSED
2 passed in 0.06s
```

Training mode produced all 4 decisions {0, 1, 2, 3} (EXPLORE, INTERRUPT, COMMIT_NEXT, ROLLBACK) over 200 samples in the verification script.

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Task | Commit  | Message                                                          |
|------|---------|------------------------------------------------------------------|
| 1    | 755c16e | feat(01-01): add Categorical sampling in training mode, argmax in inference |
| 2    | 2a2d10c | test(01-01): add test infrastructure and TRAIN-01 validation tests |

## Self-Check: PASSED

All 5 files verified present. Both commit hashes (755c16e, 2a2d10c) confirmed in git log.
