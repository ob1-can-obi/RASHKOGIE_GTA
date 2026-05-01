# Phase 3: Deferred Items

## Out-of-Scope Discoveries

### 1. test_training_correctness.py dimension mismatch (discovered in 03-01)

**File:** `tests/test_training_correctness.py`
**Issue:** Multiple tests call `metacontroller()` directly with `fused_dim=64`, producing 173-dim feature vectors instead of the production 237-dim. The new `META_INPUT_DIM` assertion correctly catches this mismatch.
**Impact:** Tests fail with AssertionError on the dimension check. These are Phase 1 validation tests (TRAIN-01 through TRAIN-05) that were written for the old dynamic-dimension behavior.
**Resolution:** Update test_training_correctness.py to use `fused_dim=128` (matching production) or pass a pre-created MetaMLP with matching `input_dim`. Not in scope for plan 03-01 which is restricted to modifying only `metacontroller/metacontroller.py` and `tests/conftest.py`.
