# Phase 3: Architecture Upgrades - Pattern Map

**Mapped:** 2026-04-30
**Files analyzed:** 6 (3 modified, 2 updated, 1 new)
**Analogs found:** 6 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `metacontroller/metacontroller.py` | model | request-response | self (current `nn.Sequential` lazy init) | exact |
| `main_model/main_model.py` | model | request-response | self (`create_encoder_weights` + `encode_state`) | exact |
| `action_planner/action_planner.py` | model | request-response | self (current `nn.Sequential` lazy init) | exact |
| `metacontroller/trainer.py` | service | CRUD | self (`TrainingState.save_checkpoint`/`load_checkpoint`) | exact |
| `tests/conftest.py` | config | test-fixture | self (`mock_meta_mlp` fixture) | exact |
| `tests/test_architecture.py` | test | unit | `tests/test_batch_training.py` | role-match |

## Pattern Assignments

### `metacontroller/metacontroller.py` (model, request-response) -- ARCH-01 + ARCH-04

**Analog:** self (current file)

**Imports pattern** (lines 36-38):
```python
import torch
from torch import nn
from torch.distributions import Categorical
```

**Module-level constants pattern** (lines 40-43):
```python
EXPLORE = 0       # expand next child and descend into it -- go deeper on this branch
INTERRUPT = 1     # stop current token now, switch immediately
COMMIT_NEXT = 2   # current token finishes, then switch to best found
ROLLBACK = 3      # this branch is done, go back up to parent and try a sibling
```
ARCH-04: Add `META_INPUT_DIM = 237` here, following the same module-level constant style.

**Lazy init pattern** (lines 151-156) -- the pattern being REPLACED:
```python
if meta_mlp is None:
    meta_mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 4),
    )
```
ARCH-01: Replace `nn.Sequential` with `MetaMLP()` class instantiation. The lazy init guard (`if meta_mlp is None`) stays. The new class goes above the function definition, after the constants block.

**Dynamic dim pattern** (line 145) -- the pattern being REPLACED:
```python
input_dim = features.shape[-1]
```
ARCH-04: Replace with `assert features.shape[-1] == META_INPUT_DIM`.

**Feature concatenation pattern** (lines 134-143) -- KEEP UNCHANGED:
```python
features = torch.cat(
    [drift, elapsed_ratio, token_frames_left,
     best_q, mean_q,
     urgency, parent_unexplored,
     current_path_value, best_path_value,
     best_current_q, mean_current_q,
     current_candidate_durations,
     current_candidate_emb],
    dim=-1,
)
```

**Forward pass + return pattern** (lines 162-189) -- KEEP UNCHANGED:
```python
decision_logits = meta_mlp(features)  # [batch, 4]

if training:
    dist = Categorical(logits=decision_logits)
    decision = dist.sample()
else:
    decision = decision_logits.argmax(dim=-1)

# ... gather best token ...

return {
    "decision":          decision,
    "decision_logits":   decision_logits,
    "features":          features,
    "selected_token_id": selected_token_id,
    "meta_mlp":          meta_mlp,
}
```
The `MetaMLP` class must be callable with the same `meta_mlp(features)` interface (standard `nn.Module.__call__` via `forward()`).

---

### `main_model/main_model.py` (model, request-response) -- ARCH-02

**Analog:** self (current file)

**Imports pattern** (lines 24-44):
```python
import sys
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parent.parent

for _d in ("intuition_head", "action_planner", "metacontroller"):
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from intuition_head import intuition_head
from action_planner import action_planner
from frame_loop import drive_token
from multi_head_attention import create_multi_head_attention_weights, multi_head_attention
```

**Encoder weight creation pattern** (lines 175-212):
```python
def create_encoder_weights(
    embed_dim  = EMBED_DIM,
    hidden_dim = HIDDEN_DIM,
    fused_dim  = FUSED_DIM,
    num_heads  = NUM_HEADS,
):
    ego_mlp = nn.Sequential(
        nn.Linear(EGO_DIM,   hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim),
    )
    # ... other sub-MLPs same pattern ...
    qw, kw, vw, ow = create_multi_head_attention_weights(embed_dim * 3, embed_dim)

    return {
        "ego_mlp":    ego_mlp,
        "scene_mlp":  scene_mlp,
        "route_mlp":  route_mlp,
        "entity_mlp": entity_mlp,
        "fusion_mlp": fusion_mlp,
        "qw": qw, "kw": kw, "vw": vw, "ow": ow,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
    }
```
ARCH-02: Rename `qw`->`qw1`, `kw`->`kw1`, `vw`->`vw1`, `ow`->`ow1`. Add second block via `create_multi_head_attention_weights(embed_dim, embed_dim)` (query_dim=64, not 192). Add `ln_attn1 = nn.LayerNorm(embed_dim)` and `ln_attn2 = nn.LayerNorm(embed_dim)` to the dict.

**Attention call pattern in encode_state** (lines 237-248):
```python
query_input = torch.cat([ego_emb, scene_emb, route_emb], dim=-1)
attn = multi_head_attention(
    query_input = query_input,
    entity_embs = entity_embs,
    mask        = t["mask"],
    qw          = weights["qw"],
    kw          = weights["kw"],
    vw          = weights["vw"],
    ow          = weights["ow"],
    num_heads   = weights["num_heads"],
)
entity_context = attn["entity_context"]
```
ARCH-02: Duplicate this block for block 1 (with `qw1`/`kw1`/`vw1`/`ow1`), apply `ln_attn1`, then block 2 (with `qw2`/`kw2`/`vw2`/`ow2`, query=`ctx1`, not `query_input`), apply residual + `ln_attn2`.

**Fusion pattern** (lines 250-252) -- KEEP UNCHANGED except `entity_context` -> `ctx2`:
```python
fusion_input = torch.cat([ego_emb, scene_emb, route_emb, entity_context], dim=-1)
z_t = weights["fusion_mlp"](fusion_input)
return z_t
```

---

### `main_model/multi_head_attention.py` (utility, request-response) -- NO CHANGES

**Analog:** self

The `multi_head_attention()` function (lines 38-118) and `create_multi_head_attention_weights()` (lines 15-35) need NO modifications. They already accept explicit Q/K/V/O weights and arbitrary `query_dim`. The second attention block simply calls them with different weights and `query_dim=64`.

---

### `action_planner/action_planner.py` (model, request-response) -- ARCH-03

**Analog:** self (current file)

**Lazy init pattern** (lines 68-73) -- the pattern being MODIFIED:
```python
if planner_mlp is None:
    planner_mlp = nn.Sequential(
        nn.Linear(fused_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, vocab_size),
    )
```
ARCH-03: Insert a middle layer. Change to:
```python
if planner_mlp is None:
    planner_mlp = nn.Sequential(
        nn.Linear(fused_dim * 2, hidden_dim * 2),  # 256 -> 256
        nn.ReLU(),
        nn.Linear(hidden_dim * 2, hidden_dim),      # 256 -> 128
        nn.ReLU(),
        nn.Linear(hidden_dim, vocab_size),           # 128 -> V
    )
```
Everything else in the file (input concat, softmax, top-k, return dict) stays identical.

---

### `metacontroller/trainer.py` (service, CRUD) -- checkpoint compatibility update

**Analog:** self (current file)

**TrainingState constructor pattern** (lines 614-647):
```python
class TrainingState:
    def __init__(
        self,
        meta_mlp,
        reward_mlp,
        rf_predictor,
        lr=DEFAULT_LR,
        eps=DEFAULT_EPS,
        max_grad_norm=DEFAULT_MAX_GRAD_NORM,
        batch_size=DEFAULT_BATCH_SIZE,
        buffer_capacity=DEFAULT_BUFFER_CAPACITY,
    ):
        # ...
        self.optimizer_meta = Adam(meta_mlp.parameters(), lr=lr, eps=eps)
```
This already uses `meta_mlp.parameters()` which works for any `nn.Module` subclass including the new `MetaMLP`. No structural change needed to the constructor.

**Checkpoint save pattern** (lines 925-932):
```python
meta_path = session_dir / "meta_mlp.pt"
torch.save({
    "model_state_dict": meta_mlp.state_dict(),
    "optimizer_state_dict": self.optimizer_meta.state_dict(),
    "step_count": self.step_count,
    "batch_size": self.batch_size,
    "max_grad_norm": self.max_grad_norm,
}, meta_path)
```
This already uses `meta_mlp.state_dict()` which works for any `nn.Module`. No change needed.

**Checkpoint load pattern** (lines 1020-1026):
```python
meta_path = session_dir / "meta_mlp.pt"
if meta_path.exists():
    ckpt = torch.load(meta_path, map_location="cpu", weights_only=True)
    meta_mlp.load_state_dict(ckpt["model_state_dict"])
    self.optimizer_meta.load_state_dict(ckpt["optimizer_state_dict"])
    self.step_count = ckpt["step_count"]
```
This will crash if loading a Phase 2 checkpoint into the new MetaMLP (different state_dict keys/shapes). The update_metapolicy function docstring at line 281 references `nn.Sequential` -- update the docstring only.

**Key finding:** `trainer.py` needs only docstring updates, not structural changes. The `state_dict()`/`load_state_dict()` API works identically for `nn.Sequential` and custom `nn.Module` subclasses.

---

### `tests/conftest.py` (config, test-fixture) -- mock fixture update

**Analog:** self (current file)

**Import pattern** (lines 1-12):
```python
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
if str(METACONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(METACONTROLLER_DIR))
```
ARCH-01 update: Add `from metacontroller import MetaMLP` after the sys.path setup.

**Mock MLP fixture pattern** (lines 14-22) -- the pattern being MODIFIED:
```python
@pytest.fixture
def mock_meta_mlp():
    """A small MLP matching metacontroller's lazy init pattern: Linear->ReLU->Linear(4)."""
    input_dim = 10
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    )
```
ARCH-01 update: Replace with `return MetaMLP(input_dim=10, output_dim=4)`. The test-size `input_dim=10` is intentionally smaller than production's 237 for fast tests.

**Mock trajectory fixture pattern** (lines 25-36) -- KEEP UNCHANGED:
```python
@pytest.fixture
def mock_meta_trajectory():
    return [
        {"decision": 0, "features": torch.randn(1, 10), "predicted_q": 0.1},
        {"decision": 0, "features": torch.randn(1, 10), "predicted_q": 0.15},
        {"decision": 2, "features": torch.randn(1, 10), "predicted_q": 0.2},
    ]
```
Feature dim=10 matches `mock_meta_mlp`'s `input_dim=10`. No change needed.

**Helper function pattern** (lines 110-131):
```python
def _make_trajectory_dict(input_dim=10, n_steps=3):
    meta_trajectory = [
        {"decision": i % 4, "features": torch.randn(1, input_dim), "predicted_q": 0.1 * i}
        for i in range(n_steps)
    ]
    return {
        "meta_trajectory": meta_trajectory,
        "realized_return": 0.5,
        # ...
    }
```
No change needed -- `input_dim=10` matches the fixture pattern.

---

### `tests/test_architecture.py` (test, unit) -- NEW FILE

**Analog:** `tests/test_batch_training.py`

**Test file boilerplate pattern** (test_batch_training.py lines 1-34):
```python
"""
Tests for Phase 2: Batch Training and Checkpointing (BATCH-01 through BATCH-04).

Each test validates one requirement from REQUIREMENTS.md.
Tests use synthetic data -- they do not require GTA or real game states.
"""

import sys
from pathlib import Path
import math

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
TESTS_DIR = PROJECT_ROOT / "tests"
if str(METACONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(METACONTROLLER_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from trainer import (
    TrainingState,
    train_step,
    compute_token_return,
    # ...
)
from conftest import _make_trajectory_dict
```

**Test function naming pattern** (test_batch_training.py lines 41-63):
```python
def test_buffer_accumulation():
    """
    BATCH-01: Buffer accumulates trajectory dicts and reports not-ready
    until batch_size is reached.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    # ... setup ...
    assert len(ts.buffer) == 3, (
        f"Buffer should have 3 entries, got {len(ts.buffer)}"
    )
```
Convention: test function named `test_<behavior>`, docstring starts with requirement ID (e.g., `ARCH-01:`), assertions include descriptive error messages.

**Inline module creation pattern** (test_batch_training.py lines 47-49):
```python
input_dim = 10
meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))
```
For test_architecture.py: use `MetaMLP(input_dim=10)` instead of `nn.Sequential`, and import from metacontroller. Use `create_encoder_weights()` for encoder tests. Use action_planner's lazy init for planner tests.

---

## Shared Patterns

### sys.path Resolution
**Source:** `tests/test_batch_training.py` lines 15-21, `metacontroller/trainer.py` lines 33-36, `main_model/main_model.py` lines 34-39
**Apply to:** `tests/test_architecture.py` (new file), `tests/conftest.py` (add metacontroller import)

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
if str(METACONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(METACONTROLLER_DIR))
```
This project does not use packages or pip install -e. All cross-module imports are resolved by manually inserting parent directories into sys.path. Every new file that imports across module boundaries must follow this pattern.

### Lazy Init Guard
**Source:** `metacontroller/metacontroller.py` line 151, `action_planner/action_planner.py` line 68
**Apply to:** Both ARCH-01 and ARCH-03 modifications

```python
if meta_mlp is None:
    meta_mlp = MetaMLP()  # or nn.Sequential(...)
```
All modules use the "pass None on first call, create-and-return" pattern. The module is returned in the output dict so the caller can pass it back on subsequent calls. This pattern is unchanged by the architecture upgrades -- only the created object changes.

### Encoder Weight Dict Pattern
**Source:** `main_model/main_model.py` lines 203-212
**Apply to:** ARCH-02 modifications

```python
return {
    "ego_mlp":    ego_mlp,
    "scene_mlp":  scene_mlp,
    # ... nn.Sequential and nn.Parameter objects mixed in a plain dict ...
    "qw": qw, "kw": kw, "vw": vw, "ow": ow,
    "embed_dim": embed_dim,
    "num_heads": num_heads,
}
```
The encoder uses a plain dict (not an `nn.Module`) to hold its weights. This means `nn.LayerNorm` modules added for ARCH-02 are stored as dict values alongside `nn.Parameter` objects. This is the established pattern -- do not refactor to `nn.Module` in Phase 3.

### Checkpoint Serialization (weights_only=True)
**Source:** `metacontroller/trainer.py` lines 1022, 1031, 1038, 1045, 1053, 1060, 1067
**Apply to:** `tests/test_architecture.py` (save/load roundtrip tests)

```python
ckpt = torch.load(meta_path, map_location="cpu", weights_only=True)
meta_mlp.load_state_dict(ckpt["model_state_dict"])
```
All `torch.load` calls use `weights_only=True` for security (prevents pickle code execution). All `torch.save` calls use `{"model_state_dict": module.state_dict()}` dict wrapping. Test roundtrips must follow the same pattern.

### Test Assertion Style
**Source:** `tests/test_batch_training.py` lines 58-63
**Apply to:** `tests/test_architecture.py`

```python
assert len(ts.buffer) == 3, (
    f"Buffer should have 3 entries, got {len(ts.buffer)}"
)
```
All assertions include f-string error messages showing expected vs actual values. Multi-line assertions use parenthesized continuation.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | -- | -- | All files have exact or role-match analogs within the existing codebase |

Every file to be created or modified has a direct analog in the codebase. The `MetaMLP` class is the only genuinely new construct, and its pattern is well-defined by PyTorch's `nn.Module` API combined with the existing project conventions for lazy init and checkpoint serialization.

## Metadata

**Analog search scope:** `metacontroller/`, `main_model/`, `action_planner/`, `tests/`
**Files scanned:** 8 (metacontroller.py, trainer.py, main_model.py, multi_head_attention.py, action_planner.py, conftest.py, test_batch_training.py, test_training_correctness.py)
**Pattern extraction date:** 2026-04-30
