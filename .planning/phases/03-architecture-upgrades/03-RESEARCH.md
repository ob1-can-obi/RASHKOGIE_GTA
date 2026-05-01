# Phase 3: Architecture Upgrades - Research

**Researched:** 2026-04-30
**Domain:** PyTorch nn.Module architecture -- deeper MLPs, skip connections, LayerNorm, stacked multi-head attention, checkpoint compatibility
**Confidence:** HIGH

## Summary

Phase 3 upgrades three neural network modules to have sufficient capacity for their 237-dimension metacontroller input and normalized hidden representations. The four requirements are structurally independent (each touches a different module or constant) but share a common concern: every architecture change must produce modules that serialize and reload cleanly via `torch.save`/`torch.load`, which is tested by the existing Phase 2 checkpoint infrastructure.

The primary technical challenge is the metacontroller MLP upgrade (ARCH-01). The current metacontroller lazily creates a 2-layer `nn.Sequential` inside `metacontroller()` when `meta_mlp is None`. The requirement demands a 3-hidden-layer MLP (256-256-128-4) with a skip connection from input to layer 2 and LayerNorm at every hidden layer. A skip connection cannot be expressed as `nn.Sequential` because it requires additive composition of two tensors from different points in the computation graph. This means ARCH-01 must replace `nn.Sequential` with a proper `nn.Module` subclass. The same change propagates to every call site that currently creates or references the metacontroller MLP via `nn.Sequential`, including the `TrainingState` checkpoint save/load, the test fixtures in `conftest.py`, and the lazy init fallback in `metacontroller()`.

The encoder attention upgrade (ARCH-02) adds a second attention block with LayerNorm after each. The current single-block architecture uses bare projection matrices (`nn.Parameter`) with a query dimension of `embed_dim * 3 = 192` (ego+scene+route concatenated). The second block must use the attention output (dim 64) as its query, so it needs its own set of Q/K/V/O weights with `query_dim = embed_dim = 64`. The entity embeddings (K/V source) remain the same across both blocks.

The action planner (ARCH-03) and input dimension pinning (ARCH-04) are straightforward changes.

**Primary recommendation:** Create a `MetaMLP` module class in `metacontroller.py` that encapsulates the 3-layer architecture with skip connection and LayerNorm. Pin `META_INPUT_DIM = 237` as a constant derived from the known feature composition. Update `create_encoder_weights()` to include a second attention block's projection weights plus two LayerNorm modules. Upgrade the action planner to a 2-layer MLP in its lazy init path. Verify all four modules save/load cleanly.

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ARCH-01 | Metacontroller MLP upgraded to 3 layers (256-256-128-4) with skip connection and LayerNorm | Requires `nn.Module` subclass (`MetaMLP`) replacing current `nn.Sequential`. Skip connection from input to layer 2 needs a learned projection (237 -> 256). LayerNorm at each hidden layer. See Pattern 1. |
| ARCH-02 | Encoder attention upgraded to 2 blocks with LayerNorm (keep 4 heads, head_dim=16) | Add second set of Q/K/V/O weights with `query_dim=64` (not 192). LayerNorm after each block on the entity_context output. Residual connection on second block. See Pattern 2. |
| ARCH-03 | Action planner upgraded to 2-layer MLP | Change lazy init from `Linear(256,128)->ReLU->Linear(128,V)` to `Linear(256,256)->ReLU->Linear(256,128)->ReLU->Linear(128,V)`. See Pattern 3. |
| ARCH-04 | Input dimension pinned as a constant (not recomputed dynamically on every call) | Replace `input_dim = features.shape[-1]` at line 145 of `metacontroller.py` with `META_INPUT_DIM = 237` constant. Compute once from known feature layout. See Pattern 4. |

</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Metacontroller MLP architecture | metacontroller/metacontroller.py | metacontroller/trainer.py (checkpoint) | MLP defined in metacontroller.py, but TrainingState and test fixtures reference its structure for optimizers and checkpoint save/load |
| Encoder attention stacking | main_model/main_model.py + multi_head_attention.py | -- | Encoder weights dict and encode_state() both live in main_model; multi_head_attention is a helper |
| Action planner depth | action_planner/action_planner.py | -- | Self-contained lazy init pattern |
| Input dimension constant | metacontroller/metacontroller.py | -- | Single file change, but validated by integration with search_tree.py callers |
| Checkpoint compatibility | metacontroller/trainer.py | tests/conftest.py, tests/test_batch_training.py | TrainingState.save_checkpoint/load_checkpoint must handle new module shapes; test fixtures must match |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.11.0 | Neural network modules, LayerNorm, save/load | Already installed in project venv [VERIFIED: venv python] |
| pytest | 9.0.3 | Test framework for validation tests | Already installed in project venv [VERIFIED: venv python] |

### Key APIs Used
| API | Module | Purpose |
|-----|--------|---------|
| `nn.Module` | torch.nn | Custom module class for MetaMLP with skip connection [VERIFIED: PyTorch docs] |
| `nn.LayerNorm(dim)` | torch.nn | Normalization at every hidden layer [VERIFIED: runtime test] |
| `nn.Linear(in, out)` | torch.nn | Dense layers in upgraded MLPs [VERIFIED: existing codebase] |
| `nn.ReLU()` | torch.nn | Activation function [VERIFIED: existing codebase] |
| `nn.Parameter` | torch.nn | Attention projection weights [VERIFIED: existing codebase] |
| `torch.save` / `torch.load` | torch | Checkpoint serialization [VERIFIED: existing codebase] |

**No new dependencies required.** All changes use PyTorch APIs already in the project.

## Architecture Patterns

### System Architecture Diagram

```
                        Current Architecture
                        ====================

metacontroller.py:
  features [237] --> nn.Sequential(Linear(237,128), ReLU, Linear(128,4)) --> logits [4]
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     2 layers, no skip, no LayerNorm, input_dim dynamic

main_model.py (encoder):
  [ego_emb|scene_emb|route_emb] --> attention(Q/K/V/O) --> entity_context [64]
                                    ^^^^^^^^^^^^^^^^
                                    1 block, no LayerNorm

  [ego_emb|scene_emb|route_emb|entity_context] --> fusion_mlp --> z_t [128]

action_planner.py:
  [z_t|z_next_pred] [256] --> nn.Sequential(Linear(256,128), ReLU, Linear(128,V)) --> logits [V]
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              1 hidden layer


                        Target Architecture
                        ====================

metacontroller.py:
  features [237] --> MetaMLP(
                       Linear(237,256) -> LN(256) -> ReLU          -- h1
                       Linear(256,256) + skip_proj(237,256) -> LN(256) -> ReLU  -- h2 (skip from input)
                       Linear(256,128) -> LN(128) -> ReLU          -- h3
                       Linear(128,4)                                -- output
                     ) --> logits [4]

main_model.py (encoder):
  [ego_emb|scene_emb|route_emb] --> attention_block_1(Q1/K1/V1/O1) --> LN --> ctx1 [64]
  ctx1                           --> attention_block_2(Q2/K2/V2/O2) + ctx1 --> LN --> ctx2 [64]
                                                                                      ^^^^
                                                                   residual connection + LayerNorm

  [ego_emb|scene_emb|route_emb|ctx2] --> fusion_mlp --> z_t [128]

action_planner.py:
  [z_t|z_next_pred] [256] --> nn.Sequential(
                                Linear(256,256), ReLU,
                                Linear(256,128), ReLU,
                                Linear(128,V)
                              ) --> logits [V]
```

### Recommended Changes by File

```
metacontroller/
  metacontroller.py      # ARCH-01: MetaMLP class, META_INPUT_DIM constant (ARCH-04)
main_model/
  main_model.py          # ARCH-02: create_encoder_weights + encode_state (2nd attn block + LN)
action_planner/
  action_planner.py      # ARCH-03: 2-layer MLP in lazy init
metacontroller/
  trainer.py             # Update: TrainingState checkpoint must handle MetaMLP (not nn.Sequential)
tests/
  conftest.py            # Update: mock_meta_mlp fixture must use MetaMLP class
  test_architecture.py   # NEW: validation tests for ARCH-01 through ARCH-04
```

### Pattern 1: MetaMLP with Skip Connection and LayerNorm (ARCH-01)

**What:** Replace the 2-layer `nn.Sequential` metacontroller MLP with a proper `nn.Module` subclass that has 3 hidden layers (256-256-128), LayerNorm at every hidden layer, and a skip connection from input to layer 2.

**Why nn.Module, not nn.Sequential:** A skip connection adds the original input (projected to match dimensions) to the output of layer 2. This requires referencing two different tensors at the same point, which `nn.Sequential`'s linear forward chain cannot express.

**When to use:** Always -- this replaces the existing metacontroller MLP everywhere.

**Example:**
```python
# Source: PyTorch nn.Module docs + project requirements
META_INPUT_DIM = 237  # ARCH-04: pinned constant

class MetaMLP(nn.Module):
    """Metacontroller decision MLP with skip connection and LayerNorm.

    Architecture: 237 -> 256 -> 256 -> 128 -> 4
    Skip connection: input projected to 256 and added at layer 2
    LayerNorm: applied at every hidden layer (before activation)
    """
    def __init__(self, input_dim=META_INPUT_DIM, output_dim=4):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)

        self.layer2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.skip_proj = nn.Linear(input_dim, 256)  # project input to match layer2 dim

        self.layer3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)

        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.ln1(self.layer1(x)))
        h2 = self.relu(self.ln2(self.layer2(h1) + self.skip_proj(x)))
        h3 = self.relu(self.ln3(self.layer3(h2)))
        return self.out(h3)
```

**Critical integration points:**
1. `metacontroller()` line 151-156: Replace `nn.Sequential(...)` with `MetaMLP()`
2. `metacontroller()` line 145: Replace `input_dim = features.shape[-1]` with assertion `assert features.shape[-1] == META_INPUT_DIM`
3. `tests/conftest.py` line 18-22: `mock_meta_mlp` fixture must create `MetaMLP(input_dim=10)` (test-size input)
4. `trainer.py` TrainingState: `optimizer_meta` and checkpoint save/load work with any `nn.Module` -- no changes needed IF MetaMLP uses standard `state_dict()` (which it does as an `nn.Module` subclass)

[VERIFIED: runtime test confirmed MetaMLP save/load roundtrip works correctly]

### Pattern 2: Stacked Attention Blocks with LayerNorm (ARCH-02)

**What:** Replace single attention block with 2 stacked blocks, each followed by LayerNorm. The second block uses a residual connection.

**Architecture detail:**
- Block 1: query_dim = embed_dim * 3 = 192 (ego+scene+route concatenated) -- same as current
- Block 1 output: entity_context_1 [batch, 64] followed by LayerNorm(64)
- Block 2: query_dim = embed_dim = 64 (uses output of block 1 as query)
- Block 2 output: entity_context_2 + entity_context_1 (residual) followed by LayerNorm(64)
- K/V source: same entity_embs for both blocks (re-projected with different weights)

**Example:**
```python
# Source: Standard transformer block pattern, project ARCH-02 requirement
def create_encoder_weights(embed_dim=64, hidden_dim=128, fused_dim=128, num_heads=4):
    # ... existing sub-MLPs (ego, scene, route, entity, fusion) unchanged ...

    # Attention block 1 (same as current)
    qw1, kw1, vw1, ow1 = create_multi_head_attention_weights(embed_dim * 3, embed_dim)
    ln_attn1 = nn.LayerNorm(embed_dim)

    # Attention block 2 (query from block 1 output, not concatenated ego/scene/route)
    qw2, kw2, vw2, ow2 = create_multi_head_attention_weights(embed_dim, embed_dim)
    ln_attn2 = nn.LayerNorm(embed_dim)

    return {
        # ... existing sub-MLPs ...
        "qw1": qw1, "kw1": kw1, "vw1": vw1, "ow1": ow1, "ln_attn1": ln_attn1,
        "qw2": qw2, "kw2": kw2, "vw2": vw2, "ow2": ow2, "ln_attn2": ln_attn2,
        "embed_dim": embed_dim, "num_heads": num_heads,
    }

def encode_state(raw_state, weights):
    # ... build sub-embeddings as before ...

    # Block 1: cross-attention from ego/scene/route query to entity K/V
    query_input = torch.cat([ego_emb, scene_emb, route_emb], dim=-1)  # [1, 192]
    attn1 = multi_head_attention(
        query_input, entity_embs, mask,
        weights["qw1"], weights["kw1"], weights["vw1"], weights["ow1"],
        weights["num_heads"],
    )
    ctx1 = weights["ln_attn1"](attn1["entity_context"])  # [1, 64]

    # Block 2: self-refine using block 1 output as query, same entity K/V
    attn2 = multi_head_attention(
        ctx1, entity_embs, mask,
        weights["qw2"], weights["kw2"], weights["vw2"], weights["ow2"],
        weights["num_heads"],
    )
    ctx2 = weights["ln_attn2"](attn2["entity_context"] + ctx1)  # residual + LN

    # Fusion: same as before but uses ctx2
    fusion_input = torch.cat([ego_emb, scene_emb, route_emb, ctx2], dim=-1)
    z_t = weights["fusion_mlp"](fusion_input)
    return z_t
```

**Key considerations:**
- The weight dict keys change from `qw/kw/vw/ow` to `qw1/kw1/vw1/ow1` and `qw2/kw2/vw2/ow2` -- this is a **breaking change** for existing checkpoints
- `ln_attn1` and `ln_attn2` are `nn.LayerNorm` modules stored in the weight dict (they have trainable `weight` and `bias` parameters)
- `multi_head_attention()` function itself needs no changes -- it already accepts explicit Q/K/V/O weights
- The residual connection on block 2 is safe because both ctx1 and attn2 output have the same shape [batch, embed_dim=64]
- No residual on block 1 because the query dimension (192) differs from the output dimension (64)

[VERIFIED: embed_dim=64 is divisible by num_heads=4, giving head_dim=16 as required]

### Pattern 3: Action Planner 2-Layer MLP (ARCH-03)

**What:** Add a second hidden layer to the action planner MLP.

**Current:** `Linear(256, 128) -> ReLU -> Linear(128, vocab_size)`
**Target:** `Linear(256, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, vocab_size)`

**Example:**
```python
# Source: action_planner.py lazy init, project ARCH-03 requirement
if planner_mlp is None:
    planner_mlp = nn.Sequential(
        nn.Linear(fused_dim * 2, hidden_dim * 2),   # 256 -> 256 (was 256 -> 128)
        nn.ReLU(),
        nn.Linear(hidden_dim * 2, hidden_dim),       # 256 -> 128
        nn.ReLU(),
        nn.Linear(hidden_dim, vocab_size),            # 128 -> V
    )
```

**Integration:** The action planner still uses `nn.Sequential`, so no structural changes are needed beyond updating the lazy init. The `hidden_dim` parameter is 128 by default, so the first layer uses `hidden_dim * 2 = 256`.

### Pattern 4: Pinned Input Dimension Constant (ARCH-04)

**What:** Replace the dynamic `input_dim = features.shape[-1]` with a computed constant.

**Current problem:** At line 145 of `metacontroller.py`, `input_dim` is recomputed from the tensor shape on every call. If the feature composition changes unexpectedly (e.g., wrong top_k or embed_dim), the MLP silently creates with the wrong input dimension and fails much later.

**Computation:**
```python
# Source: metacontroller.py lines 134-143, verified by dimensional analysis
# drift:                       fused_dim = 128
# elapsed_ratio:               1
# token_frames_left:           1
# best_q:                      1
# mean_q:                      1
# urgency:                     1
# parent_unexplored:           1
# current_path_value:          1
# best_path_value:             1
# best_current_q:              1
# mean_current_q:              1
# current_candidate_durations: top_k = 3
# current_candidate_emb:       top_k * token_embed_dim = 3 * 32 = 96
#
# Total: 128 + 10 + 3 + 96 = 237

META_INPUT_DIM = 237
```

[VERIFIED: computed independently and confirmed = 237]

**Implementation:** Define `META_INPUT_DIM = 237` as a module-level constant in `metacontroller.py`. Add a runtime assertion `assert features.shape[-1] == META_INPUT_DIM` to catch mismatches early. The `MetaMLP` class uses this constant as its default `input_dim`.

### Anti-Patterns to Avoid

- **Lazy init with wrong dimensions:** The current pattern `if meta_mlp is None: meta_mlp = nn.Sequential(...)` uses `input_dim = features.shape[-1]`. After ARCH-04, the lazy init should use the pinned constant, not re-derive from the tensor. The assertion guards against drift.

- **Skip connection with dimension mismatch:** The skip connection adds `skip_proj(input)` to `layer2(h1)`. Both must produce tensors of the same size (256). The `skip_proj` linear layer handles the 237->256 projection. Do NOT attempt to add tensors of different sizes.

- **Breaking checkpoint compatibility silently:** Changing MLP layer sizes or the attention weight dict keys means old checkpoints cannot be loaded into the new architecture. This is expected and acceptable (Phase 3 is an architecture upgrade, not a migration), but the code should fail loudly if shapes mismatch rather than silently producing garbage.

- **Forgetting LayerNorm parameters in checkpoint:** `nn.LayerNorm` has trainable `weight` and `bias`. If the encoder weights dict stores `ln_attn1` and `ln_attn2`, these must be included in checkpoint save/load. Since the encoder currently uses a plain dict (not an `nn.Module`), the planner must ensure the LayerNorm modules are properly saved.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Layer normalization | Custom mean/variance normalization | `nn.LayerNorm(dim)` | Handles numerical stability, has trainable affine parameters, works with autograd |
| Skip connection module | Inline tensor addition without proper class | `nn.Module` subclass with named layers | Required for `state_dict()` serialization and optimizer parameter discovery |
| Attention weight management | Manual parameter tracking | `create_multi_head_attention_weights()` (existing helper) | Already handles Xavier init and proper Parameter wrapping |

**Key insight:** The only "new" construct needed is a single `nn.Module` subclass (`MetaMLP`). Everything else uses existing PyTorch primitives and the project's existing helper functions.

## Common Pitfalls

### Pitfall 1: Optimizer State Shape Mismatch After Architecture Change
**What goes wrong:** After changing the MLP architecture (different number of parameters), loading an old optimizer checkpoint causes a shape mismatch crash because Adam stores per-parameter momentum and variance tensors.
**Why it happens:** `TrainingState.load_checkpoint()` calls `self.optimizer_meta.load_state_dict(ckpt["optimizer_state_dict"])` which fails if parameter count or shapes changed.
**How to avoid:** Old checkpoints from Phase 2 are incompatible with Phase 3's architecture. Either: (a) skip optimizer state loading when shapes don't match (catch the RuntimeError), or (b) document that Phase 3 requires fresh training (no checkpoint resume from Phase 2 sessions). Option (b) is simpler and appropriate since the architecture is fundamentally different.
**Warning signs:** `RuntimeError: Error(s) in loading state_dict` mentioning size mismatches.

### Pitfall 2: Encoder Weight Dict Key Renaming Breaks Callers
**What goes wrong:** Renaming `qw` to `qw1` in the encoder weights dict breaks every call site that accesses `weights["qw"]`.
**Why it happens:** The encoder weights are a plain dict, not a typed object. There's no compile-time check for key names.
**How to avoid:** Search for all references to `weights["qw"]`, `weights["kw"]`, `weights["vw"]`, `weights["ow"]` and update them all to `weights["qw1"]`, etc. The only call site is `encode_state()` in `main_model.py`.
**Warning signs:** `KeyError: 'qw'` at runtime.

### Pitfall 3: LayerNorm in Encoder Weights Dict Not Tracked by Optimizer
**What goes wrong:** The encoder weights dict stores `nn.Parameter` objects and now `nn.LayerNorm` modules. But there is no encoder optimizer yet (encoder training is Phase 4). The risk is that when Phase 4 creates an optimizer, it might miss the LayerNorm parameters.
**Why it happens:** The encoder weights are a plain dict, not an `nn.Module`. PyTorch's `parameters()` method only works on `nn.Module` subclasses.
**How to avoid:** For Phase 3, this is not immediately dangerous because the encoder is not being trained yet. But document that the encoder weights dict now contains both `nn.Parameter` and `nn.LayerNorm` objects so Phase 4 can iterate correctly. A future improvement would be to wrap the encoder in an `nn.Module`, but that is out of scope for Phase 3.
**Warning signs:** In Phase 4, encoder loss not decreasing because LayerNorm parameters are not included in the optimizer.

### Pitfall 4: Action Planner Checkpoint Shape Mismatch
**What goes wrong:** Old planner_mlp checkpoints have shapes `[(256,128), (128,), (128,V), (V,)]` but the new architecture has `[(256,256), (256,), (256,128), (128,), (128,V), (V,)]`. Loading the old checkpoint into the new MLP crashes.
**Why it happens:** Same as Pitfall 1 -- architecture upgrade changes parameter shapes.
**How to avoid:** Same strategy as metacontroller: document that Phase 3 checkpoints are incompatible with Phase 2 checkpoints. Start training fresh after the upgrade.
**Warning signs:** `size mismatch for 0.weight` errors during `load_state_dict`.

### Pitfall 5: MetaMLP Input Dimension Assertion Failure
**What goes wrong:** The runtime assertion `assert features.shape[-1] == META_INPUT_DIM` fires because a caller passes features with the wrong dimension.
**Why it happens:** If `top_k` or `embed_dim` is changed without updating `META_INPUT_DIM`, or if the feature vector composition in `metacontroller()` is modified.
**How to avoid:** Define `META_INPUT_DIM` as a computed constant derived from the feature layout, not a magic number. Include a comment documenting the breakdown. Consider computing it from the actual constants (`FUSED_DIM`, `TOP_K`, `TOKEN_EMBED_DIM`) rather than hardcoding 237.
**Warning signs:** `AssertionError` on the first forward pass.

### Pitfall 6: Skip Connection Gradient Scale
**What goes wrong:** The skip connection adds the raw projected input to layer 2's output. If the input has much larger magnitude than the layer output, the skip dominates and the intermediate layers learn nothing.
**Why it happens:** Without normalization, the skip path and the main path may operate at different scales.
**How to avoid:** LayerNorm is applied AFTER the addition (`ln2(layer2(h1) + skip_proj(x))`), which normalizes the combined signal. This is the standard pre-activation residual pattern used in transformers and works correctly here.
**Warning signs:** Layer 1 weights not updating during training (gradient near zero).

## Code Examples

### Current vs Target Metacontroller MLP

```python
# CURRENT (metacontroller.py line 151-156):
if meta_mlp is None:
    meta_mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),  # 237 -> 128
        nn.ReLU(),
        nn.Linear(hidden_dim, 4),          # 128 -> 4
    )

# TARGET (ARCH-01 + ARCH-04):
META_INPUT_DIM = 237

class MetaMLP(nn.Module):
    def __init__(self, input_dim=META_INPUT_DIM, output_dim=4):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.skip_proj = nn.Linear(input_dim, 256)
        self.layer3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.ln1(self.layer1(x)))
        h2 = self.relu(self.ln2(self.layer2(h1) + self.skip_proj(x)))
        h3 = self.relu(self.ln3(self.layer3(h2)))
        return self.out(h3)

# Usage in metacontroller():
if meta_mlp is None:
    meta_mlp = MetaMLP()
assert features.shape[-1] == META_INPUT_DIM, (
    f"Expected {META_INPUT_DIM}, got {features.shape[-1]}"
)
```

### Current vs Target Encoder Attention

```python
# CURRENT (main_model.py encode_state):
query_input = torch.cat([ego_emb, scene_emb, route_emb], dim=-1)
attn = multi_head_attention(query_input, entity_embs, mask,
                            weights["qw"], weights["kw"], weights["vw"], weights["ow"],
                            weights["num_heads"])
entity_context = attn["entity_context"]  # [1, 64]

# TARGET (ARCH-02):
query_input = torch.cat([ego_emb, scene_emb, route_emb], dim=-1)

# Block 1
attn1 = multi_head_attention(query_input, entity_embs, mask,
                             weights["qw1"], weights["kw1"], weights["vw1"], weights["ow1"],
                             weights["num_heads"])
ctx1 = weights["ln_attn1"](attn1["entity_context"])

# Block 2 with residual
attn2 = multi_head_attention(ctx1, entity_embs, mask,
                             weights["qw2"], weights["kw2"], weights["vw2"], weights["ow2"],
                             weights["num_heads"])
entity_context = weights["ln_attn2"](attn2["entity_context"] + ctx1)  # residual + LN
```

### Current vs Target Action Planner

```python
# CURRENT (action_planner.py line 69-73):
if planner_mlp is None:
    planner_mlp = nn.Sequential(
        nn.Linear(fused_dim * 2, hidden_dim),   # 256 -> 128
        nn.ReLU(),
        nn.Linear(hidden_dim, vocab_size),       # 128 -> V
    )

# TARGET (ARCH-03):
if planner_mlp is None:
    planner_mlp = nn.Sequential(
        nn.Linear(fused_dim * 2, hidden_dim * 2),  # 256 -> 256
        nn.ReLU(),
        nn.Linear(hidden_dim * 2, hidden_dim),      # 256 -> 128
        nn.ReLU(),
        nn.Linear(hidden_dim, vocab_size),           # 128 -> V
    )
```

### Checkpoint Save/Load Compatibility Test

```python
# Source: project test patterns + ARCH-04 success criterion
def test_save_load_roundtrip():
    """ARCH-04: All modules serialize and reload with no shape mismatches."""
    from metacontroller import MetaMLP, META_INPUT_DIM

    meta = MetaMLP()
    x = torch.randn(1, META_INPUT_DIM)
    out_before = meta(x)

    # Save
    torch.save({"model_state_dict": meta.state_dict()}, "/tmp/test_meta.pt")

    # Reload into fresh module
    meta2 = MetaMLP()
    ckpt = torch.load("/tmp/test_meta.pt", map_location="cpu", weights_only=True)
    meta2.load_state_dict(ckpt["model_state_dict"])

    out_after = meta2(x)
    assert torch.allclose(out_before, out_after)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Dynamic input_dim from tensor shape | Pinned constant with assertion | This phase | Catches dimension mismatches at first forward pass instead of producing silent garbage |
| Single attention block | 2 stacked blocks with residual + LN | This phase | Standard transformer pattern; improves entity context quality through iterative refinement |
| nn.Sequential for all MLPs | nn.Module subclass when skip needed | This phase | Required for skip connections; Sequential still used where sufficient (action planner) |

**Deprecated/outdated:**
- None -- all PyTorch APIs used are current and stable in 2.11.0

## Assumptions Log

> List all claims tagged [ASSUMED] in this research.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Layer sizes 256-256-128 are specified in requirements, not computed from any principle | Pattern 1 | None -- these are explicitly locked in REQUIREMENTS.md |
| A2 | The second attention block should use entity_embs (same K/V source) rather than different data | Pattern 2 | LOW -- this follows standard transformer cross-attention stacking; ARCH-02 says "2 blocks" without specifying different K/V sources |
| A3 | Residual connection should be on block 2 only (not block 1) because query_dim (192) != output_dim (64) for block 1 | Pattern 2 | LOW -- standard practice; adding a projection for block 1 residual would work but adds complexity not specified in requirements |
| A4 | Old Phase 2 checkpoints are expected to be incompatible with Phase 3 architecture | Pitfall 1, 4 | MEDIUM -- if users expect checkpoint continuity, this needs explicit documentation. The roadmap says Phase 3 precedes Phase 4 training, so no trained checkpoints of value exist yet |
| A5 | Action planner first hidden layer should be 256 (hidden_dim * 2), not 128, to match the "2-layer MLP" requirement | Pattern 3 | LOW -- ARCH-03 says "2-layer MLP" meaning 2 hidden layers; the first hidden dim is a design choice. Using hidden_dim * 2 = 256 matches the wider-then-narrow pattern of the metacontroller |

## Open Questions

1. **Should META_INPUT_DIM be computed from constants or hardcoded as 237?**
   - What we know: The breakdown is 128 (fused_dim) + 10 (scalars) + 3 (top_k durations) + 96 (top_k * embed_dim) = 237. All constituent values are already constants in the codebase (FUSED_DIM=128, top_k=3, embed_dim=32).
   - What's unclear: Whether to define `META_INPUT_DIM = FUSED_DIM + 10 + TOP_K + TOP_K * TOKEN_EMBED_DIM` (explicit) or `META_INPUT_DIM = 237` (simple).
   - Recommendation: Use the computed form with a comment documenting the breakdown. This is self-documenting and automatically updates if any constituent constant changes. However, TOP_K and TOKEN_EMBED_DIM are not currently module-level constants -- they would need to be extracted from default function arguments.

2. **Should the encoder weights dict be refactored into an nn.Module?**
   - What we know: The encoder uses a plain dict of `nn.Parameter` objects and `nn.Module` instances (sub-MLPs). Adding LayerNorm modules adds more `nn.Module` instances. Phase 4 will need to iterate all encoder parameters for optimizer construction.
   - What's unclear: Whether to do this refactoring now (Phase 3) or defer to Phase 4.
   - Recommendation: Defer. Phase 3's scope is architecture upgrades, not refactoring the weight management pattern. The encoder weights dict pattern is established and functional. Phase 4 can refactor if needed when implementing the encoder training loop.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | none -- uses default discovery |
| Quick run command | `.venv/bin/python -m pytest tests/ -x -q` |
| Full suite command | `.venv/bin/python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ARCH-01 | MetaMLP has 3 hidden layers (256-256-128), skip connection, LayerNorm at every hidden layer | unit | `.venv/bin/python -m pytest tests/test_architecture.py::test_meta_mlp_structure -x` | Wave 0 |
| ARCH-01 | MetaMLP forward pass produces [batch, 4] from [batch, 237] input | unit | `.venv/bin/python -m pytest tests/test_architecture.py::test_meta_mlp_forward -x` | Wave 0 |
| ARCH-01 | MetaMLP save/load roundtrip produces identical output | unit | `.venv/bin/python -m pytest tests/test_architecture.py::test_meta_mlp_save_load -x` | Wave 0 |
| ARCH-02 | Encoder uses 2 attention blocks, each followed by LayerNorm | unit | `.venv/bin/python -m pytest tests/test_architecture.py::test_encoder_two_attention_blocks -x` | Wave 0 |
| ARCH-02 | Encoder output shape unchanged at [1, 128] after upgrade | unit | `.venv/bin/python -m pytest tests/test_architecture.py::test_encoder_output_shape -x` | Wave 0 |
| ARCH-03 | Action planner MLP has 2 hidden layers | unit | `.venv/bin/python -m pytest tests/test_architecture.py::test_planner_two_layers -x` | Wave 0 |
| ARCH-04 | META_INPUT_DIM constant equals 237 | unit | `.venv/bin/python -m pytest tests/test_architecture.py::test_meta_input_dim_constant -x` | Wave 0 |
| ARCH-04 | All modules serialize to disk and reload cleanly | integration | `.venv/bin/python -m pytest tests/test_architecture.py::test_full_checkpoint_roundtrip -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/python -m pytest tests/test_architecture.py -x -q`
- **Per wave merge:** `.venv/bin/python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_architecture.py` -- covers ARCH-01 through ARCH-04
- [ ] Update `tests/conftest.py` -- `mock_meta_mlp` fixture must use `MetaMLP` class instead of `nn.Sequential`

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | -- |
| V3 Session Management | no | -- |
| V4 Access Control | no | -- |
| V5 Input Validation | no | N/A -- all inputs are locally generated tensors, not user/network input |
| V6 Cryptography | no | -- |

This phase modifies neural network architecture code with no network I/O, no user input, no file system access beyond local checkpoints, and no authentication. No ASVS controls apply.

### Known Threat Patterns for PyTorch

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Pickle deserialization in torch.load | Tampering | `weights_only=True` parameter (already used in codebase per Phase 2 decision) |
| Malicious checkpoint files | Elevation | Only load checkpoints from local trusted directories; `weights_only=True` prevents code execution |

## Sources

### Primary (HIGH confidence)
- **Codebase inspection:** All module files read directly -- metacontroller.py, main_model.py, action_planner.py, reward_head.py, trainer.py, frame_loop.py, search_tree.py, multi_head_attention.py, time_context.py, intuition_head.py
- **PyTorch runtime verification:** `nn.LayerNorm`, `nn.Module` subclass with skip connection, `torch.save`/`torch.load` roundtrip -- all verified in local venv (PyTorch 2.11.0)
- **Dimensional analysis:** META_INPUT_DIM = 237 computed from feature vector composition and verified programmatically

### Secondary (MEDIUM confidence)
- **PyTorch nn.LayerNorm documentation** -- standard API, stable across all PyTorch 2.x versions
- **Transformer residual connection pattern** -- widely documented standard practice

### Tertiary (LOW confidence)
- None -- all claims verified against codebase or runtime

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- PyTorch 2.11.0 verified, all APIs confirmed available
- Architecture: HIGH -- all dimensions computed from source, patterns verified via runtime tests
- Pitfalls: HIGH -- derived from direct codebase analysis of checkpoint save/load paths and integration points

**Research date:** 2026-04-30
**Valid until:** 2026-05-30 (stable -- PyTorch APIs are mature and unchanging)
