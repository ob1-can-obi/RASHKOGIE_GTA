# Architecture Research

**Domain:** Training optimization for module-per-directory autonomous driving agent
**Researched:** 2026-05-04
**Confidence:** HIGH (all findings derived from direct codebase reading — no assumptions)

---

## Focus: v1.1 Integration Analysis

This document answers the specific question: how do compact tensor format, learned embeddings, batched GPU forward passes, and CUDA mixed precision integrate with the existing module-per-directory architecture? What needs to change vs what stays?

---

## Current Architecture (Confirmed by Code Reading)

### Training Loop Pattern — All Three Trainers Do This

Every trainer (`main_model/train.py`, `reward_head/train.py`, `action_planner/train.py`) uses the same pseudo-batch pattern:

```
for epoch:
  shuffle(indices)
  for i in range(0, N, batch_size):       # batch_size = 8 or 16
    batch_indices = indices[i : i+batch_size]
    optimizer.zero_grad()
    total_loss = 0.0

    for idx in batch_indices:             # Python loop — no GPU parallelism
      tensors = pt_data[key][idx].unsqueeze(0).to(device)   # [1, ...] per record
      z = encode_tensors(tensors, weights)                   # [1, 128] forward pass
      loss += compute_loss(z)

    loss /= len(batch_indices)
    loss.backward()
    optimizer.step()
```

This is gradient accumulation over a Python loop. Every forward pass is `[1, ...]`. There is no GPU batching. The RTX 3070 Ti sees `batch_size` separate kernel invocations where it could see one.

### Data Pipeline — Current State

```
GTA V → raw_state dict
    ↓
capture_states.py  →  json.dumps()  →  JSONL line (~370 KB/record avg)
                       [session_*.jsonl written per frame]

                       [OFFLINE — must run before training]
preprocess_data.py:
  json.loads() per line
  build_state_tensors(raw_state)       → ego[1,46], scene[1,16], route[1,14],
                                         entities[1,32,24], mask[1,32]
  torch.cat() over all records
  torch.save()  →  preprocessed.pt    (~800 MB for 236k records)

train.py:
  torch.load("preprocessed.pt")       → all tensors in CPU RAM
  for idx in batch: tensor[idx].unsqueeze(0).to(device)
  encode_tensors([1,...], weights)     → z_t [1, 128]
```

**Key facts confirmed:**
- `preprocess_data.py` already exists and already writes `.pt` format
- `main_model/train.py` already has the `preprocessed.pt` fast path (falls back to streaming JSONL if absent)
- `reward_head/train.py` has the same dual path
- `action_planner/train.py` has NO preprocessed path — still streaming JSONL only
- Current `batch_size` in `training_config.json`: 8 for encoder and reward, 16 for action planner

### Categorical Field Problem — Where It Lives

From `main_model/main_model.py` field lists:

| Field | List | Current position | Problem |
|-------|------|-----------------|---------|
| `v_class` | `ego_fields` | index 37 | Cast via `_to_float()` — meaningless ordinal |
| `weather` | `scene_fields` | index 3 | Cast via `_to_float()` — meaningless ordinal |
| `type_id` | `entity_fields` | index 0 | Cast via `_to_float()` — meaningless ordinal |
| `bucket_id` | `entity_fields` | index 1 | Cast via `_to_float()` — meaningless ordinal |

`_to_float(v)` converts everything to float. `v_class=5` and `v_class=6` are numerically adjacent but may have no semantic relationship.

---

## Target Architecture (v1.1)

### System Data Flow — After Changes

```
GTA V → raw_state dict
    ↓
capture_states.py (MODIFIED)
  build_state_tensors(raw_state)        inline per frame
  extract_reward_features(raw_state)    inline per frame
  buffer tensors in RAM lists
  torch.save(dict_of_tensors) every 1000 frames + on close()
    ↓
  main_model/training_data/session_*.pt    reward_head/training_data/session_*.pt
  [compact tensor format — no JSONL]

                    [OFFLINE — for legacy JSONL only]
preprocess_data.py (MODIFIED — adds categorical ID extraction + action_planner path)

train.py (ALL THREE MODIFIED)
  torch.load all session_*.pt + concatenate
  TensorDataset + DataLoader(batch_size=256, shuffle=True, pin_memory=True)
  for batch in dataloader:
    batch.to(device, non_blocking=True)         # one transfer per B records
    with torch.autocast("cuda", dtype=torch.float16):
      z = encode_tensors(batch, weights)         # [B, 128] — true GPU batch
      loss = compute_loss(z)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(params, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
```

---

## Component Analysis: Change vs Stay

### Files That Stay Unchanged

| File | Reason |
|------|--------|
| `training_utils.py` — `ConvergenceDetector` | Works correctly as-is |
| `training_utils.py` — `freeze_module` | Works correctly as-is — handles both `nn.Module` and dict-of-modules |
| `training_utils.py` — `update_training_status` | No change needed |
| `training_utils.py` — `load_training_config` | No change needed |
| `training_utils.py` — `StreamingJSONLDataset` | Keep as fallback path |
| `main_model/multi_head_attention.py` | Already batch-dimension agnostic — no change needed |
| `reward_head/reward_head.py` | `extract_reward_features` called by preprocessor, not trainer |
| All `*/checkpoints/` structure | Same file format, same `weights_only=True` discipline |
| `training_status.json` schema | Additive only — new keys fine |
| `DashboardParamReceiver` in `main_model/train.py` | Hot-reload still works; batch_size hot-reload becomes meaningful |
| `metacontroller/` | Online RL, not batched offline — out of scope for v1.1 |

### Files That Are Modified

**`capture_states.py`** — Significant rewrite of `StateCaptureSession`

What changes:
- `__init__`: Open two in-RAM buffer dicts instead of two file handles. Buffer lists for each tensor key.
- `record_frame`: Call `build_state_tensors()` and `extract_reward_features()` inline. Append tensors to buffers. Call `_maybe_flush()` every N frames (N=1000 recommended).
- `close()`: Final `torch.save()` of buffered tensors. Close files.
- New method `_maybe_flush()`: If buffer exceeds N frames, call `torch.save()` to append to current session `.pt` (or accumulate per-session and save once at close — see trade-off below).

What stays:
- Session timestamp pattern (`session_YYYYMMDD_HHMMSS`)
- Two output dirs (`main_model/training_data/`, `reward_head/training_data/`)
- The `write_synthetic_*` helpers (update to write `.pt` format)
- The `__main__` block structure

Flush trade-off: A single `.torch.save()` per session at `close()` is simplest and avoids fragmentation, but risks data loss if the game crashes mid-session. Flushing every 1000 frames means partial sessions are recoverable. Recommended: flush every 1000 frames into an accumulator list, save once on close by concatenating all accumulated chunks.

---

**`preprocess_data.py`** — Additive changes only

What changes:
- `preprocess_main_model()`: Add extraction of categorical integer IDs (`cat_weather_t`, `cat_v_class_t`, `cat_type_id`, `cat_bucket_id`) alongside the existing float tensors. Add them as new keys in the output `.pt` dict.
- `preprocess_reward_head()`: Same — add categorical IDs for `before`/`after` states.
- Add `preprocess_action_planner()`: New function. Reads `action_planner/training_data/session_*.jsonl` (schema: `{"state": {...}, "token_id": int}`). Calls `build_state_tensors(state)` and saves to `action_planner/training_data/preprocessed.pt`. This enables true batched training for the action planner.
- Update `__main__` block: Add `--module action` option.

What stays:
- All existing tensor keys in output dicts — backward compatible
- JSONL reading logic — unchanged, just extended

---

**`main_model/main_model.py`** — Moderate additions to encoder

What changes:
- `create_encoder_weights()`: Include embedding tables from `embedding_tables.py` in the returned dict under keys `"cat_embeds"`.
- `encode_state()` and `encode_tensors()`: Before passing ego/scene/entity float vectors to MLPs, look up categorical embeddings and concatenate. The float tensor dimensions (`EGO_DIM`, `SCENE_DIM`, `ENTITY_DIM`) passed to MLPs increase accordingly.
- `create_encoder_weights()` MLP input dimensions: Must match new concatenated dims.
- Checkpoint save/load in `main_model/train.py`: Must include embedding table state dicts.

What stays:
- `build_state_tensors()` — keep float conversion as-is, categorical IDs extracted separately
- Attention block structure — no change to `multi_head_attention.py`
- The `encode_tensors()` fast path that bypasses `build_state_tensors()`
- `EMBED_DIM=64`, `HIDDEN_DIM=128`, `FUSED_DIM=128`, `NUM_HEADS=4`

Input dimension changes to MLPs if concatenation approach is used:

| MLP | Old input dim | New input dim (CAT_EMBED_DIM=8) |
|-----|--------------|--------------------------------|
| `ego_mlp` | 46 | 46 + 8 (v_class) = 54 |
| `scene_mlp` | 16 | 16 + 8 (weather) = 24 |
| `entity_mlp` | 24 | 24 + 8 (type_id) + 8 (bucket_id) = 40 |

This is a breaking change to MLP input sizes — existing checkpoints are incompatible. That is acceptable for v1.1 since the goal is to retrain from scratch with improved representations.

---

**`main_model/train.py`** — Training loop refactor

What changes:
- Load data: Replace single `preprocessed.pt` load with `load_tensor_sessions(data_dir)` that concatenates multiple `session_*.pt` files (or still load `preprocessed.pt` if preprocess step was run).
- Training loop: Replace inner Python loop with `TensorDataset` + `DataLoader`. One `to(device)` call per batch, not per record.
- Forward pass: Full batch `[B, ...]` through `encode_tensors()` and `intuition_head()`.
- AMP: Wrap forward in `torch.autocast(device_type="cuda", dtype=torch.float16)`.
- Gradient scaler: Use `GradScaler` from `training_utils.make_scaler(device)`.
- Gradient clipping order: Must be after `scaler.unscale_(optimizer)`.
- Embedding snapshot: `z_t` is now `[B, 128]` — take one sample for the JSONL snapshot.

What stays:
- Convergence detection (`ConvergenceDetector`) — identical
- Checkpoint save/load structure — same files, same discipline (add embedding table keys)
- Dashboard WebSocket receiver — identical
- `training_status.json` updates — identical

---

**`reward_head/train.py`** — Same changes as `main_model/train.py`

Additional consideration: `z_parent` and `z_child` must remain `.detach()`ed (frozen encoder). With true batching, encode the full batch then call `.detach()` on the batch tensor — same logic, just applied to `[B, 128]` instead of `[1, 128]`.

---

**`action_planner/train.py`** — Same batching + AMP changes, but requires preprocessed data first

Currently streams JSONL only. After `preprocess_action_planner()` is added to `preprocess_data.py`, this trainer can use the same `TensorDataset` + `DataLoader` pattern. The frozen encoder+intuition head outputs `z_t` and `z_next_pred` from the precomputed tensors — no raw state parsing at training time.

---

**`training_utils.py`** — Small additive changes only

Add:
- `make_scaler(device)`: Returns `torch.cuda.amp.GradScaler()` on CUDA, `None` on CPU. Callers check `if scaler is not None`.
- `load_tensor_sessions(data_dir, prefix="session_")`: Loads all `session_*.pt` files from a directory, concatenates tensors with `torch.cat`, returns merged dict. This replaces the single `preprocessed.pt` dependency.

Keep everything else unchanged.

---

**`training_config.json`** — Config only

Update each module's batch_size. Add AMP flag. No code changes required.

```json
{
    "encoder_intuition": {
        "lr": 3e-4,
        "batch_size": 256,
        "amp": true,
        ...
    },
    "reward_head": {
        "lr": 3e-4,
        "batch_size": 256,
        "amp": true,
        ...
    },
    "action_planner": {
        "lr": 3e-4,
        "batch_size": 128,
        "amp": true,
        ...
    }
}
```

---

### New Files Required

**`main_model/embedding_tables.py`**

Purpose: Centralize categorical vocabulary sizes and `nn.Embedding` creation. Keep this separate from `main_model.py` so it can be imported by `preprocess_data.py` and `train.py` without pulling in the full encoder.

Minimal content:
- Vocabulary size constants (`WEATHER_VOCAB`, `V_CLASS_VOCAB`, `TYPE_ID_VOCAB`, `BUCKET_ID_VOCAB`)
- `CAT_EMBED_DIM = 8` constant
- `create_embedding_tables()` → dict of `nn.Embedding`
- `save_embedding_tables(tables, path)` / `load_embedding_tables(path)` — matching existing checkpoint discipline (`weights_only=True`)

Vocab sizes must be determined by scanning real capture data before setting constants. GTA V has 23 vehicle classes (indices 0-22), ~13 weather types, and entity type/bucket IDs that must be confirmed from real sessions.

No other new files are required. All other changes are modifications to existing files.

---

## Project Structure After v1.1

```
rash_kog/
├── capture_states.py           # MODIFIED: write session_*.pt, buffer-then-flush
├── preprocess_data.py          # MODIFIED: add categorical IDs + action_planner path
├── training_utils.py           # MODIFIED: add make_scaler(), load_tensor_sessions()
├── training_config.json        # MODIFIED: batch_size=256, amp=true
│
├── main_model/
│   ├── main_model.py           # MODIFIED: categorical embeddings in encoder
│   ├── embedding_tables.py     # NEW: nn.Embedding definitions, vocab sizes
│   ├── multi_head_attention.py # NO CHANGE
│   └── train.py                # MODIFIED: DataLoader + AMP + true batch
│
├── reward_head/
│   ├── reward_head.py          # NO CHANGE
│   └── train.py                # MODIFIED: DataLoader + AMP + true batch
│
├── action_planner/
│   ├── action_planner.py       # NO CHANGE (verify batch dim works)
│   └── train.py                # MODIFIED: DataLoader + AMP + true batch
│
└── metacontroller/             # NO CHANGE (online RL, out of scope v1.1)
```

---

## VRAM Budget

RTX 3070 Ti, 8 GB. GTA V is NOT running during offline training — full VRAM available.

| Component | VRAM estimate | Basis |
|-----------|---------------|-------|
| Encoder MLPs (ego, scene, route, entity, fusion) | ~15 MB | ~375k params × 4 bytes |
| Encoder attention weights (2 blocks × 4 matrices) | ~5 MB | 4 × embed_dim² × num_heads |
| Embedding tables (4 tables, vocab ≤32, dim=8) | < 1 MB | negligible |
| Intuition head (token_embed 874×32 + MLP) | ~2 MB | |
| Reward head (reward_mlp + rf_predictor) | ~1 MB | |
| Action planner (planner_mlp) | ~1 MB | |
| Batch tensors B=256, encoder training | ~55 MB | ego[256,46]+scene[256,16]+route[256,14]+entities[256,32,24]+mask[256,32] × 2 (t and t1) |
| Batch tensors B=256, reward training | ~35 MB | before+after state pairs + rf tensors |
| Activations + gradients (fp16) | ~250 MB | rough estimate, fp16 halves activation memory |
| Optimizer state (Adam: 2 momentum buffers) | ~60 MB | 2 × param_count × 4 bytes |
| **Total peak (encoder training)** | **~390 MB** | Well within 8 GB |

B=512 is feasible if needed. B=256 already saturates RTX 3070 Ti Tensor Cores for these layer sizes — going larger gives diminishing throughput returns and does not improve convergence speed per epoch.

If GTA runs simultaneously (e.g., data collection + training interleaved), cap at B=64 to leave VRAM headroom for GTA's GPU usage.

---

## Suggested Build Order

Dependencies drive this order. Each step can be validated independently before the next starts.

**Step 1: `main_model/embedding_tables.py` (new file)**

No dependencies. Scan real capture data to determine vocabulary sizes. Define constants and `create_embedding_tables()`. Write `save_embedding_tables()` / `load_embedding_tables()`. Run unit test: create tables, forward pass a batch of integer IDs, verify output shape.

Confidence gate: vocab size constants confirmed from data before proceeding.

**Step 2: `main_model/main_model.py` — add categorical embedding support**

Depends on Step 1. Modify `create_encoder_weights()` to include embedding tables. Modify `encode_state()` and `encode_tensors()` to perform embedding lookups and concatenate. Update MLP input dimensions (`EGO_DIM` fed to `ego_mlp` becomes 54, etc.).

Validation: `python main_model/main_model.py` smoke test must still pass. `z_t` shape must still be `[1, 128]`.

**Step 3: `preprocess_data.py` — add categorical IDs and action_planner path**

Depends on Step 2 (uses `build_state_tensors` to understand field extraction). Add categorical integer ID extraction to existing `preprocess_main_model()` and `preprocess_reward_head()`. Add `preprocess_action_planner()`. Run on a small JSONL subset to validate output. Run on full 88 GB dataset — this will take significant wall time.

Validation: `torch.load(preprocessed.pt)` on output, verify all expected keys present, shapes correct.

**Step 4: `training_utils.py` — add `make_scaler()` and `load_tensor_sessions()`**

No dependency on Steps 1-3 (pure additions). Write and test in isolation.

**Step 5: `main_model/train.py` — DataLoader + AMP refactor**

Depends on Steps 2 (embedding tables in encoder), 3 (preprocessed data with cat IDs), 4 (scaler helper). Replace inner loop with DataLoader. Add autocast. Fix gradient clipping order. Update checkpoint save to include embedding table state dicts.

Validation: Run on small slice of data. Verify loss curve matches pre-refactor behavior (within noise) over first 100 steps. Profile GPU utilization — should see near-100% with B=256.

**Step 6: `reward_head/train.py` — same changes**

Depends on Step 5 (pattern validated). Same structure as Step 5.

**Step 7: `action_planner/train.py` — same changes**

Depends on Steps 3 (preprocessed data for action_planner) and 5-6 (frozen encoder checkpoint with new format).

**Step 8: `capture_states.py` — compact capture format**

No training dependency — can be done at any point. Only affects future captures. Does not block Steps 1-7. Do last so training pipeline is validated before committing to the new capture format.

---

## Critical Integration Points

### 1. Embedding Table in Checkpoint

The encoder checkpoint (`main_model/checkpoints/session_*/encoder_weights.pt`) currently saves state dicts for `ego_mlp`, `scene_mlp`, `route_mlp`, `entity_mlp`, `fusion_mlp`, and attention weights. After Step 2, it must also save embedding table state dicts.

In `save_training_checkpoint()`:
```python
# Add to encoder_state dict:
for key, val in encoder_weights.get("cat_embeds", {}).items():
    encoder_state[f"cat_embeds.{key}"] = val.state_dict()
```

In `load_training_checkpoint()`:
```python
# Restore embedding tables:
for key, sd in encoder_state.items():
    if key.startswith("cat_embeds."):
        embed_key = key[len("cat_embeds."):]
        encoder_weights["cat_embeds"][embed_key].load_state_dict(sd)
```

The reward head and action planner trainers load encoder checkpoints — they must also be updated to restore embedding tables.

### 2. `freeze_module()` with Embedding Tables

The existing `freeze_module()` in `training_utils.py`:
```python
if isinstance(module, dict):
    for key, value in module.items():
        if hasattr(value, "parameters"):
            value.requires_grad_(False)
```

`nn.Embedding` has `.parameters()` but the check is `hasattr(value, "parameters")` — this checks for the attribute, not a call. `nn.Embedding` does have this attribute. The freeze will work correctly.

However, `encoder_weights["cat_embeds"]` is a nested dict inside the encoder_weights dict. `freeze_module(encoder_weights)` will see `value = encoder_weights["cat_embeds"]` which is a dict, not an `nn.Module` — it does not have `.parameters`. The embedding tables inside that nested dict will NOT be frozen.

Fix: Either flatten embedding tables into `encoder_weights` at top level (e.g., `encoder_weights["weather_embed"]`, `encoder_weights["v_class_embed"]`), or update `freeze_module()` to recurse into nested dicts. Flattening is the simpler fix and consistent with how `ego_mlp`, `scene_mlp`, etc. are stored.

### 3. Mixed Precision and Gradient Clipping Order

The current trainers call `clip_grad_norm_` after `loss.backward()`. With AMP, scaled gradients exist after `scaler.scale(loss).backward()`. Clipping at this point clips the wrong magnitude.

Correct order (must be enforced in all three trainers):
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)          # must come before clip_grad_norm_
clip_grad_norm_(all_params, max_grad_norm)
scaler.step(optimizer)
scaler.update()
```

### 4. Action Planner Vocabulary Size Dependency

The action planner forward pass takes `vocab_size=874`. With batched training, `vocab_size` remains a fixed constant — no change needed. The planner still produces `logits [B, vocab_size]` and the DataLoader batches `target_token_ids [B]` as `torch.long`.

Verify `action_planner.py` handles `[B, 128]` input for `z_t` and `z_next_pred` — the MLP inside is a `nn.Sequential` of `nn.Linear` layers which are batch-dimension agnostic by construction.

### 5. `build_state_tensors()` Batch Dimension

Currently `build_state_tensors(raw_state)` returns tensors with shape `[1, ...]` (`.unsqueeze(0)` inside). During preprocessing, this is called once per record and the `[1,...]` tensors are concatenated into `[N,...]`.

After preprocessing, `DataLoader` yields `[B, ...]` batches. The encoder receives `[B, ...]` inputs directly — `build_state_tensors` is NOT called during training. This is already the case in the current `encode_tensors()` fast path. No change needed to `build_state_tensors()`.

---

## Anti-Patterns to Avoid in v1.1

### Anti-Pattern 1: AMP Inside the Python Accumulation Loop

Apply `torch.autocast` around the full DataLoader iteration, not inside the inner record loop. A loop of B `[1, ...]` autocast calls is still B separate kernel launches — no Tensor Core benefit.

### Anti-Pattern 2: Staging All Data on VRAM

Do not call `pt_data.to(device)` after loading. Keep all data in CPU RAM. Use `pin_memory=True` on DataLoader and `batch.to(device, non_blocking=True)` inside the loop. Only one batch lives on GPU at a time (~55 MB vs ~800 MB total).

### Anti-Pattern 3: Clipping Gradients Before `scaler.unscale_()`

Always call `scaler.unscale_(optimizer)` before `clip_grad_norm_`. Scaled gradients are ~32768x larger than real gradients — clipping at `max_grad_norm=0.5` on scaled gradients does nothing.

### Anti-Pattern 4: Hardcoding Categorical Vocabulary Sizes

Scan real GTA capture data before setting `WEATHER_VOCAB`, `V_CLASS_VOCAB`, etc. GTA can return edge values (e.g., `v_class=255` for an unknown vehicle). Add `.clamp(0, VOCAB_SIZE - 1)` before every embedding lookup. An out-of-bounds index raises a silent wrong result in some PyTorch versions and an exception in others.

### Anti-Pattern 5: Embedding Tables Not in Checkpoint

If `create_embedding_tables()` returns new tables but `save_training_checkpoint()` does not save them, every resume starts with randomly initialized categorical embeddings. The encoder checkpoint load must include embedding tables or the resumed model produces garbage for categorical fields.

### Anti-Pattern 6: Incompatible Checkpoint After Input Dim Change

Increasing MLP input dimensions (ego_mlp: 46 → 54) invalidates existing `encoder_weights.pt` checkpoints. Do not attempt to load old checkpoints into the new encoder. v1.1 training starts from scratch. Document this clearly.

---

## Sources

All findings are from direct reading of the current codebase. No external sources consulted — this is an integration analysis, not ecosystem research.

Key files read:
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/main_model/main_model.py` — encoder architecture, field definitions, EGO_DIM/SCENE_DIM/ENTITY_DIM constants, `encode_tensors()` fast path
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/main_model/train.py` — current training loop structure, pseudo-batch pattern, preprocessed.pt dual path
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/reward_head/train.py` — Stage 2 training loop
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/action_planner/train.py` — Stage 3 training loop, JSONL-only (no preprocessed path)
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/capture_states.py` — current JSONL write path, buffer-less per-frame flush
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/preprocess_data.py` — existing preprocessing confirming .pt format already designed
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/training_utils.py` — `freeze_module` implementation, `StreamingJSONLDataset`
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/training_config.json` — current batch sizes (8 for encoder/reward, 16 for planner)
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/reward_head/reward_head.py` — `RF_DIM=6`, `extract_reward_features` signature
- `/mnt/c/Users/laksh/OneDrive/Documents/rash_kog/.planning/PROJECT.md` — v1.1 goals, RTX 3070 Ti 8 GB constraint

---
*Architecture research for: RASHKOGIE GTA v1.1 Training Optimization*
*Researched: 2026-05-04*
