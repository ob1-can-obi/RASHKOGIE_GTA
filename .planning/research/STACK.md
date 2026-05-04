# Technology Stack

**Project:** RASHKOGIE GTA — v1.1 Training Optimization
**Researched:** 2026-05-04
**Context:** Subsequent milestone — adding compact tensor format, learned embeddings, batched GPU forward passes, and CUDA mixed precision to an existing PyTorch codebase. All core ML infrastructure from v1.0 remains unchanged.
**Confidence:** HIGH

---

## What This Document Covers

Only the **stack additions and changes** needed for v1.1. Do not re-research the existing stack (PyTorch, FastAPI, SQLite, websockets — all locked from v1.0).

---

## Critical First Step: Install the CUDA Build

The `.venv` currently has `torch 2.11.0+cpu`. The RTX 3070 Ti is only accessible from the Windows-native Python environment, not from WSL2. `torch.cuda.is_available()` returns `False` in WSL2 regardless of torch version.

**Action required:** All training scripts must run via the Windows-native Python installation (not WSL2). The `requirements.txt` already specifies `--extra-index-url https://download.pytorch.org/whl/cu124`, so reinstalling in the Windows Python venv will pull the CUDA build.

```bash
# Run from Windows cmd/PowerShell (not WSL2), inside the project venv:
pip install "torch>=2.11.0" --index-url https://download.pytorch.org/whl/cu124
```

Verify with: `python -c "import torch; print(torch.cuda.is_available())"` — must print `True`.

**Confidence:** HIGH — cu124 index URL is already in requirements.txt. RTX 3070 Ti is Ampere (sm_86), fully supported by CUDA 12.4+.

---

## Core Technologies — New for v1.1

### 1. Mixed Precision Training: `torch.amp`

**Use:** `torch.amp.autocast` + `torch.amp.GradScaler`

Both are part of PyTorch core (no new package). The older `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler` APIs are deprecated as of PyTorch 2.x and will produce warnings. Use the unified API.

```python
# Correct API (PyTorch 2.x):
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
    z_t = encode_tensors(tensors_t, encoder_weights)
    z_next_pred, _, token_embed, intuition_mlp = intuition_head(...)
    loss = F.mse_loss(z_next_pred, z_t1_real)

scaler.scale(loss).backward()

# Must unscale before gradient clipping:
scaler.unscale_(optimizer)
clip_grad_norm_(all_params, max_grad_norm)

scaler.step(optimizer)
scaler.update()
```

**Why fp16 not bf16:** RTX 3070 Ti (Ampere sm_86) supports bf16 compute, but fp16 has higher throughput on consumer Ampere cards and is the standard for non-LLM training. bf16 eliminates the need for GradScaler (no underflow risk) but its benefit is mainly for transformers with large dynamic range. Our MLP/attention models with MSE and cross-entropy losses are stable in fp16 with GradScaler.

**VRAM impact:** fp16 halves activation memory. For our model (encoder ~500K params + intuition head ~200K params), fp16 saves roughly 2-3 GB activation VRAM at batch_size=128, making the 8 GB VRAM budget comfortable.

**Batch size constraint:** Batch size must be a multiple of 8 to use Tensor Cores efficiently. Use 64 or 128 for the flat training datasets (236K records each fitting in RAM after preprocessing).

**GradScaler checkpointing:** Save and restore scaler state alongside optimizer state:

```python
torch.save({"scaler_state": scaler.state_dict(), ...}, ckpt_path)
scaler.load_state_dict(ckpt["scaler_state"])
```

**Confidence:** HIGH — verified against PyTorch 2.11 AMP docs and confirmed deprecation of `torch.cuda.amp` APIs.

---

### 2. Compact Tensor Storage: `torch.save` with Single `.pt` File

**Use:** `torch.save(dict_of_tensors, path)` — already implemented in `preprocess_data.py`.

The existing `preprocess_data.py` already writes a single `.pt` file with stacked tensors (`ego_t [N,46]`, `scene_t [N,16]`, etc.). This is the correct approach.

**Changes needed for v1.1:**
- Add integer index tensors for categorical fields (`v_class_idx`, `weather_idx`, `entity_type_id`, `entity_bucket_id`) as `torch.long` rather than encoding them as floats inside the float tensors
- Separate the categorical index columns out of `ego` and `scene` tensors during preprocessing so the encoder can route them to `nn.Embedding` layers

**Inline capture format** (for new sessions, replacing JSONL write):

```python
# Write directly at capture time instead of raw JSONL:
torch.save({
    "ego_t": ego_tensor,        # [46] float32
    "scene_t": scene_tensor,    # [16] float32
    "route_t": route_tensor,    # [14] float32
    "entities_t": ent_tensor,   # [32, 24] float32
    "mask_t": mask_tensor,      # [32] float32
    "v_class_idx": ...,         # [] int64
    "weather_idx": ...,         # [] int64
    "token_id": ...,            # [] int64
}, f"frame_{idx:07d}.pt")
```

Alternatively, accumulate frames in-memory per session and flush a single `.pt` shard every N=10,000 frames. Single-shard approach avoids per-frame file I/O overhead and is simpler to load.

**Memory-mapped loading** (`mmap=True`): Use only if the preprocessed `.pt` file exceeds available CPU RAM. For ~236K records at ~3.4 KB each, preprocessed size is ~800 MB — fits in RAM on any modern PC. Load normally with `torch.load(path, weights_only=True, map_location="cpu")`.

Reserve `mmap=True` for future sessions where data exceeds 4 GB RAM.

**Confidence:** HIGH — `torch.save`/`torch.load` is the standard and already in use; sizing is derived from PROJECT.md figures.

---

### 3. Learned Embeddings: `nn.Embedding`

**Use:** `torch.nn.Embedding` — PyTorch core, no new package.

Categorical fields currently treated as continuous floats (meaningless to MLP dot products):
- `weather` — small integer (0–10 typical weather codes)
- `v_class` — vehicle class integer (0–23 in GTA)
- `v_model` — vehicle model hash (high-cardinality, may need hashing trick)
- Entity `type_id` — entity type integer
- Entity `bucket_id` — distance bucket integer

**Recommended embedding dims** (rule of thumb: `min(50, (n_categories + 1) // 2)`):

| Field | Cardinality | Embed Dim | Notes |
|-------|-------------|-----------|-------|
| `weather` | ~12 | 4 | Low cardinality; 4 dims sufficient |
| `v_class` | 24 | 8 | GTA has 22 vehicle classes |
| `v_model` | ~100-400 | 16 | Use hash trick if >500 unique models |
| `entity type_id` | ~5 | 4 | Ped/vehicle/object/bike/etc. |
| `entity bucket_id` | ~8 | 4 | Distance buckets |

**Integration pattern:** Add embedding tables to `create_encoder_weights()` and concatenate their outputs to the corresponding MLP inputs.

```python
# In create_encoder_weights():
weather_embed = nn.Embedding(num_embeddings=16, embedding_dim=4)
v_class_embed = nn.Embedding(num_embeddings=32, embedding_dim=8)

# In encode_tensors(), before ego_mlp:
weather_emb = weather_embed(weather_idx)   # [B, 4]
v_class_emb = v_class_embed(v_class_idx)  # [B, 8]
ego_cat = torch.cat([ego_continuous, weather_emb, v_class_emb], dim=-1)
ego_emb = ego_mlp(ego_cat)               # input dim grows by 12
```

This means `ego_mlp` input dim changes from 46 to `46 - (removed_cat_fields) + sum(embed_dims)`. Remove the float versions of categorical fields from the continuous tensors to avoid redundancy.

**Confidence:** HIGH — `nn.Embedding` for categorical tabular features is established PyTorch pattern; cardinalities estimated from GTA domain knowledge.

---

### 4. Batched DataLoader: `torch.utils.data.TensorDataset` + `DataLoader`

**Use:** `torch.utils.data.TensorDataset` + `torch.utils.data.DataLoader` — PyTorch core, no new package.

The current training loops iterate record-by-record in Python (one `.unsqueeze(0)` call per record). Replace with native DataLoader batching.

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(
    pt_data["ego_t"],       # [N, 46]
    pt_data["scene_t"],     # [N, 16]
    pt_data["route_t"],     # [N, 14]
    pt_data["entities_t"],  # [N, 32, 24]
    pt_data["mask_t"],      # [N, 32]
    pt_data["token_ids"],   # [N]
)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    pin_memory=True,         # async CPU→GPU transfer
    num_workers=0,           # Windows: keep at 0 to avoid spawn issues
    drop_last=True,          # stable batch size for Tensor Cores
)

for batch in loader:
    ego, scene, route, entities, mask, token_ids = [x.to(device, non_blocking=True) for x in batch]
```

**Windows constraint on `num_workers`:** PyTorch multiprocessing uses `spawn` on Windows, which requires all top-level code to be guarded by `if __name__ == '__main__'`. Since training scripts may be imported as modules, `num_workers=0` is the safe default. The training bottleneck is the GPU forward pass, not data loading from an in-RAM TensorDataset, so `num_workers=0` costs nothing here.

**`pin_memory=True`:** Enables async DMA transfers from CPU to GPU when data is not already on GPU. Safe and beneficial with CUDA. Do not use with CPU-only training.

**`drop_last=True`:** Drops the final incomplete batch. Keeps batch size uniform, which matters for Tensor Core alignment (must be multiple of 8).

**Batch size recommendation:** 128 for encoder/intuition training (~800 MB dataset, 236K records). 64 for reward head training (same dataset size). Both fit comfortably in 8 GB VRAM for these model sizes.

**Confidence:** HIGH — TensorDataset + DataLoader is the standard batched training pattern in PyTorch; Windows num_workers caveat is well-documented.

---

## Supporting Libraries — No Changes Needed

| Library | Version | Status | Notes |
|---------|---------|--------|-------|
| `torch` | 2.11.0+cu124 | Upgrade from +cpu | Switch to CUDA build; all v1.1 features are in core torch |
| `numpy` | >=1.24.0 | Keep | Used in preprocessing; do not introduce new numpy dependencies in training loops |
| `scikit-learn` | >=1.3.0 | Keep | PCA for dashboard embeddings; no changes needed |

No new packages are required for v1.1. All four feature areas (AMP, compact tensors, learned embeddings, batched DataLoader) are served by existing PyTorch core APIs.

---

## What NOT to Add

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `torch.cuda.amp.GradScaler()` | Deprecated in PyTorch 2.x, produces warnings | `torch.amp.GradScaler("cuda")` |
| `torch.cuda.amp.autocast()` | Deprecated in PyTorch 2.x | `torch.amp.autocast(device_type="cuda")` |
| `tensordict` / `torchdata` | Adds heavy dependencies for problems already solved by TensorDataset | `torch.utils.data.TensorDataset` |
| `num_workers > 0` on Windows | Causes EOFError with spawn multiprocessing in imported modules | `num_workers=0` (no perf cost for in-RAM datasets) |
| `mmap=True` on torch.load | Adds disk-seek latency per access during training; only justified if data doesn't fit in RAM | Normal `torch.load()` for <4 GB datasets |
| `torch.float16` for optimizer states | Optimizer states (Adam moments) must stay fp32 even in mixed precision | Keep optimizer in fp32; only forward pass uses fp16 |
| Gradient checkpointing | Trades VRAM for recomputation; unnecessary for 500K-param models on 8 GB VRAM | Not needed at this model scale |
| bf16 | Slightly less throughput than fp16 on consumer Ampere; eliminates GradScaler need but adds complexity | fp16 + GradScaler is the standard for this hardware |

---

## VRAM Budget on RTX 3070 Ti (8 GB)

Approximate VRAM usage at batch_size=128 with fp16:

| Component | fp32 (before) | fp16 (after) |
|-----------|---------------|--------------|
| Model parameters (~1M params) | ~4 MB | ~2 MB |
| Activations at batch_size=128 | ~500 MB | ~250 MB |
| Gradient tensors | ~4 MB | ~4 MB (kept fp32) |
| Adam optimizer states (2 moments) | ~8 MB | ~8 MB (kept fp32) |
| GTA V (separate process) | ~2-3 GB | ~2-3 GB |
| **Available for training** | ~5 GB | ~5 GB |
| **Training headroom** | Comfortable | Comfortable |

These are small MLP + attention models (~1M total parameters). VRAM is not a constraint at this scale. The benefit of fp16 is throughput from Tensor Cores, not VRAM relief.

**Recommended batch sizes:**
- Encoder + intuition head training: 128
- Reward head training: 128
- Action planner training: 128
- All must be multiples of 8 for Tensor Core alignment

---

## Full Installation

```bash
# Switch from CPU to CUDA build (Windows cmd/PowerShell, inside venv):
pip install "torch>=2.11.0" --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

No other requirements.txt changes needed for v1.1.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Mixed precision | `torch.amp.autocast` + `GradScaler` | bf16 (no scaler) | fp16 higher throughput on consumer Ampere; bf16 not worth the change in training patterns |
| Compact tensors | `torch.save` single `.pt` file | Per-frame `.pt` files, HDF5, parquet | Single file has no per-file overhead; HDF5/parquet adds heavy deps (h5py, pyarrow) for no benefit |
| Categorical encoding | `nn.Embedding` | One-hot vectors | One-hot for 400-way v_model is 400x more memory; embeddings are differentiable and learn semantic relationships |
| Batched loading | `TensorDataset` + `DataLoader` | Manual index slicing (current) | DataLoader handles shuffle, pin_memory, and drop_last automatically; 10x simpler |
| Inline capture format | Accumulated shard `.pt` file | Per-frame `.pt` file | Per-frame creates thousands of small files; shard approach writes one file per N frames |

---

## Version Compatibility

| Component | Version | Compatible With | Notes |
|-----------|---------|-----------------|-------|
| `torch` 2.11.0+cu124 | CUDA 12.4 | RTX 3070 Ti (sm_86) | Ampere architecture, full Tensor Core support |
| `torch.amp.autocast` | PyTorch 2.0+ | `torch.amp.GradScaler` | Unified API; replaces deprecated `torch.cuda.amp` namespace |
| `nn.Embedding` | All PyTorch versions | `torch.amp.autocast` | Embedding lookup is a supported autocast op; no casting issues |
| `TensorDataset` + `DataLoader` | All PyTorch versions | `pin_memory=True` | pin_memory requires CUDA to be available; guard with `if torch.cuda.is_available()` |

---

## Sources

- PyTorch AMP documentation (2.11): [https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- `torch.amp.GradScaler` unified API deprecation: [https://github.com/flairNLP/flair/pull/3682](https://github.com/flairNLP/flair/pull/3682) — confirmed `torch.cuda.amp.GradScaler` is deprecated
- PyTorch AMP stable docs: [https://docs.pytorch.org/docs/stable/amp](https://docs.pytorch.org/docs/stable/amp)
- DataLoader best practices (2025-2026): [https://www.progressiverobot.com/2026/02/04/pytorch-dataloader-tutorial/](https://www.progressiverobot.com/2026/02/04/pytorch-dataloader-tutorial/)
- pin_memory explanation: [https://medium.com/data-scientists-diary/when-to-set-pin-memory-to-true-in-pytorch-75141c0f598d](https://medium.com/data-scientists-diary/when-to-set-pin-memory-to-true-in-pytorch-75141c0f598d)
- Windows num_workers multiprocessing: [https://iamholumeedey007.medium.com/pytorch-windows-eoferror-ran-out-of-input-when-num-workers-0-4d372157512](https://iamholumeedey007.medium.com/pytorch-windows-eoferror-ran-out-of-input-when-num-workers-0-4d372157512)
- torch.load mmap=True behavior: [https://discuss.pytorch.org/t/torch-load-should-i-just-always-use-mmap/191305](https://discuss.pytorch.org/t/torch-load-should-i-just-always-use-mmap/191305)
- PyTorch CUDA cu124/cu126 install (Windows): [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)

---

*Stack research for: RASHKOGIE GTA v1.1 Training Optimization*
*Researched: 2026-05-04*
