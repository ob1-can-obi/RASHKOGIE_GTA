# Feature Research

**Domain:** Training optimization for autonomous driving RL agent (v1.1 milestone)
**Researched:** 2026-05-04
**Confidence:** HIGH (PyTorch APIs verified against official docs; hardware constraints from RTX 3070 Ti Ampere architecture specs; codebase audited directly)

---

## Context: What Already Exists

The project has a working training pipeline. Understanding the existing state is required to
assess what "adding" each feature actually means.

| Existing Component | Current State |
|-------------------|---------------|
| `preprocess_data.py` | Exists. Converts JSONL to `.pt`. Stores continuous float tensors. Does NOT extract categorical indices |
| `main_model/train.py` | Branches on `preprocessed.pt` if found, falls back to streaming JSONL. Still loops per-record inside each batch iteration |
| `encode_tensors()` in `main_model.py` | Exists. Accepts pre-built tensor dicts. Called with `unsqueeze(0)` on single records — not truly batched |
| Categorical fields | `weather`, `v_class` in ego; `v_model` implied; `type_id`, `bucket_id` in entity fields. All read via `_to_float()` and stored as continuous floats — semantically meaningless to MLPs |
| Mixed precision | Not implemented. All training is fp32 |
| `capture_states.py` | Writes full raw GTA state dicts as JSONL including unused `near_vehs`, `near_peds`, `near_objects` lists — source of 74x bloat |
| 88 GB JSONL data | 6 sessions, ~236k records. ~800 MB of useful numeric content once converted |

---

## Feature Landscape

### Table Stakes (Required for v1.1 Goal)

Missing any of these means training either cannot run on 8 GB VRAM or runs at CPU speed with
no meaningful GPU utilization.

| Feature | Why Required | Complexity | Notes |
|---------|--------------|------------|-------|
| True batched encoder forward pass | Current per-record loop passes `[1, dim]` tensors one at a time. GPU is idle between iterations. Batching stacks to `[B, dim]` and processes in a single kernel call — the fundamental switch from CPU-paced to GPU-paced training | MEDIUM | `encode_tensors()` already accepts tensor dicts. The fix is in the training loop: slice `pt_data["ego_t"][batch_indices]` directly (shape `[B, 46]`) and pass to encoder once. Verify attention layers handle `[B, 32, 64]` entity tensors correctly |
| `torch.utils.data.DataLoader` integration | Replaces manual Python `for idx in batch_indices` inner loops. Provides automatic prefetch, pin_memory for fast CPU-to-GPU transfers, and shuffle per epoch. Without this, GPU is data-starved between steps | LOW | Standard PyTorch `TensorDataset` + `DataLoader`. `num_workers=2` on Windows/WSL (avoid fork issues). `pin_memory=True` when training on CUDA |
| Compact `.pt` tensor format for all training data | 88 GB JSONL cannot be held in RAM; 800 MB preprocessed `.pt` loads in ~3 seconds. Batching from a memory-resident tensor is O(1) per record; streaming JSONL is O(seek). The tool exists — needs categorical index extraction added | LOW | `preprocess_data.py` exists and produces `.pt`. `train.py` already branches on `preprocessed.pt`. Gap: categorical fields need separate `torch.long` index tensors |
| CUDA AMP mixed precision (`autocast` + `GradScaler`) | fp16 activations halve VRAM footprint of the forward pass. The entity attention block produces `[B, 32, 64]` tensors — at batch=128 in fp32 that is 128 x 32 x 64 x 4 = 100 MB just for that one activation. fp16 halves it. RTX 3070 Ti Ampere has 3rd-gen Tensor Cores that accelerate fp16 matmul | MEDIUM | `torch.cuda.amp.autocast()` wraps forward + loss. `GradScaler` wraps backward. `clip_grad_norm_` must happen after `scaler.unscale_()`. Apply to all three trainers: `main_model/train.py`, `reward_head/train.py`, `action_planner/train.py` |
| CPU-to-GPU transfer minimization | Current code calls `.to(device)` per-record in a Python loop. Each call has CUDA synchronization overhead. With DataLoader `pin_memory=True`, one `.to(device, non_blocking=True)` per batch replaces hundreds of individual transfers | LOW | Achieved automatically once DataLoader with `pin_memory=True` is in place |

---

### Differentiators (Beneficial — Improve Training Quality)

Features that improve model correctness or long-term training efficiency. All are in-scope for
v1.1 but are not VRAM blockers on their own.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Learned embeddings for categorical fields | `weather`, `v_class`, `v_model`, entity `type_id`, entity `bucket_id` are nominal categories. Treating category 3 and category 7 as continuous values implies a false ordinal relationship. `nn.Embedding` learns a dense vector per category that captures real semantic similarity via gradient descent | MEDIUM | Five embedding tables. Suggested dims: weather 4, v_class 8, v_model 8 (cap at 200 values), type_id 4, bucket_id 4. Output concatenated into corresponding feature vectors before MLP. Requires `preprocessing.py` to extract integer indices, and `capture_states.py` to write indices at capture time |
| Updated `preprocess_data.py` with categorical index extraction | The current preprocessor stores all fields as `float32` via `_to_float()`. Embedding lookup requires `torch.long` integer indices stored separately. This is the prerequisite for learned embeddings | LOW | Add fields: `weather_idx`, `v_class_idx`, `v_model_idx`, `entity_type_ids`, `entity_bucket_ids` alongside existing float tensors in the `.pt` output |
| Inline compact capture (write tensors directly, not JSONL) | New captures write pre-built tensor dicts directly, eliminating the preprocessing step for future sessions. Capture overhead is the same `build_state_tensors()` call that preprocessing already does | MEDIUM | Risk: raw tensors are harder to inspect than JSON. Mitigation: keep a human-readable session header and write a small sidecar JSON with session metadata. The 88 GB of existing JSONL still needs offline preprocessing regardless |
| Preprocessing tool for existing 88 GB JSONL | One-shot conversion of all historical data including the new categorical index fields | LOW | `preprocess_data.py` is 90% of the way there. Add categorical extraction and the tool is complete |

---

### Anti-Features (Avoid)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| BF16 instead of FP16 | BF16 is numerically more stable (wider exponent range, fewer NaN risks) | RTX 3070 Ti (Ampere) does NOT have hardware-accelerated BF16 matmul. BF16 on Ampere runs at FP32 speed — no throughput gain, no VRAM savings. PyTorch AMP defaults to fp16 on Ampere for this reason | Use fp16 + GradScaler. Reserve BF16 for H100/RTX 4090+ |
| Memory-mapped `.pt` (mmap) | Sounds efficient for large files | At 236k records, the full preprocessed `.pt` is ~800 MB. This fits easily in system RAM (most PCs have 16-32 GB). mmap adds seek overhead and complexity without benefit. mmap is useful when the dataset cannot fit in RAM | Load full `.pt` with `torch.load()` at training start. 3-second load, then RAM-resident |
| `torch.compile()` on encoder | Promises 10-30% throughput improvement | The encoder uses a dict-based weight API (`encoder_weights["ego_mlp"](x)`) rather than a standard `nn.Module`. `torch.compile()` may fail to trace or produce incorrect graphs through this pattern. Not worth debugging until the dict API is replaced | Profile first after batching + AMP. If still GPU-bound, consider `torch.compile()` after refactoring encoder to `nn.Module` |
| fp8 quantization | More aggressive than fp16, more VRAM savings | fp8 (`torch.float8_e4m3fn`) requires Hopper architecture (H100). Not available on Ampere. PyTorch fp8 will raise a runtime error on RTX 3070 Ti | Stay with fp16 |
| Gradient checkpointing | Reduces activation VRAM by recomputing during backward | The encoder is small. At batch=256 with fp16, all activations fit well within the 5.5 GB training budget. Gradient checkpointing adds recompute overhead that would slow training with no benefit at this scale | Only consider if batch_size > 512 is needed and still OOM |
| Separate DataLoader class per module | Each module has different data schemas | They share the same `.pt` format and loading logic. Duplicating DataLoader code per module creates maintenance burden | One `TensorDataset` wrapper in `training_utils.py` parameterized by which keys to load |
| Per-record `.pt` files (one file per capture record) | Preserves update ability | Random-access across thousands of files has poor OS cache behavior. A single consolidated file loaded into RAM is always faster | Single `.pt` per session (or per all sessions merged), loaded once at training start |

---

## Feature Dependencies

```
[Updated preprocess_data.py — adds categorical indices]
    └──required by──> [Learned categorical embeddings in encoder]
    └──required by──> [Inline capture writing integer indices]

[Compact .pt data in RAM]
    └──required by──> [True batched encoder forward pass]
                          └──required by──> [DataLoader integration]
                                                └──enables──> [AMP mixed precision]
                                                └──enables──> [pin_memory CPU-to-GPU transfers]

[Learned categorical embeddings]
    └──requires change to──> [create_encoder_weights() — add nn.Embedding tables]
    └──requires change to──> [encode_tensors() — look up embeddings, concat with floats]
    └──requires change to──> [build_state_tensors() — or bypass: look up in train loop]

[Inline compact capture]
    └──enhances──> [Compact .pt data] (eliminates preprocessing step for new captures)
    └──independent of──> [Preprocessing tool] (existing 88 GB still needs offline conversion)
    └──requires──> [Updated preprocess_data.py format] (inline format must match .pt schema)
```

### Key Dependency Notes

- **Batching requires `.pt` data:** Slicing `pt_data["ego_t"][batch_indices]` gives a `[B, 46]` tensor in one operation. Without `.pt`, each record requires calling `build_state_tensors()` which is a Python loop over dict fields — the batching benefit vanishes.
- **AMP requires batching:** fp16 Tensor Core speedup only activates when tensor matmul dimensions are multiples of 8 and batch sizes are large enough to saturate GPU warps. At batch=1, fp16 adds type-conversion overhead and is slower than fp32.
- **Embeddings require long index tensors:** `nn.Embedding(num_classes, dim)` takes `torch.long` integer class indices, not `float32` values. The current preprocessor stores everything as float. The preprocessor must be updated before embeddings can be wired in.
- **Inline capture and preprocessing are parallel workstreams:** Inline capture for new sessions and offline preprocessing for existing 88 GB can be developed independently and in any order.

---

## MVP Definition

### Launch With (v1.1 — this milestone)

Minimum required to achieve the milestone goal: batched GPU training on RTX 3070 Ti with mixed precision.

- [ ] **Update `preprocess_data.py`** to extract categorical fields as `torch.long` index tensors (`weather_idx`, `v_class_idx`, `entity_type_ids`, `entity_bucket_ids`) — prerequisite for embeddings and compatible with existing float tensors
- [ ] **Verify and fix batched encoder forward pass** — `encode_tensors()` receives `[B, dim]` tensors; verify attention layers handle `[B, 32, 24]` entity input without shape errors
- [ ] **Add `nn.Embedding` tables** to `create_encoder_weights()` for the 5 categorical fields; update `encode_tensors()` to look up embeddings and concatenate into feature vectors before MLP input
- [ ] **Replace per-record inner loops** in all three trainers (`main_model/train.py`, `reward_head/train.py`, `action_planner/train.py`) with `torch.utils.data.DataLoader` slicing full batches from `.pt` tensors
- [ ] **Add AMP training** via `torch.cuda.amp.autocast()` + `GradScaler` in all three training scripts; unscale before `clip_grad_norm_`, step, update scaler
- [ ] **Update `capture_states.py`** to write compact tensor format instead of full JSONL — call `build_state_tensors()` at capture time and save tensor dict per record (or accumulate and save per session)

### Add After Validation (v1.1.x)

- [ ] **Batch size sweep** — once AMP is working, sweep batch_size from 64 to 256 to find the VRAM ceiling on RTX 3070 Ti with GTA running concurrently; use `torch.cuda.memory_allocated()` to profile
- [ ] **Parallel JSONL sidecar for new captures** — write first 1000 frames as both tensor and JSON to validate capture correctness before removing the JSON path entirely

### Future Consideration (v2+)

- [ ] **`torch.compile()` on encoder** — after encoder is refactored from dict API to `nn.Module`, `torch.compile()` may provide additional 10-30% throughput
- [ ] **Chunked DataLoader for multi-hundred-GB data** — if captures grow beyond RAM capacity, move from RAM-resident `.pt` to chunked mmap; not needed at current 800 MB scale

---

## Feature Prioritization Matrix

| Feature | Training Value | Implementation Cost | VRAM Impact | Priority |
|---------|----------------|---------------------|-------------|----------|
| Batched encoder forward pass | HIGH — core GPU utilization | MEDIUM — verify shape assumptions | None (correctness) | P1 |
| DataLoader integration | HIGH — prefetch, pin_memory | LOW — standard API | Indirect (faster data supply) | P1 |
| Compact .pt + updated preprocessor | HIGH — enables batching | LOW — tool exists, add index fields | None (data format) | P1 |
| AMP fp16 + GradScaler | HIGH — halves activation VRAM | MEDIUM — wrap 3 scripts | -50% activations | P1 |
| Learned categorical embeddings | MEDIUM — model correctness | MEDIUM — update encoder + preprocess | Negligible (<1 MB tables) | P1 |
| Inline compact capture | MEDIUM — convenience for new sessions | MEDIUM — redesign write path | None | P2 |
| Batch size tuning | LOW — sweep after AMP works | LOW — one config value | Direct | P2 |

---

## VRAM Budget Analysis (RTX 3070 Ti, 8 GB Total)

| Consumer | Estimated VRAM | Notes |
|----------|---------------|-------|
| GTA V (concurrent) | 1.5-2 GB | Cannot control; measured empirically |
| OS + CUDA driver + PyTorch runtime | 0.3-0.5 GB | One-time overhead |
| **Available for training** | **5.5-6 GB** | |
| Encoder weights (fp32, pinned in memory) | ~15 MB | Small MLPs + attention projections |
| Embedding tables (5 fields, 4-8 dim each) | <1 MB | Negligible |
| Optimizer Adam state (fp32) | ~30-60 MB | Two copies of all trainable params |
| Forward activations, batch=64, fp32 | ~200-400 MB | Entity attention `[64, 32, 64]` dominates |
| Forward activations, batch=64, fp16 (AMP) | ~100-200 MB | AMP halves this |
| Gradient tensors (same size as activations) | ~100-200 MB | |
| **Estimated total at batch=64, fp16** | **~350-500 MB** | Well within 5.5 GB budget |
| **Estimated maximum safe batch** | **256-512** | Test empirically; model is small |

Key insight: This model is small relative to the VRAM available. The bottleneck is not model size
but Python-loop overhead between GPU calls. Batching eliminates that overhead. AMP is a secondary
win for throughput on Tensor Cores, not a survival requirement.

Batch size multiple-of-8 rule: To activate Tensor Cores on Ampere, batch_size and all matmul
inner dimensions must be multiples of 8. Current hidden dims (64, 128) already satisfy this.
Set batch_size to 64, 128, or 256 — not 100 or 150.

---

## Sources

- [PyTorch AMP documentation](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html) — official autocast + GradScaler API
- [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — Tensor Core alignment requirements (multiples of 8), architecture support matrix
- [PyTorch: What Every User Should Know About Mixed Precision](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/) — fp16 vs bf16 per architecture; BF16 not accelerated on Ampere
- [PyTorch DataLoader tactics](https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8) — pin_memory, num_workers, prefetch guidance
- [PyTorch Embedding for categorical data](https://discuss.pytorch.org/t/nn-embedding-layer-in-categorical-features/148558) — categorical embedding patterns for tabular/structured data
- [RTX 3070 Ti CUDA Guide](https://www.rightnowai.co/guides/gpu-comparison/rtx-3070-ti) — 8 GB GDDR6X, Ampere architecture, 3rd-gen Tensor Cores
- Direct codebase reading: `main_model/main_model.py`, `main_model/train.py`, `preprocess_data.py`, `capture_states.py`, `training_utils.py`, `reward_head/train.py`

---

*Feature research for: rash_kog v1.1 Training Optimization*
*Researched: 2026-05-04*
