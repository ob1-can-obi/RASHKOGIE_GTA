# Project Research Summary

**Project:** RASHKOGIE GTA — v1.1 Training Optimization
**Domain:** PyTorch offline training pipeline optimization for autonomous driving RL agent
**Researched:** 2026-05-04
**Confidence:** HIGH

## Executive Summary

RASHKOGIE GTA v1.1 is a targeted performance optimization milestone for an existing PyTorch-based autonomous driving agent. The project has a working training pipeline, 88 GB of captured JSONL data (~236k records, ~800 MB of useful numeric content once converted), and a module-per-directory architecture covering encoder, reward head, action planner, and metacontroller. The v1.1 goal is to move from a CPU-paced pseudo-batch training loop (GPU utilization ~10-20%) to true GPU-batched training on an RTX 3070 Ti (8 GB VRAM) using compact tensor storage, learned categorical embeddings, DataLoader integration, and CUDA AMP mixed precision.

The recommended approach follows a strict dependency chain: data format comes first (preprocessing must extract categorical integer indices and guard against OneDrive corruption before the 88 GB conversion runs), then embedding integration (new `nn.Embedding` tables added to the encoder dict with explicit optimizer registration), then training loop batching (inner Python loop replaced with `TensorDataset` + `DataLoader`), and finally AMP (fp16 `autocast` + `GradScaler` applied across all three trainers). All four areas use PyTorch core APIs only — no new packages are required. The single infrastructure change is switching from `torch+cpu` to `torch+cu124` in the Windows-native Python environment (WSL2 does not expose the RTX 3070 Ti regardless of torch version).

The primary risks are silent-failure class errors: preprocessing building 16+ GB intermediate lists on a system already loaded by GTA V; old float32 categorical fields silently passing to embedding lookup and producing wrong results; GradScaler skipping every optimizer step when embedding gradients overflow to `inf` in fp16; and OneDrive corrupting large `.pt` files during background sync. All of these are detectable and preventable with explicit assertions and monitored validation gates at each phase boundary. Existing checkpoints are incompatible with the new encoder (MLP input dims grow: ego 46→54, scene 16→24, entity 24→40); v1.1 trains from scratch by design.

---

## Key Findings

### Recommended Stack

No new packages are required for v1.1. All four feature areas are served by existing PyTorch 2.x core APIs. The only infrastructure change is switching from the CPU build to the CUDA build in the Windows-native Python environment. The CUDA index URL (`https://download.pytorch.org/whl/cu124`) is already in `requirements.txt`. WSL2 cannot access the RTX 3070 Ti regardless of torch version — all training scripts must run from Windows cmd/PowerShell, not WSL2.

**Core technologies:**
- `torch.amp.autocast` + `torch.amp.GradScaler("cuda")` — mixed precision training; must use the unified 2.x API, not the deprecated `torch.cuda.amp.*` namespace which produces warnings in PyTorch 2.x
- `torch.utils.data.TensorDataset` + `DataLoader` — batched training; `pin_memory=True`, `num_workers=0` (Windows spawn safety), `drop_last=True` for Tensor Core alignment
- `torch.nn.Embedding` — learned categorical embeddings for `weather`, `v_class`, `v_model`, entity `type_id`, entity `bucket_id`; five tables totaling less than 1 MB VRAM
- `torch.save` / `torch.load` — compact `.pt` tensor format already in use; extended to include `torch.long` categorical index tensors alongside existing float tensors

**Critical hardware and version constraints:**
- fp16 not bf16: RTX 3070 Ti (Ampere sm_86) does not hardware-accelerate bf16 matmul; bf16 runs at fp32 speed on this card
- `num_workers=0` on Windows: `num_workers>0` causes `EOFError` in spawn multiprocessing when training scripts are imported as modules; no performance cost since the dataset is RAM-resident (~800 MB)
- Batch sizes must be multiples of 8 for Tensor Core activation; use 64, 128, or 256 — not arbitrary values like 100 or 150

### Expected Features

**Must have — table stakes for GPU training:**
- True batched encoder forward pass: `encode_tensors()` accepts `[B, dim]` tensors; replaces per-record `[1, dim]` Python inner loop that causes 10-50x GPU underutilization
- `DataLoader` integration: replaces manual index slicing; provides automatic shuffle, `pin_memory`, and `drop_last`
- Compact `.pt` format with categorical `torch.long` indices: prerequisite for both batching and embeddings; updated `preprocess_data.py` extracts integer index tensors alongside existing float tensors
- CUDA AMP (`autocast` + `GradScaler`): applied to all three trainers; `clip_grad_norm_` must be called after `scaler.unscale_()`, not before
- Action planner preprocessing path: `action_planner/train.py` currently has no `.pt` fast path; `preprocess_action_planner()` must be added to `preprocess_data.py`

**Should have — model quality improvements:**
- Learned categorical embeddings (`nn.Embedding`): weather (dim 4), v_class (dim 8), v_model (dim 8, capped at 200 vocab entries), entity `type_id` (dim 4), entity `bucket_id` (dim 4); eliminates the false ordinal relationship imposed by float-encoding nominal categories
- New `main_model/embedding_tables.py`: centralizes vocabulary size constants and `create_embedding_tables()` so preprocessor and trainer share the same definitions
- Updated `capture_states.py` compact capture: new game sessions write buffered tensor dicts instead of raw JSONL; eliminates the preprocessing step for future captures

**Defer to v1.1.x / v2+:**
- `torch.compile()` on the encoder: blocked by the dict-based weight API; requires an `nn.Module` refactor first
- Chunked mmap DataLoader: unnecessary at 800 MB; only relevant if the dataset grows beyond system RAM
- Batch size sweep (64→256): run after AMP is stable; not a prerequisite for correctness
- bf16: only beneficial on Hopper/Ada hardware (RTX 4090+, H100); produces no gain on RTX 3070 Ti

### Architecture Approach

The existing module-per-directory architecture is preserved entirely. v1.1 changes are surgical: five files are modified, one new file is created, and seven files are explicitly confirmed unchanged. The encoder dict API (`encoder_weights["ego_mlp"](x)`) is retained for this milestone; refactoring to `nn.Module` is deferred. The build order is dictated by hard code dependencies: embedding vocabulary constants must be scanned from real capture data before coding begins, and the data format must be locked before the 88 GB preprocessing run starts.

**Major components and their v1.1 changes:**

1. `main_model/embedding_tables.py` (NEW) — vocabulary size constants, `create_embedding_tables()`, save/load helpers; must be built first as it is a dependency of both `main_model.py` and `preprocess_data.py`
2. `preprocess_data.py` (MODIFIED) — adds categorical `torch.long` index extraction to main model and reward head paths; adds new `preprocess_action_planner()` function; switches to two-pass RAM-efficient processing to avoid 16+ GB intermediate peak
3. `training_utils.py` (MODIFIED) — adds `make_scaler(device)` and `load_tensor_sessions(data_dir)` helper functions; all existing utilities unchanged
4. `main_model/main_model.py` (MODIFIED) — embedding lookup and concatenation into MLP inputs before each sub-encoder; MLP input dims grow (ego 46→54, scene 16→24, entity 24→40)
5. `main_model/train.py`, `reward_head/train.py`, `action_planner/train.py` (MODIFIED) — inner loop replaced with `DataLoader`; forward pass wrapped in `autocast`; gradient clipping reordered after `scaler.unscale_()`
6. `capture_states.py` (MODIFIED) — `StateCaptureSession` writes buffered tensor dicts instead of per-frame JSONL; done last as it has no training dependency

**Files confirmed unchanged:** `multi_head_attention.py`, `reward_head/reward_head.py`, `metacontroller/`, all checkpoint structures, `training_status.json` schema, `StreamingJSONLDataset` fallback.

### Critical Pitfalls

1. **RAM explosion in preprocessing** — the current list-based stacking approach peaks at 16+ GB RAM for a single 25 GB session file (Python JSON overhead plus simultaneous tensor list plus stacked output). Fix: two-pass approach (count records first, pre-allocate `torch.zeros(N, dim)`, fill row-by-row) or per-session chunked processing. Must be resolved in Phase 1 before touching the 88 GB dataset.

2. **Float32 categoricals silently corrupting embedding lookup** — old `preprocessed.pt` stores `type_id`, `bucket_id`, etc. as float32 inside `entities_t`; a `.long()` cast silently truncates `3.9999` to 3 or produces out-of-range indices from float values outside `[0, num_categories)`. Fix: define a new separate `entities_cat` tensor of dtype `torch.long`; add a `format_version` key; delete all old `preprocessed.pt` files before training with embeddings.

3. **Embedding tables not registered with optimizer** — `create_encoder_weights()` returns a nested dict; if new embedding sub-modules are added under `encoder_weights["cat_embeds"]` but not explicitly added to the Adam `all_params` list, their parameters receive no gradient updates silently. Fix: rebuild `all_params` after any dict mutation; add assertion that param count in optimizer matches expected total including embedding parameters.

4. **GradScaler silent step-skipping** — new `nn.Embedding` tables initialized at default std=1.0 with a high learning rate can produce `inf` gradients in fp16 on the first backward pass; GradScaler halves the scale and skips the step every iteration, leaving the model frozen while loss appears constant. Fix: initialize embedding weights with `std=0.01`; log `scaler.get_scale()` per step; alert if it drops below 256.

5. **OneDrive corrupts large `.pt` files** — the project lives under `C:\Users\laksh\OneDrive\Documents\`; OneDrive's file system filter can acquire a read lock during `torch.save()` of 800 MB files, producing a truncated file that raises `EOFError` at next load with no write-time error. Fix: write to a temp directory outside OneDrive and use `shutil.move()` atomically, or use a `.tmp` suffix and rename on success.

---

## Implications for Roadmap

The research produces a clear, dependency-driven phase structure. The phases map directly to the pitfall-to-phase table in PITFALLS.md and the build order in ARCHITECTURE.md. Each phase has a hard validation gate before the next may begin.

### Phase 1: Data Format

**Rationale:** Every downstream feature depends on correct data format. The 88 GB preprocessing run is the longest-running wall-time operation in the milestone. It must be started early and must produce correct output — getting the format wrong and having to reprocess is the highest time cost in the entire milestone.

**Delivers:** `preprocessed.pt` files for all three modules (main model, reward head, action planner) containing float tensors and separate `torch.long` categorical index tensors; a `format_version` key for compatibility detection; write-to-temp-then-rename pattern applied to all saves to guard against OneDrive corruption.

**Addresses:** Compact `.pt` format (table stakes), updated preprocessor with categorical index extraction, action planner preprocessing path, preprocessing tool for existing 88 GB JSONL.

**Avoids:** RAM explosion in preprocessing (Pitfall 1), float32 categoricals breaking embedding lookup (Pitfall 2), OneDrive file corruption (Pitfall 7).

**Validation gate:** `torch.load(preprocessed.pt)["entities_cat"].dtype == torch.long`; `format_version` key present; file size matches expected; action planner `.pt` loads without error.

### Phase 2: Learned Categorical Embeddings

**Rationale:** Embeddings require the `torch.long` index tensors from Phase 1. They must be built before the training loop refactor because the refactor must know the correct MLP input dimensions (`ego_mlp` 46→54, etc.). Embedding tables must also be explicitly registered with the optimizer before AMP is layered on — catching the optimizer-registration pitfall in isolation is simpler than debugging it alongside GradScaler skip behavior.

**Delivers:** `main_model/embedding_tables.py` with confirmed vocabulary sizes scanned from real data; updated `create_encoder_weights()` and `encode_tensors()` with embedding lookup and concatenation; updated MLP input dimensions; smoke test confirming `z_t.shape == [1, 128]` with embedding tables active; checkpoint save/load updated to include embedding table state dicts.

**Addresses:** Learned categorical embeddings (differentiator), embedding table centralization.

**Avoids:** Embedding tables not in optimizer (Pitfall 3), embedding state not in checkpoint (Architecture anti-pattern 5), out-of-bounds embedding index from edge GTA category values (add `.clamp(0, VOCAB_SIZE - 1)` before every lookup).

**Validation gate:** Assert optimizer param count equals expected total including embedding parameters; verify `z_t.dtype` is correct; confirm embedding keys are present in saved checkpoint.

### Phase 3: Batched Training Loop

**Rationale:** True batching is the primary GPU utilization fix and the core deliverable of v1.1. It requires Phase 1 (`.pt` data in RAM for O(1) slicing) and Phase 2 (correct MLP input dims so `[B, dim]` forward passes produce correct `[B, 128]` output). Batching is validated in fp32 before AMP is added so that AMP-related issues do not obscure batching shape errors.

**Delivers:** All three trainers using `TensorDataset` + `DataLoader`; single `.to(device, non_blocking=True)` call per batch; `encode_tensors()` verified to handle `[B, 46]` input and return `[B, 128]`; GPU utilization above 60% confirmed by `nvidia-smi`.

**Addresses:** Per-record Python loop defeating GPU batching (table stakes), DataLoader integration (table stakes), CPU-to-GPU transfer minimization.

**Avoids:** Staging all data on VRAM (keep in CPU RAM, slice and transfer one batch at a time), `unsqueeze(0).to(device)` inside the batch loop, `torch.cat` on a list of per-record tensors.

**Validation gate:** `nvidia-smi` GPU utilization above 60% during training; `encode_tensors()` returns `[B, 128]` for B=32; training loss curve matches pre-refactor behavior within noise over first 100 steps.

### Phase 4: CUDA Mixed Precision (AMP)

**Rationale:** AMP is last because it depends on everything before it. Tensor Core throughput gain requires batched inputs (Phase 3). GradScaler behavior is easier to diagnose once the optimizer already correctly registers all parameters including embeddings (Phase 2). Start with batch_size=32 (not 256) to verify VRAM headroom with GTA V potentially running before scaling up.

**Delivers:** `torch.amp.autocast` wrapping the forward pass in all three trainers; `GradScaler` wrapping the backward with correct gradient clipping order (`scaler.unscale_()` before `clip_grad_norm_`); `make_scaler(device)` helper in `training_utils.py`; `training_config.json` updated with `amp: true` and increased batch sizes; per-step GradScaler scale monitor; VRAM headroom verified.

**Addresses:** AMP mixed precision (table stakes), batch size tuning (P2 feature).

**Avoids:** GradScaler silent step-skipping (Pitfall 6 — initialize embeddings at std=0.01, monitor scale per step), AMP OOM from GTA VRAM contention (Pitfall 5 — log `max_memory_allocated()` after first backward, start at batch_size=32), MSE underflow in fp16 for small prediction errors (cast loss inputs to fp32: `mse_loss(z_next_pred.float(), z_t1_real.float())`).

**Validation gate:** `z_t.dtype == torch.float16` inside autocast block; `scaler.get_scale()` above 256 after 10 steps; `torch.cuda.max_memory_allocated()` below 5 GB; GPU utilization above 60% at target batch size.

### Phase 5: Compact Capture Format

**Rationale:** `capture_states.py` changes do not block training and do not depend on any Phase 1-4 output. Doing this last means the entire training pipeline is validated before committing to the new capture format. Future game sessions will write pre-built tensor dicts, eliminating the preprocessing step.

**Delivers:** `StateCaptureSession` writes buffered tensor dicts to `session_*.pt` files; flushes every 1000 frames into an accumulator and saves once on `close()`; small sidecar JSON metadata for human inspection; write-to-temp-then-rename for crash safety.

**Addresses:** Inline compact capture (P2 differentiator).

**Avoids:** Data loss on game crash (flush-then-concatenate strategy), OneDrive corruption of per-session `.pt` files.

### Phase Ordering Rationale

- Data format is first because the 88 GB preprocessing run is the longest wall-time operation and must produce correct `torch.long` indices before embeddings can be coded.
- Embeddings are second because they change MLP input dimensions, and those dimensions must be correct before the batching refactor is written.
- Batching is third because it is the core GPU utilization fix and should be debugged in fp32 before AMP is layered on top.
- AMP is last because it depends on correct optimizer registration (Phase 2) and batch dimensions (Phase 3); debugging GradScaler skip behavior is much easier when everything else is already working.
- Capture format is decoupled and done last to avoid blocking the training improvement work.

### Research Flags

Phases needing extra care during implementation:

- **Phase 1 (Data Format):** Vocabulary sizes for `weather`, `v_class`, `v_model`, entity `type_id`, and `bucket_id` must be scanned from real capture session files before writing `embedding_tables.py` constants. GTA V has been observed to return edge values (e.g., `v_class=255` for unknown vehicles); without `.clamp(0, VOCAB_SIZE - 1)` before every embedding lookup, this will cause an `IndexError` at training time.
- **Phase 2 (Embeddings):** `freeze_module()` in `training_utils.py` does not recurse into nested dicts. Embedding tables stored under `encoder_weights["cat_embeds"]` will not be frozen when the reward head trainer calls `freeze_module(encoder_weights)`. Either flatten embedding tables to top-level keys in `encoder_weights` (simpler, consistent with how `ego_mlp` etc. are stored) or update `freeze_module()` to recurse into nested dicts.
- **Phase 4 (AMP):** AMP + MSE loss between fp16 tensors can underflow for small prediction errors (below 6e-5), giving zero loss silently. Cast loss inputs to fp32 before computing MSE: `mse_loss(z_next_pred.float(), z_t1_real.float())`.

Phases with standard, well-documented patterns where additional research is not needed:

- **Phase 3 (Batching):** `TensorDataset` + `DataLoader` with `pin_memory=True` is canonical PyTorch; no implementation ambiguity.
- **Phase 5 (Capture):** Buffer-and-flush with `shutil.move` atomic rename is a standard safe-write pattern; straightforward implementation.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All APIs verified against PyTorch 2.11 official docs; cu124 URL confirmed already in requirements.txt; RTX 3070 Ti Ampere sm_86 hardware specs confirmed; bf16 non-acceleration on Ampere confirmed from NVIDIA docs |
| Features | HIGH | Derived from direct code audit of all training scripts, preprocessor, and capture pipeline; current per-record loop pattern confirmed by reading `main_model/train.py` line-by-line |
| Architecture | HIGH | All findings from direct codebase reading; build order validated against actual import dependencies; MLP input dimension changes computed from field lists in `main_model.py` |
| Pitfalls | HIGH | RAM explosion derived from measured session file sizes (25 GB largest session) and Python dict overhead characteristics; OneDrive pitfall from confirmed project path; GradScaler skip from PyTorch source |

**Overall confidence:** HIGH

### Gaps to Address

- **Embedding vocabulary sizes:** Exact cardinality of `weather`, `v_class`, `v_model`, entity `type_id`, and entity `bucket_id` must be determined by scanning real session JSONL files before writing `embedding_tables.py`. ARCHITECTURE.md provides estimates (weather ~13, v_class ~23, type_id ~5, bucket_id ~8) but these must be confirmed. This is a 10-minute scan script, not a research gap — it is the mandatory first task of Phase 1.
- **`multi_head_attention.py` batch dimension verification:** ARCHITECTURE.md states the attention module is "batch-dimension agnostic" based on the presence of standard `nn.Linear` layers, but this has not been verified with a `[B, 32, ...]` forward pass. Add an explicit smoke test (batch=32 through the attention block) at the start of Phase 3 before the full training loop refactor.
- **GTA VRAM footprint during simultaneous training:** The headroom estimate (3 GB for GTA, 5 GB available for training) is an upper bound. Actual GTA VRAM usage varies with scene complexity and graphics settings. Measure empirically with `torch.cuda.max_memory_allocated()` after the first backward pass. If training without GTA running (offline-only), the full 8 GB is available and batch_size up to 512 is feasible.

---

## Sources

### Primary (HIGH confidence)

- PyTorch AMP documentation 2.11 — `autocast`, `GradScaler`, unified API vs deprecated `torch.cuda.amp` namespace: `https://docs.pytorch.org/docs/stable/amp`
- PyTorch AMP tutorial — `https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html`
- NVIDIA Mixed Precision Training Guide — Tensor Core alignment requirements, bf16 architecture support matrix: `https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html`
- PyTorch DataLoader best practices — pin_memory, num_workers, prefetch: `https://medium.com/@Modexa/8-pytorch-dataloader-tactics-to-max-out-your-gpu-22270f6f3fa8`
- PyTorch multiprocessing best practices — Windows spawn constraints for num_workers: `https://docs.pytorch.org/docs/stable/notes/multiprocessing.html`
- Direct codebase audit — `main_model/main_model.py`, `main_model/train.py`, `reward_head/train.py`, `action_planner/train.py`, `preprocess_data.py`, `capture_states.py`, `training_utils.py`, `training_config.json`

### Secondary (MEDIUM confidence)

- RTX 3070 Ti CUDA guide — 8 GB GDDR6X, Ampere sm_86, 3rd-gen Tensor Cores: `https://www.rightnowai.co/guides/gpu-comparison/rtx-3070-ti`
- GradScaler skip logic — PyTorch source `torch/amp/grad_scaler.py`
- CUDA memory fragmentation — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: `https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.empty_cache.html`
- PyTorch Embedding for categorical data: `https://discuss.pytorch.org/t/nn-embedding-layer-in-categorical-features/148558`

### Tertiary (needs empirical validation)

- GTA V VRAM footprint during training — estimated 2-4 GB; must be measured at actual graphics settings in the target environment
- Categorical vocabulary sizes — estimated from GTA domain knowledge; must be confirmed by scanning real session data before coding begins

---

*Research completed: 2026-05-04*
*Ready for roadmap: yes*
