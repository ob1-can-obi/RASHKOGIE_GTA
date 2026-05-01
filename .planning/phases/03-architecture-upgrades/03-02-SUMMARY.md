---
phase: 03-architecture-upgrades
plan: 02
subsystem: encoder, action-planner
tags: [architecture, attention, layernorm, mlp, encoder]
dependency_graph:
  requires: []
  provides: [ARCH-02, ARCH-03]
  affects: [main_model/main_model.py, action_planner/action_planner.py]
tech_stack:
  added: []
  patterns: [stacked-attention, residual-connection, layernorm, 2-layer-mlp]
key_files:
  created: []
  modified:
    - main_model/main_model.py
    - action_planner/action_planner.py
decisions:
  - "Block 2 query_dim=64 from ctx1 output (not 192 from ego/scene/route concat)"
  - "Residual connection only on block 2 (block 1 has dim mismatch: 192 query vs 64 output)"
  - "Action planner uses hidden_dim*2=256 for first hidden layer, matching wider-then-narrow pattern"
metrics:
  duration: 301s
  completed: 2026-05-01
  tasks: 2
  files_modified: 2
---

# Phase 3 Plan 02: Encoder 2-Block Attention + Action Planner 2-Layer MLP Summary

2-block stacked cross-attention with LayerNorm and residual on block 2; action planner widened to 256-128-V MLP.

## What Was Done

### Task 1: Upgrade encoder to 2 stacked attention blocks with LayerNorm (ARCH-02)

**Commit:** f31aa1c

In `create_encoder_weights()`:
- Replaced single attention weight creation (`qw/kw/vw/ow`) with two blocks:
  - Block 1: `qw1/kw1/vw1/ow1` with query_dim=192 (ego+scene+route concat)
  - Block 2: `qw2/kw2/vw2/ow2` with query_dim=64 (block 1 output)
- Added `ln_attn1 = nn.LayerNorm(64)` and `ln_attn2 = nn.LayerNorm(64)` to the weight dict

In `encode_state()`:
- Block 1: cross-attention from ego/scene/route query to entity K/V, followed by LayerNorm
- Block 2: refine using ctx1 as query (same entity K/V), residual connection (attn2 + ctx1), followed by LayerNorm
- Fusion block unchanged -- uses entity_context from the 2-block pipeline

Updated module docstring to reflect 2-block architecture diagram.

**Key dimensions verified:**
- qw1 shape: [192, 64] (query_dim=192)
- qw2 shape: [64, 64] (query_dim=64)
- Encoder output: [1, 128] (unchanged)

### Task 2: Upgrade action planner to 2-layer MLP (ARCH-03)

**Commit:** 0dc5c8c

In lazy init block:
- Changed from `Linear(256,128) -> ReLU -> Linear(128,V)` (1 hidden layer)
- To `Linear(256,256) -> ReLU -> Linear(256,128) -> ReLU -> Linear(128,V)` (2 hidden layers)
- Updated module and function docstrings to reflect new architecture

**Key dimensions verified:**
- Layer 0: in=256, out=256
- Layer 1: in=256, out=128
- Layer 2: in=128, out=874 (vocab_size)
- Output shapes unchanged: logits [1, 874], top_k_ids [1, 3]

## Deviations from Plan

None - plan executed exactly as written.

Note: The plan's verification test used `qw.shape[1]` to check query_dim, but `create_multi_head_attention_weights` creates qw with shape `[query_dim, embed_dim]`, so the correct index is `shape[0]`. The verification was adjusted accordingly; the implementation is correct per the architecture specification.

## Decisions Made

1. **Block 2 query_dim=64:** Second attention block uses the output of block 1 (ctx1, dim=64) as its query, not the original ego/scene/route concatenation (dim=192). This follows the standard iterative refinement pattern.

2. **Residual on block 2 only:** No residual on block 1 because query_dim (192) differs from output_dim (64). Block 2 has matching dimensions (64 in, 64 out) enabling the residual add.

3. **Action planner first hidden dim = hidden_dim*2 = 256:** Wider first layer matches the input dimension and follows the wider-then-narrow pattern consistent with the metacontroller MLP design.

## Verification Results

| Check | Result |
|-------|--------|
| Encoder weight keys qw1/kw1/vw1/ow1 present | PASS |
| Encoder weight keys qw2/kw2/vw2/ow2 present | PASS |
| LayerNorm modules ln_attn1/ln_attn2 present | PASS |
| Old keys qw/kw/vw/ow removed | PASS |
| Block 1 query_dim = 192 | PASS |
| Block 2 query_dim = 64 | PASS |
| Encoder output shape [1, 128] | PASS |
| Residual connection (+ ctx1) present | PASS |
| Action planner has 3 Linear layers | PASS |
| Layer dimensions 256->256->128->V | PASS |
| Action planner output shapes unchanged | PASS |
| Smoke test (python main_model/main_model.py) clean | PASS |

## Self-Check: PASSED

- main_model/main_model.py: FOUND
- action_planner/action_planner.py: FOUND
- 03-02-SUMMARY.md: FOUND
- Commit f31aa1c: FOUND
- Commit 0dc5c8c: FOUND
