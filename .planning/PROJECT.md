# RASHKOGIE GTA

## What This Is

An autonomous driving agent for GTA V that uses MCTS-style tree search guided by a learned metacontroller to plan actions in real-time. The agent tokenizes driving controls via BPE, encodes the full game world into a fused embedding, and runs a search tree during token execution to decide the best next action — all while the car is driving. The research contribution is **Rational Cognition**: the metacontroller learns the Value of Cognition — when thinking deeper is worth the computational cost, and when to commit.

## Core Value

The metacontroller must learn to search intelligently — using the tree to find better actions than the planner's top-1, while staying ready before the current token ends. Thinking is the product. If it can't think well, nothing else matters.

## Requirements

### Validated

- Validated GTA state streamer (C# SHVDN script + Python reader + control pipe + WebSocket bridge) — existing
- Validated BPE tokenizer with stable IDs, session boundaries, and incremental rebuild — existing
- Validated module architecture: encoder, intuition head, action planner, reward head, metacontroller, search tree, executor, frame loop, trainer — existing code structure
- ✓ Metacontroller training correctness (sampling, entropy, penalties) — v1.0 Phase 1
- ✓ Batch training + checkpointing (replay buffer, save/resume) — v1.0 Phase 2
- ✓ Architecture upgrades (MetaMLP skip connections, dual attention, LayerNorm) — v1.0 Phase 3
- ✓ Module training pipelines (encoder, reward, action planner, orchestrator) — v1.0 Phase 4
- ✓ Training dashboard (live metrics, PCA embeddings, hot-reload) — v1.0 Phase 5

## Current Milestone: v1.1 Training Optimization

**Goal:** Make training fast, GPU-accelerated, and memory-efficient on RTX 3070 Ti (8 GB VRAM)

**Target features:**
- Inline data preprocessing during capture — write compact tensors, not bloated JSONL
- Learned embeddings for categorical fields (weather, v_class, v_model, entity type_id/bucket_id)
- Batched encoder forward pass — process full batches through MLPs and attention in one GPU call
- CUDA optimization — mixed precision, proper device management, minimize CPU↔GPU transfers
- Preprocessing tool for existing captured data

### Active

- [ ] Capture pipeline writes compact tensor format instead of raw JSONL (88 GB → ~1.2 GB)
- [ ] Preprocessing tool converts existing 88 GB JSONL captures to compact format
- [ ] Learned embeddings for categorical fields in encoder (weather, v_class, v_model, entity type_id, bucket_id)
- [ ] Batched encoder forward pass (all records in batch processed in one GPU call)
- [ ] Batched training loops for all trainers (main_model, reward_head, action_planner)
- [ ] CUDA mixed precision (fp16) training with gradient scaling
- [ ] Minimize CPU↔GPU data transfers during training

### Out of Scope

- Multi-agent scenarios — single vehicle only for v1
- Image/pixel-based perception — state comes from SHVDN game data, not screenshots
- Cloud training — same Windows PC runs GTA + training
- Mobile or web deployment — this is a research project running locally
- Pretrained model integration (GPT, etc.) — all modules are trained from scratch on GTA data

## Context

**Architecture overview:**
```
GTA V → C# streamer → named pipe → Python reader → raw state dict
                                                         ↓
                                                    main_model encoder → z_t [1, 128]
                                                         ↓
                                              intuition_head → z_next_pred [1, 128]
                                              action_planner → top-k tokens
                                                         ↓
                                              ┌──────────────────────────────┐
                                              │        FRAME LOOP           │
                                              │ executor plays token frames │
                                              │ search_tree expands nodes   │
                                              │ metacontroller decides:     │
                                              │   EXPLORE / ROLLBACK /      │
                                              │   INTERRUPT / COMMIT_NEXT   │
                                              └──────────────────────────────┘
                                                         ↓
                                              trainer → policy gradient + reward head update
```

**Training order (strict dependency chain):**
1. Tokenizer — build from player driving captures (done)
2. Encoder — trains to produce useful z_t embeddings
3. Intuition head — forward model, trains on (z_t, action) → z_{t+1} pairs from real gameplay
4. Reward head — trains on (z_parent, z_child) → realized_return from real gameplay
5. Freeze intuition head + reward head
6. Action planner — imitation learning from player demonstration data
7. Metacontroller — RL (REINFORCE with metalevel credit assignment) while driving in GTA

**Known code issues (from audit):**
- ✓ FIXED in v1.0: argmax, entropy, penalties, architecture, batch training
- Categorical fields (weather, v_class, v_model, entity type_id/bucket_id) treated as continuous floats — meaningless to MLPs
- Training data is 88 GB raw JSONL but only ~1.2 GB of useful numeric data — 74x bloat from JSON overhead and unused fields (near_vehs, near_peds, near_objects)
- Training loops process one record at a time in Python — no GPU batching
- ~236k records across 6 sessions (4.6 hours of driving)

**Research angle:**
Rational Cognition / Value of Cognition — the metacontroller learns WHEN to think deeper vs commit. Standard MCTS uses UCB mechanically; this agent learns that meta-decision as a policy. The tree search is not just planning — it's learned deliberation.

**Environment:**
- GTA V Enhanced on Windows PC with GPU
- State stream at ~20 Hz via named pipe
- Python 3.12 + PyTorch
- All training and inference on the same machine

## Constraints

- **Hardware**: Single Windows PC with RTX 3070 Ti (8 GB VRAM) — GTA + training share resources
- **Latency**: Agent must keep up with ~20 Hz game loop, decisions within token duration
- **Data**: All training data comes from playing GTA — no external datasets
- **Training order**: Modules must train in strict order due to dependencies (intuition head before reward head before action planner before metacontroller)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| BPE tokenization of controls | Compresses multi-frame sequences into single tokens, enables variable-duration actions | Validated — tokenizer works |
| Fused embedding z_t [128] | Single representation consumed by all modules, clean interface | Validated — encoder produces z_t |
| Separate reward.py (math) vs reward_head.py (NN) | Ground truth for training vs fast scoring for search | Validated — clean separation |
| MCTS guided by learned metacontroller | Research contribution — Value of Cognition, not mechanical UCB | — Pending |
| Custom web dashboard over TensorBoard | Full control over hyperparameter tuning UI and session management | — Pending |
| Manual SGD in trainer | Simple, no optimizer state — may need upgrade to Adam with batches | — Pending |
| Module independence (train separately, freeze, detach) | Each module can be debugged and validated independently | Validated — clean boundaries |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-03 after milestone v1.1 start*
