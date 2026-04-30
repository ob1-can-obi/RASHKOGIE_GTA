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

### Active

- [ ] Fix metacontroller decision sampling (replace argmax with categorical sampling during training, argmax only at inference)
- [ ] Add entropy regularization to metacontroller loss to prevent decision collapse across all 4 actions
- [ ] Add penalty for not being ready (metacontroller fails to commit before token ends → large negative reward)
- [ ] Add penalty for lazy commits (immediate COMMIT_NEXT without meaningful search → negative signal)
- [ ] Determine architecture sizes: attention heads for encoder, MLP depth/width for metacontroller, action planner output heads
- [ ] Implement batch training infrastructure (replay buffer, batch sampling, progress tracking per batch)
- [ ] Build training data pipeline: central training_data folder structure for all modules
- [ ] Implement intuition head training loop (automated during gameplay, MSE on z_next vs real z_{t+1})
- [ ] Implement reward head training loop (automated during gameplay, MSE on r_edge vs realized return)
- [ ] Implement action planner training loop (imitation learning from player driving data)
- [ ] Freeze intuition head + reward head when converged, then train metacontroller via RL
- [ ] Build training dashboard: custom Flask/FastAPI web app with live metrics, session history, loss curves
- [ ] Add hyperparameter control panel to dashboard (tune params from browser, launch training runs)
- [ ] Add training session management (session logging, batch progress, before/after comparisons)
- [ ] Wire full end-to-end loop: data capture → train modules in order → run agent → evaluate → iterate

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
- Metacontroller uses argmax instead of sampling → no exploration during training
- No entropy regularization → metacontroller collapses to one decision
- No penalty for not being ready when token ends
- Think_cost incentivizes lazy immediate commits
- All MLPs are single hidden layer — likely insufficient for 237-dim metacontroller input
- No batch training — everything is online, one sample at a time
- Single attention block in encoder — may need more

**Research angle:**
Rational Cognition / Value of Cognition — the metacontroller learns WHEN to think deeper vs commit. Standard MCTS uses UCB mechanically; this agent learns that meta-decision as a policy. The tree search is not just planning — it's learned deliberation.

**Environment:**
- GTA V Enhanced on Windows PC with GPU
- State stream at ~20 Hz via named pipe
- Python 3.12 + PyTorch
- All training and inference on the same machine

## Constraints

- **Hardware**: Single Windows PC with GPU — GTA + training share resources
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
*Last updated: 2026-04-30 after initialization*
