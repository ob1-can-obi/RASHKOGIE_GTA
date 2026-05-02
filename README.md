# RASHKOGIE GTA

An autonomous GTA V driving agent that uses MCTS-style tree search to plan actions in real time.

The agent encodes the live game world into a compact embedding, searches a tree of predicted futures to find the best next action, and learns from the results. All of this happens while the current action is still executing in-game.

## How It Works

```
GTA V  ──>  gta_stream (state reader)  ──>  main_model (encoder)  ──>  z_t [128]
                                                                          |
                                              ┌───────────────────────────┘
                                              v
                                       action_planner ──> top-k candidate tokens
                                              |
                                              v
                                    ┌─────────────────────────┐
                                    │       FRAME LOOP        │
                                    │                         │
                                    │  executor        search │
                                    │  plays current   tree   │
                                    │  token in GTA    plans  │
                                    │                  next   │
                                    │                  token  │
                                    └─────────────────────────┘
                                              |
                                              v
                                          trainer (learns from outcome)
```

1. **Encode** — `main_model` fuses ego state, scene, route, and entities into a 128-dim embedding `z_t`
2. **Propose** — `action_planner` suggests the top-k candidate tokens (multi-frame control sequences)
3. **Search** — `metacontroller` runs MCTS using the `intuition_head` (forward model) and `reward_head` (value estimator) to score candidates without executing them in GTA
4. **Execute** — the best token plays frame-by-frame in GTA while the search continues planning the next token
5. **Learn** — the `trainer` computes realized rewards and updates all networks

## Modules

| Folder | What It Does |
|--------|-------------|
| `gta_stream/` | Reads live GTA V state over named pipes, sends controls back, WebSocket bridge |
| `main_model/` | Encoder that fuses raw game state into `z_t [128]` embedding |
| `intuition_head/` | Forward model — predicts next world state given current state + action |
| `reward_head/` | Neural network that scores state transitions for the search tree |
| `action_planner/` | Proposes candidate tokens from the current embedding |
| `metacontroller/` | MCTS search tree, frame loop, executor, trainer |
| `tokenizer/` | BPE tokenizer that converts raw GTA controls into a discrete token vocabulary |
| `dashboard/` | FastAPI + Vue 3 web dashboard for live training monitoring |

## Key Concepts

**Tokens** are multi-frame control chunks (steering + throttle + brake sequences). Raw GTA controls are discretized and merged via BPE into a vocabulary of ~870 tokens. Each token has a duration (number of frames it plays).

**The search tree** is built fresh each token execution. While the current token plays, the metacontroller makes one search decision per frame: EXPLORE (go deeper), ROLLBACK (try a sibling), COMMIT_NEXT (done searching), or INTERRUPT (abort current token early).

**Two reward systems**: `metacontroller/reward.py` is a math formula applied to real GTA states (ground truth). `reward_head/reward_head.py` is a neural network that approximates rewards from predicted embeddings (used inside the search tree for fast scoring).

## Training Pipeline

Training follows a strict staged order with convergence-triggered freezes:

```
1. Encoder + Intuition Head  ──>  learns to predict next world state
2. Reward Head               ──>  learns to score transitions from embeddings
3. Action Planner             ──>  learns from player demonstrations (imitation)
4. Metacontroller             ──>  learns to search efficiently (RL / REINFORCE)
```

Each stage freezes after converging. The metacontroller only trains after the reward head is frozen, so its reward signal is stable.

Run the full pipeline:
```bash
python -m training.coordinator run-all
```

## Dashboard

Live training monitoring at `http://localhost:8000`:

```bash
python -m dashboard.server
```

Shows loss curves, decision distributions, hyperparameter controls, session history, embedding visualizations, and checkpoint management.

## Project Structure

```
RASHKOGIE_GTA/
  gta_stream/          GTA V state reader and control sender
  main_model/          encoder (z_t embedding)
  intuition_head/      forward model (predicts next state)
  reward_head/         transition scorer (NN reward)
  action_planner/      token proposal network
  metacontroller/      MCTS search, executor, trainer
  tokenizer/           BPE control tokenizer
  dashboard/           FastAPI + Vue 3 training dashboard
  training/            coordinator CLI for staged training pipeline
  training_config.json hyperparameters for all modules
  training_status.json current pipeline state (which modules are frozen)
```
