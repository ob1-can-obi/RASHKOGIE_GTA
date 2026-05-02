# Main Model

This folder holds the neural network code for the GTA driving agent.

It is intentionally separate from `gta_stream/` (game I/O) and `metacontroller/`
(search and execution):

- `gta_stream/` — GTA V state export, control input, WebSocket bridge
- `main_model/` — encoder, intuition head, action planner (neural network code only)
- `metacontroller/` — MCTS search, executor, trainer, reward head

## Files

- `main_model.py` — main fused embedding model

  Encodes the full GTA world state into a single embedding `z_t [1, 128]`:

  ```
  ego state        →  ego encoder
  scene/road       →  scene encoder
  route/waypoints  →  route encoder
  nearby entities  →  entity encoder
                         ↓
                     attention fusion
                         ↓
                      z_t [1, 128]
  ```

  This embedding is the shared representation used by the action planner,
  intuition head, reward head, and metacontroller.

- `action_planner.py` — one-step token distribution

  Takes `z_t` and the previous token id, runs the intuition head to predict
  `z_next`, then produces a softmax distribution over the full token vocabulary.
  Returns the top-k token ids and their probabilities.

  ```
  z_t + prev_token_id  →  intuition_head  →  z_next_pred
  [z_t | z_next_pred]  →  planner_mlp    →  logits  →  top-k
  ```

  Called at two points:
  1. Before each token execution — to seed the search tree root with candidates
  2. Inside `expand_next_child` — to populate each child node's candidate set
     (enabling multi-depth MCTS search)

## What Talks to This

```
gta_stream  →  raw GTA state dict
                    ↓
              main_model.py  →  z_t [1, 128]
                    ↓
              action_planner  →  top-k token ids + probs
                    ↓
              metacontroller/search_tree  →  MCTS search
```

The main model reads raw saved GTA JSON frames from `../gta_stream/stats/` for
smoke tests, but the model code itself stays isolated in this folder.

Game execution, live training, labels, and checkpoints live outside this folder.
