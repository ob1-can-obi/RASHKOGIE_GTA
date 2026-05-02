# Action Planner

Proposes which tokens the agent should consider executing next.

Given the current world embedding `z_t` and the previous token, the action planner
produces a ranked list of candidate tokens from the vocabulary.

## How It Works

```
z_t [128]  +  prev_token_id
        |
        v
  intuition_head  →  z_next_pred [128]
        |
        v
  [z_t | z_next_pred]  →  planner_mlp  →  logits [vocab_size]  →  softmax  →  top-k
```

1. The intuition head predicts what the world will look like next
2. The planner MLP takes both current and predicted embeddings
3. Softmax over the full vocabulary produces a probability distribution
4. Top-k candidates (default k=3) are returned with their probabilities

## Where It Is Called

1. **Before each token execution** — seeds the search tree root with candidates
2. **Inside tree expansion** — populates each child node's candidate set,
   enabling the metacontroller to search deeper than one level

## Files

```
action_planner/
  action_planner.py   planner MLP, top-k selection
  train.py            imitation learning from player demonstrations
  training_data/      JSONL files of player driving captures
```

## Training

The action planner trains via **imitation learning** on recorded player driving sessions.
The player drives in GTA, controls are captured and tokenized, and the planner learns
to predict which token the player chose given the world state.

```bash
python action_planner/train.py
```

Target: top-1 accuracy above 60% on held-out player demonstrations.

Training data format (one JSONL line per frame):
```json
{"state": {...}, "player_controls": {...}, "token_id": 42, "session_ts": "...", "frame_idx": 0}
```
