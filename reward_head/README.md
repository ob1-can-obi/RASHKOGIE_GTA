# Reward Head

Neural network node evaluator for the MCTS search tree.

During tree search, the intuition head predicts what the world will look like
after each candidate token runs.  The reward head takes those predicted
embeddings — plus explicit reward-relevant signals — and estimates the reward
for that transition without needing GTA to actually execute it.

Once trained, the reward head lets the search tree score every candidate in
milliseconds using embeddings alone.

## Two Separate Reward Systems

| | `metacontroller/reward.py` | `reward_head/reward_head.py` |
|---|---|---|
| Type | Pure math formula | Neural network |
| Runs on | Real GTA states | Predicted embeddings |
| Used by | Executor (ground truth) | Search tree (fast scoring) |
| Output | Per-frame scalar reward | `r_edge` for one transition |

`reward.py` is the source of truth.  The reward head NN is trained to approximate
it from embeddings alone, so the search tree can score candidates cheaply.

## Explicit Reward Features (RF_DIM = 6)

Rather than hoping the network discovers reward-relevant information buried in a
128-dim embedding, the most important signals are kept explicit:

```
rf = [wp_dist, hp, v_engine_hp, v_body_hp, road_dist, dead]
```

| Field | Meaning |
|---|---|
| `wp_dist` | Distance to next waypoint (lower = better progress) |
| `hp` | Player health |
| `v_engine_hp` | Vehicle engine health |
| `v_body_hp` | Vehicle body health |
| `road_dist` | Distance from road centre (higher = off-road) |
| `dead` | 1.0 if player is dead |

For root nodes, these come from the real GTA state via `extract_reward_features`.
For child nodes (predicted states), they are estimated by `predict_reward_features`.

## Functions

### `extract_reward_features(gta_state) → [1, RF_DIM]`

Pulls reward-relevant fields directly from a raw GTA state dict.
Used on real states — root node and post-execution states.

### `predict_reward_features(z_parent, delta_z, rf_parent) → (rf_child, rf_predictor)`

Predicts what the reward features will look like at a child node.
Since the child state is only predicted (via the intuition head), we cannot
read its raw GTA values.  This MLP estimates them from the parent features
and the embedding change.

```
input:  [z_parent | delta_z | rf_parent]  →  fused_dim*2 + RF_DIM
output: rf_child  [batch, RF_DIM]
```

### `reward_head(z_parent, z_child, rf_parent, rf_child, duration, time_left) → (r_edge, reward_mlp)`

Scores a parent → child transition.

## Inputs

| Tensor | Shape | What it is |
|---|---|---|
| `z_parent` | `[batch, fused_dim]` | Embedding at the parent node |
| `z_child` | `[batch, fused_dim]` | Predicted embedding at the child node |
| `rf_parent` | `[batch, RF_DIM]` | Reward features at the parent (real) |
| `rf_child` | `[batch, RF_DIM]` | Reward features at the child (predicted) |
| `duration` | `[batch, 1]` | Token duration in frames |
| `time_left` | `[batch, 1]` | Frames remaining until deadline |

## Output

| Tensor | Shape | What it is |
|---|---|---|
| `r_edge` | `[batch, 1]` | Predicted reward for this transition |

## Architecture

```
z_parent   [batch, 128]  ──┐
delta_z    [batch, 128]  ──┤
rf_parent  [batch, 6]    ──┼──> cat [batch, 270]
delta_rf   [batch, 6]    ──┤        │
duration   [batch, 1]    ──┤    Linear(270, 128) → ReLU
time_left  [batch, 1]    ──┘    Linear(128,  64) → ReLU
                                Linear( 64,   1)
                                     │
                                  r_edge [batch, 1]
```

`delta_z = z_child - z_parent` — change in embedding, not just destination.
`delta_rf = rf_child - rf_parent` — change in reward features (progress, damage, etc.).

Total input dim: `fused_dim * 2 + RF_DIM * 2 + 2 = 270` (with fused_dim=128).

## Training

The reward head is trained online after each token execution in `trainer.train_reward_head`.

**reward_mlp target:** the realized discounted token return `R_token` (ground truth
from `reward.py` applied to real GTA frames).

**rf_predictor target:** the real `rf_child` extracted from the actual GTA state
after the token ran.

```
reward_loss = (r_pred - R_token)^2
rf_loss     = MSE(rf_pred, rf_child_real)
total_loss  = reward_loss + rf_loss
```

Both MLPs are updated together with manual SGD.

Training logs go to `reward_head/training_data/session_YYYYMMDD_HHMMSS.jsonl`.

Each JSONL row:
```json
{"step": 0, "reward_loss": 0.042, "rf_loss": 0.011, "predicted": 0.31, "actual": 0.28}
```

Checkpoints go to `reward_head/checkpoints/reward_head_YYYYMMDD_HHMMSS.pt`.

## Stats

Rebuild after each training session:

```bash
python reward_head/stats.py
```

Writes:
```
reward_head/training_data/stats/reward_head_stats.csv
reward_head/training_data/stats/summary.txt
reward_head/training_data/graphs/loss_curve.svg
reward_head/training_data/graphs/predicted_vs_actual.svg
```
