# Intuition Head

The intuition head answers one question: **given the current world state and the
last action taken, where will the world be next?**

It is the model's forward model — a learned prediction of how each token changes
the world embedding.

## What It Does

```
z_t  [1, 128]          current fused state embedding (from main_model)
prev_token_id  [1]     the token that was just executed (or is being considered)
        │
        ▼
token_embed(prev_token_id)  →  prev_emb  [1, 32]
        │
        ▼
[z_t | prev_emb]  →  intuition_mlp  →  delta_z  [1, 128]
        │
        ▼
z_next_pred = z_t + delta_z          predicted next world embedding  [1, 128]
```

The output is a **residual prediction**: instead of predicting `z_next` from
scratch, it predicts the *change* `delta_z` and adds it to `z_t`.  This keeps
predictions grounded — if the token does very little, `delta_z ≈ 0` and
`z_next_pred ≈ z_t`.

## Architecture

```
token_embed    nn.Embedding(vocab_size, 32)
intuition_mlp  Linear(fused_dim + 32, 128) → ReLU → Linear(128, fused_dim)
```

Both are lazily created on the first call if not passed in, then returned so the
caller can reuse them across calls.  All weights are stateless from the function's
perspective — the caller owns them and passes them back in each time.

## Where It Is Used

The intuition head is called in three places, always with the same
`token_embed` and `intuition_mlp` so the weights are shared:

### 1. Root candidate seeding (`main_model.get_candidates`)

Before the search tree starts, the main model calls the intuition head once to
get `z_next_pred` for the action planner:

```
z_t + prev_token_id  →  intuition_head  →  z_next_pred
[z_t | z_next_pred]  →  action_planner  →  top-k root candidates
```

### 2. Child node expansion (`search_tree.expand_next_child`)

When the metacontroller EXPLOREs a candidate, the intuition head rolls through
each base token of the merged token to predict `z_child`:

```
merged token 500  →  unmerge  →  [base_12, base_7, base_3]

z_0 = node.z
  intuition_head(z_0, base_12)  →  z_1
  intuition_head(z_1, base_7)   →  z_2
  intuition_head(z_2, base_3)   →  z_3 = z_child
```

Then it runs one more step from `z_child` to get `z_next_pred` for the action
planner, which populates the child's candidate set (enabling depth > 1):

```
intuition_head(z_child, token_id)  →  z_next_pred
action_planner(z_child, z_next_pred)  →  child.candidate_ids
```

### 3. Token embedding lookup (metacontroller feature)

The `token_embed` table is also read (without a full forward pass) in
`search_tree.search_step` to give the metacontroller embeddings of the current
candidate tokens.  See the metacontroller README for details.

## The Token Embedding Table

`token_embed` is `nn.Embedding(vocab_size, 32)` — a lookup table mapping every
integer token ID to a 32-dimensional float vector.

Token IDs are arbitrary integers assigned during BPE tokenization.  The raw
number carries no meaning.  The embedding translates each ID into a continuous
vector that the network can actually learn from.

During training, gradients flow back through the embedding lookup.  Tokens that
produce **similar world effects** end up with **similar embedding vectors**:

```
rs0_ls0_ft9_b0  (full throttle straight)   ─┐ embeddings drift
rs0_ls0_ft8_b0  (near-full throttle)       ─┘ together

rs0_ls6_ft5_b0  (moderate left steer)      ─┐ embeddings drift
merged left-curve token                    ─┘ together

rs0_ls0_ft0_b9  (full brake)               — drifts away from throttle tokens
```

Token 0 (`rs0_ls0_ft0_b0`) is the **idle token** — used on the first frame when
no previous action exists yet.

## Training

The intuition head trains on real GTA rollouts:

```
loss = MSE(z_next_pred, z_{t+1}_real)
```

where `z_{t+1}_real` is the encoder's output on the next actual GTA frame.

It trains **separately** from the action planner, metacontroller, and reward
head.  After training, the weights are frozen and used as-is by the other
modules.  Gradients from those modules are blocked with `.detach()` so they
never flow back into the intuition head.

## Inputs and Outputs

```python
z_next_pred, delta_z, token_embed, intuition_mlp = intuition_head(
    z_t,
    prev_token_id,
    vocab_size    = 874,
    token_embed   = token_embed,   # pass None to create fresh
    intuition_mlp = intuition_mlp, # pass None to create fresh
    embed_dim     = 32,
    hidden_dim    = 128,
)
```

| Parameter | Shape | Description |
|---|---|---|
| `z_t` | `[batch, fused_dim]` | Current world embedding from main_model |
| `prev_token_id` | `[batch]` long | Token that was just executed |
| `vocab_size` | int | Size of the token vocabulary |
| `token_embed` | `nn.Embedding` or None | Token embedding table — created if None |
| `intuition_mlp` | `nn.Sequential` or None | Prediction MLP — created if None |
| `embed_dim` | int (default 32) | Token embedding dimension |
| `hidden_dim` | int (default 128) | Hidden layer width |

| Output | Shape | Description |
|---|---|---|
| `z_next_pred` | `[batch, fused_dim]` | Predicted next world embedding |
| `delta_z` | `[batch, fused_dim]` | Predicted change (`z_next_pred - z_t`) |
| `token_embed` | `nn.Embedding` | Updated embedding table (pass back in) |
| `intuition_mlp` | `nn.Sequential` | Updated MLP (pass back in) |
