# Metacontroller

The metacontroller is the brain that runs MCTS-style tree search while the current
token drives the car, and decides when to commit to the best plan found.

## The Big Picture

The current token is already executing in GTA.  While it plays, the metacontroller
uses those frames to search for the best next token by exploring a tree of predicted
futures.

```
  GTA world
      |
      v
  main_model encodes everything into z_t [128]
      |
      v
  action_planner proposes top-k candidate tokens (A, B, C)
      |
      v
  +-------------------------------------------------------+
  |                   FRAME LOOP                           |
  |                                                        |
  |  each frame, two things happen at the same time:       |
  |                                                        |
  |  executor plays one frame       search_tree thinks     |
  |  of the current token           one step about the     |
  |  (sends controls to GTA)        next token             |
  |                                                        |
  |  when current token ends (or INTERRUPT fires):         |
  |    use best_path[0] from the search tree               |
  |    fallback: planner's top-1                           |
  +-------------------------------------------------------+
      |
      v
  trainer learns from what happened
  (realized return, metalevel credit assignment, reward head update)
```

## Files

```
metacontroller/
  metacontroller.py   decision MLP (EXPLORE / INTERRUPT / COMMIT_NEXT / ROLLBACK)
  search_tree.py      MCTS tree nodes, expansion, best-path tracking, init/step interface
  time_context.py     external timing signals (urgency, budget, elapsed ratio, token_frames_left)
  executor.py         per-frame token runner — unmerges BPE token, plays frame by frame
  reward.py           pure math reward formula (ground truth, used by executor)
  frame_loop.py       main driver — interleaves executor + search per frame, triggers training
  trainer.py          computes returns, metalevel credit assignment, policy gradient update,
                      reward head training
```

## The Frame Loop

The current token buys thinking time.

```
current token = merged_token_500, duration = 20 frames

frame 1:   executor plays frame 1   |   search: EXPLORE(A) → descend into A
frame 2:   executor plays frame 2   |   search: EXPLORE(A3) → descend into A3
frame 3:   executor plays frame 3   |   search: ROLLBACK → back to A
frame 4:   executor plays frame 4   |   search: EXPLORE(A2) → descend into A2
...
frame 20:  executor plays frame 20  |   search: COMMIT_NEXT → commit best_path[0]
```

The metacontroller is NOT choosing the current frame's action.  The current token
handles that.  The metacontroller is choosing: **what should I execute AFTER this token?**

## How the Search Tree Works

Each token duration builds a fresh MCTS tree.  One search step per frame.

### Step 1: Action planner gives root candidates

```
         [root: z_t]
        /     |      \
   token_A  token_B  token_C     <-- top-k from action planner
   (unopened) (unopened) (unopened)
```

### Step 2: Each expansion does three things

**a) Intuition head predicts z_child.**

Merged BPE tokens are unpacked first, then the intuition head rolls through each
base token:

```
merged token 500 → unmerge → [base_12, base_7, base_3]

z_0 = node.z
   intuition_head(z_0, base_12) → z_1
   intuition_head(z_1, base_7)  → z_2
   intuition_head(z_2, base_3)  → z_3 = z_child
```

**b) Reward head NN scores the transition.**

```
reward_head(z_parent, z_child, rf_parent, rf_child, duration, time_left) → r_edge
```

rf (reward features) = [wp_dist, hp, v_engine_hp, v_body_hp, road_dist, dead].
These explicit signals are what actually determines reward — the network does not
have to discover them buried in a 128-dim embedding.

For root nodes, rf comes from the real GTA state.  For child nodes, rf is predicted
by `predict_reward_features(z_parent, delta_z, rf_parent) → rf_child`.

**c) Action planner runs on z_child to populate the child's candidate set.**

This is what enables depth > 1.  Every child gets its own top-k candidates, so the
metacontroller can EXPLORE deeper:

```
expand(A) → z_A predicted → action_planner(z_A) → [A1, A2, A3]
expand(A3) → z_A3 predicted → action_planner(z_A3) → [A31, A32, A33]
```

### Step 3: Metacontroller decides

After each expansion, the metacontroller picks one of four decisions.

## The Four Decisions

### EXPLORE

Expand the next candidate at the current node and **descend into it**.

```
Before:                        After EXPLORE(A):
     [root]                          [root]
    /   |   \                       /   |   \
   A    B    C                    [A]   B    C
                                  / \
                               [A1] A2, A3  <-- now at A, next: A2
                              current_node = A
```

The tree grows deeper.  Next step will expand A2 or go deeper into A1.

### ROLLBACK

Go back up to the parent.  Try a sibling on the next step.

```
Before (current_node = A3):        After ROLLBACK:
     [root]                              [root]
    /   |   \                           /   |   \
  [A]   B    C                        [A]   B    C
  / \                                 / \
[A1][A2][A3]                        [A1][A2][A3]
         ^                           current_node = A (next: nothing, auto-rollback to root)
         current_node
```

Use when the current branch looks worse than the best plan found so far.

### INTERRUPT

Stop the current GTA token early.  Switch to `best_path[0]` immediately.

Use when: the current token is going badly and waiting is not worth the risk.

### COMMIT_NEXT

Finish the current token, then execute `best_path[0]`.

Use when: search found a good enough plan and there is no reason to rush.

## Best Path Tracking

Every expansion updates the best path found so far:

```python
path_value = sum of r_edge from root down to the new leaf
if path_value > best_path_value:
    best_path = [A, A3, A32]   # full token sequence from root to leaf
    best_path_value = path_value
```

`best_path[0]` is always the root-level token to commit to GTA.  The rest of the
path is the predicted future that justified picking it.

**This closes the learning loop:** the metacontroller's EXPLORE/ROLLBACK decisions
determine what `best_path` ends up being.  When it commits, `best_path[0]` is
executed.  The realized reward trains the metacontroller on whether its exploration
was good.

## Metacontroller Features

The decision MLP sees these inputs every step:

**World state**

| Feature | Shape | Meaning |
|---|---|---|
| `drift` | `[batch, fused_dim]` | `z_t - z_running` — how wrong the prediction was when this token was committed |
| `elapsed_ratio` | `[batch, 1]` | How far into the current token (0=start, 1=done) |
| `token_frames_left` | `[batch, 1]` | Raw frames left before the current token ends |
| `urgency` | `[batch, 1]` | Time pressure from the planning deadline |

**Root-level candidate quality** (what can be committed to GTA)

| Feature | Shape | Meaning |
|---|---|---|
| `best_q` | `[batch, 1]` | Best Q-value among root children |
| `mean_q` | `[batch, 1]` | Average Q-value among root children |

**Branch quality** (where the search currently is)

| Feature | Shape | Meaning |
|---|---|---|
| `parent_unexplored` | `[batch, 1]` | Fraction of current node's candidates still unexpanded |
| `current_path_value` | `[batch, 1]` | Cumulative r_edge from root down to current_node |
| `best_path_value` | `[batch, 1]` | Best cumulative r_edge found so far in this search |

`current_path_value` vs `best_path_value` is the key EXPLORE vs ROLLBACK signal:
- `current_path_value > best_path_value` → this branch is beating the current best → EXPLORE deeper
- `current_path_value << best_path_value` → this branch is bad → ROLLBACK

**Current node's candidates** (what tokens are on the table right now)

| Feature | Shape | Meaning |
|---|---|---|
| `best_current_q` | `[batch, 1]` | Best Q among current node's expanded children |
| `mean_current_q` | `[batch, 1]` | Average Q among current node's expanded children (0 for unexpanded) |
| `current_candidate_durations` | `[batch, top_k]` | Execution cost of each candidate in frames |
| `current_candidate_emb` | `[batch, top_k × token_embed_dim]` | Token embeddings of the candidates — what *kind* of actions are available |

`current_candidate_emb` comes from the intuition head's `token_embed` table (an `nn.Embedding` of
dim 32).  See the full explanation below.

`current_candidate_durations` matters because a 30-frame token carries more risk than a 5-frame
one — the world can change a lot in 30 frames, so the metacontroller should be more willing to
EXPLORE before committing to a long token.

Total feature dimension: `fused_dim + 10 + top_k × (1 + token_embed_dim)`.
With defaults (fused_dim=128, top_k=3, token_embed_dim=32): **237**.

---

## Token Embeddings — What They Are and Why They Matter

### What a token actually is

Every token in the vocabulary is a **multi-frame control chunk**: a sequence of
quantized GTA control frames, each with four fields:

```
right_steer    0–9  (10 bins over [0.0, 1.0])
left_steer     0–9
forward_throttle 0–9
brake          0–9
```

Base tokens are single frames — one discrete control snapshot.  The BPE merge
step fuses frequently co-occurring base tokens into longer merged tokens.  A
merged token might be 8 frames of "full throttle, 0.33 right steer" — a
short highway straight.  Another might be 15 frames of "left steer ramping
up then easing off" — a smooth left curve.

### Token IDs are arbitrary integers

Token IDs are assigned in the order they are created during BPE:
- Token 0: idle frame (rs0_ls0_ft0_b0)
- Token 1: first unique base frame seen
- Token 500: the 500th entry added — could be anything

The numbers carry **no semantic meaning**.  Token 500 is not "more" than
token 100.  You cannot do arithmetic on them.  The metacontroller cannot
learn anything useful from the raw integer IDs.

### The embedding table

`token_embed` is `nn.Embedding(vocab_size, 32)` inside the intuition head.
It is a lookup table: every token ID maps to a **32-dimensional float
vector**.  At first these are random.  During intuition head training,
gradient descent adjusts them.

The intuition head is trained to predict the next world embedding:

```
intuition_head(z_t, prev_token_id) → z_next_pred ≈ real z_{t+1}
```

The gradient flows back through the token embedding lookup.  For the
prediction to be accurate, the embedding of `prev_token_id` must encode
something useful about what that token *does* to the world.

Over time, tokens that produce **similar world effects** end up with
**similar embedding vectors**:

```
token 237  rs0_ls0_ft9_b0  (full throttle, straight)
token 241  rs0_ls0_ft8_b0  (near-full throttle, straight)
  → embeddings drift together — both "go fast straight"

token 88   rs0_ls6_ft5_b0  (moderate left steer, half throttle)
token 312  merged: [ls5_ft5] + [ls7_ft4]  (left curve, decelerating)
  → embeddings drift together — both "turn left"

token 15   rs0_ls0_ft0_b9  (full brake)
  → embedding drifts away from throttle tokens — opposite world effect
```

### What the metacontroller sees

When three candidates are available at the current node, their embeddings
are looked up and **flattened**:

```
candidate_0 embedding  [32 dims]  ─┐
candidate_1 embedding  [32 dims]  ─┼─→  current_candidate_emb  [96 dims]
candidate_2 embedding  [32 dims]  ─┘
```

The metacontroller MLP receives all 96 numbers.  It can learn patterns that
pure Q-value summaries cannot capture:

| Situation | What embeddings reveal | Useful decision |
|---|---|---|
| All 3 candidates cluster near "full throttle straight" | Tokens are interchangeable — any will do | COMMIT quickly |
| Candidates are spread across brake / steer / throttle regions | Tokens have very different effects — worth checking which the reward head prefers | EXPLORE more |
| One candidate is near the "brake" cluster | Something dangerous may be nearby — reward head may score it very differently | EXPLORE that candidate before committing |
| Candidates all cluster near a token from a previous crash | The agent has been in this situation before | Prefer ROLLBACK to a different branch |

The Q values tell the metacontroller *how good* each option looks right now.
The embeddings tell it *what kind* of action each option is — context that
matters when Q values are close or when candidates haven't been expanded yet
(Q = 0 for unexpanded nodes, so embeddings are the only signal available).

## Time Context

Updated every frame from raw frame counts:

```
current_frame = 1042        →  elapsed_ratio = 0.6
deadline_frame = 1060           urgency = 0.7
token_start_frame = 1030        budget_remaining = 4
token_duration = 20 frames      token_frames_left = 8
nodes_expanded = 6
max_budget = 10
```

`token_frames_left` tells the metacontroller how much real execution time is left
before it MUST have a decision ready.  As this approaches zero, urgency forces a commit.

## The Executor

Per-frame token runner.  Plays ONE frame at a time so the frame loop can interleave
search alongside it.

```
execution_init(token_id, token_table, unmerge_fn, read_state_fn)
  → unmerges BPE token → flattened per-frame control schedule

execution_frame()  ← called once per frame by the frame loop
  → send controls to GTA
  → read new state
  → compute reward (math formula from reward.py)
  → record to rollout

get_rollout()  ← called when token is done
  → returns per-frame rewards, states, duration
```

`reward.py` is pure math (distance progress, collision, off-road, waypoint).
It runs on real GTA states to produce ground-truth per-frame rewards.
It is completely separate from the reward head NN, which runs on embeddings
inside the search tree.

## The Trainer

Learning happens after each token execution.

### 1. Realized Token Return

```
R_token = r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^(k-1)*r_{k-1}
        + gamma^k * V(next_state)   (optional bootstrap)
```

### 2. Tree Backup

The committed branch's N/W/Q is updated with the real return so the tree
reflects what actually happened.

### 3. Metalevel Credit Assignment

Every search decision gets credit based on the final outcome:

```
step 0: EXPLORE(A)     → -think_cost
step 1: EXPLORE(A3)    → -think_cost
step 2: ROLLBACK       → -think_cost
step 3: EXPLORE(B)     → -think_cost
step 4: COMMIT_NEXT    → R_token
```

Discounted returns are computed backward.  Advantage = return - predicted_Q.

- If ROLLBACK led to finding a better branch, it gets positive advantage
- If excessive EXPLOREs explored a dead end, they get negative advantage
- The metacontroller learns to search efficiently

### 4. Policy Gradient Update

REINFORCE over the full metalevel trajectory.  Features (not logits) are stored
per step so gradients flow through the live meta_mlp during training.

### 5. Reward Head Training

After each token, the reward head and rf_predictor are trained against the
realized return:

- `reward_mlp` loss: predicted `r_edge` vs actual `token_return` (MSE)
- `rf_predictor` loss: predicted `rf_child` vs real `rf_child` from GTA (MSE)

## What Talks to What

```
action_planner
     |
  top-k candidates [A, B, C]
     |
     v
+------------------------------------------------------------+
|                    FRAME LOOP                              |
|                                                            |
|  per frame:                                                |
|                                                            |
|  executor.execution_frame()      search_tree.search_step() |
|       |                               |                    |
|       +-> GTA (send controls)         +-> intuition head   |
|       +-> reward.compute_reward       +-> reward head NN   |
|       +-> record to rollout           +-> action planner   |
|                                       +-> metacontroller   |
|                                       +-> meta_trajectory  |
|                                                            |
|  time_context refreshed every frame                        |
+------------------------------------------------------------+
     |                             |
     v                             v
  rollout                    search_state
  (rewards, states)          (meta_trajectory, root,
                              best_path, best_path_value,
                              chosen_token_id)
     |                             |
     +-------------+---------------+
                   |
                   v
               trainer
                   |
                   +-> compute_token_return
                   +-> backup_tree
                   +-> compute_metalevel_advantages
                   +-> update_metapolicy  (policy gradient)
                   +-> train_reward_head  (MSE on r_edge + rf_predictor)
```

## Drift: Why INTERRUPT Exists

When the metacontroller committed to the current token, the intuition head
predicted what the world would look like (`z_running`).  The real world keeps
changing.

```
drift = z_t - z_running

small drift  →  prediction was good, no need to switch
large drift  →  world changed unexpectedly, consider INTERRUPT
```

The metacontroller sees drift as the first feature and learns when a large
drift means the current action is no longer appropriate.
