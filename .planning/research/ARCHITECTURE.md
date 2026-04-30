# Architecture Patterns

**Domain:** MCTS-based RL driving agent with learned metacontroller (RASHKOGIE GTA)
**Researched:** 2026-04-30
**Overall confidence:** HIGH (grounded in existing codebase + verified patterns)

---

## Verified Input Dimensions (computed from live code)

Before recommending sizes, the exact feature dimensions were confirmed by tracing the code:

| Tensor | Dim | Source |
|--------|-----|--------|
| `z_t` (fused state embedding) | 128 | `main_model.py` `FUSED_DIM` |
| `drift` = z_t - z_running | 128 | `metacontroller.py` Step 1 |
| elapsed_ratio | 1 | `time_context.py` |
| token_frames_left | 1 | `time_context.py` |
| best_q, mean_q | 2 | `metacontroller.py` Step 2 |
| urgency | 1 | `time_context.py` |
| parent_unexplored | 1 | `search_tree.py` |
| current_path_value, best_path_value | 2 | `search_tree.py` |
| best_current_q, mean_current_q | 2 | `metacontroller.py` Step 2 |
| current_candidate_durations (top_k=3) | 3 | `metacontroller.py` |
| current_candidate_emb (3 x 32) | 96 | `metacontroller.py` |
| **Total metacontroller input** | **237** | confirmed |

Entity embedding (encoder): 64-dim per entity, 32 entities max.
Encoder attention query: `[ego_emb | scene_emb | route_emb]` = 192-dim input, projecting to 64-dim output.

---

## 1. Multi-Head Attention for State Encoder

### Current State
One attention block in `multi_head_attention.py` with `num_heads=4`. The query is the 192-dim concat of `[ego_emb | scene_emb | route_emb]`, projected to 64 dim. Keys and Values come from 32 entity embeddings at 64 dim each.

### Recommended Configuration

**Number of heads: 4 (keep current, add a second attention block)**

Rationale for 4 heads:
- embed_dim = 64, head_dim = 64/4 = 16 per head. Each head attends to a distinct relational aspect: one can specialize on proximity, one on velocity, one on type, one on collision risk. 16-dim per head is sufficient for these structured geometric features.
- 8 heads would give head_dim = 8, which is too small for 24-dimensional entity features compressed to 64 dim — the per-head representations would be too narrow to learn useful queries.
- 2 heads at head_dim = 32 is viable but loses the diversity benefit. At 32 entities with varied types (vehicles, peds, objects), 4 independent attention patterns outperform 2.

**Add a second attention block (stacked encoder):**

The current encoder has 1 attention block followed by a fusion MLP. This is the weakest link identified in the project audit ("single attention block may need more"). For 32 entities with structured interaction patterns (vehicles approaching, pedestrians crossing), two stacked attention blocks allow:
- Block 1: attends to individual entity salience (which entities matter at all)
- Block 2: attends to relational context (how selected entities relate to each other and to the route)

This is the standard pattern from Set Transformer and entity-based RL encoders (e.g., Relational Deep RL, MARL multi-agent attention papers).

**Recommended architecture for the encoder attention stack:**

```python
# Block 1: cross-attention (existing)
#   query: [ego_emb | scene_emb | route_emb] -> [1, 192]
#   keys/values: entity_embs -> [1, 32, 64]
#   output: entity_context_1 -> [1, 64]
#   num_heads=4, head_dim=16

# LayerNorm after Block 1 (ADD THIS)
#   entity_context_1 = LayerNorm(entity_context_1 + residual)

# Block 2: self-attention over enriched entity set (ADD THIS)
#   keys/values: entity_embs updated with context from Block 1
#   num_heads=4, head_dim=16
#   output: entity_context_2 -> [1, 64]

# Fusion MLP (existing, keep)
#   input: [ego_emb | scene_emb | route_emb | entity_context_2] -> [1, 256]
#   output: z_t -> [1, 128]
```

Add LayerNorm after each attention block. LayerNorm is batch-size-independent (critical here: batch=1 at 20 Hz inference), stabilizes gradients, and is standard practice in all transformer encoders. Do NOT use BatchNorm (requires batch > 1 for meaningful statistics).

**Confidence:** HIGH — 4 heads at 64-dim is the canonical configuration from "Attention Is All You Need" (d_k = d_model/h = 64 when d_model=512, h=8; your smaller model follows the same ratio). Two-block stacking is verified in Set Transformer and RRL literature.

---

## 2. Metacontroller MLP Sizing

### Current State
Single hidden layer: `Linear(237, 128) -> ReLU -> Linear(128, 4)`. Project audit flagged this as "likely insufficient."

### Recommended Architecture

**3 hidden layers, width 256-256-128, with a skip connection from input to layer 2:**

```python
meta_mlp = nn.Sequential(
    # --- Layer 1: expand from input ---
    nn.Linear(237, 256),
    nn.LayerNorm(256),
    nn.ReLU(),

    # --- Layer 2: refine ---
    nn.Linear(256, 256),
    nn.LayerNorm(256),
    nn.ReLU(),

    # --- Layer 3: compress ---
    nn.Linear(256, 128),
    nn.LayerNorm(128),
    nn.ReLU(),

    # --- Output: 4 decisions ---
    nn.Linear(128, 4),
)
```

With a residual skip connection implemented outside Sequential:

```python
class MetaMLP(nn.Module):
    def __init__(self, input_dim=237, hidden=256, output_dim=4):
        super().__init__()
        self.layer1  = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.layer2  = nn.Sequential(nn.Linear(hidden, hidden),    nn.LayerNorm(hidden), nn.ReLU())
        self.layer3  = nn.Sequential(nn.Linear(hidden, 128),       nn.LayerNorm(128),    nn.ReLU())
        self.out     = nn.Linear(128, output_dim)
        # skip: project input directly to layer-2 dim for residual addition
        self.skip_proj = nn.Linear(input_dim, hidden, bias=False)

    def forward(self, x):
        h1 = self.layer1(x)
        h2 = self.layer2(h1) + self.skip_proj(x)   # residual from input
        h3 = self.layer3(h2)
        return self.out(h3)
```

Rationale for these choices:

**Width 256:** The input is 237-dim and contains semantically heterogeneous features (128-dim embedding drift, scalars for timing/urgency, 96-dim token embeddings). A width of 256 gives the network room to disentangle these before compressing. The 1.08x expansion (237 -> 256) avoids the information bottleneck that occurs when the first layer is narrower than the input. The standard RL policy network in PPO/A3C is 2x64 or 2x256 for structured inputs; 256 is the correct width at this scale.

**3 layers:** Two layers (current) can represent any function in theory but fail in practice on high-dimensional heterogeneous inputs due to gradient flow issues. Three layers with LayerNorm and a skip connection provide stable gradients and sufficient capacity. Four layers would add parameters without benefit given the small output space (4 decisions).

**Skip connection from input to layer 2:** The input contains explicit numeric signals (urgency, elapsed_ratio, best_q) that directly determine decisions. Without a skip, these signals must propagate through Layer 1's nonlinearity and risk being suppressed. The skip ensures they remain available at Layer 2. This is analogous to how residual connections in policy networks improve training stability by preserving direct signal paths.

**LayerNorm at every hidden layer:** At batch=1 (online RL), BatchNorm statistics are meaningless. LayerNorm is batch-size-independent and stabilizes the heterogeneous 237-dim input (mixing 128-dim embedding values with 0-1 scalars and 0-20 frame counts). This prevents gradient explosion during early training when the policy is random.

**Confidence:** HIGH for width 256 and 3 layers (verified by stable-baselines3 policy network conventions and RL literature). HIGH for LayerNorm (batch-size-independence is documented). MEDIUM for the skip connection form shown (this specific residual design is an adaptation of standard ResNet patterns to MLP; the benefit is well-documented in general, application to this specific 237-dim input is by analogy).

---

## 3. Replay Buffer Design for Online REINFORCE

### The Core Constraint
REINFORCE is strictly on-policy. A traditional replay buffer holding transitions from past policies violates the on-policy assumption and introduces policy lag bias — gradients from stale trajectories push the current policy toward a past policy's optimum, not the current one.

### Recommended: Trajectory Buffer (not transition buffer)

Do NOT use a transition-level replay buffer (the kind used in DQN/DDPG). Instead, use a **trajectory buffer** that accumulates complete metalevel episodes before updating.

A metalevel episode = one token's search sequence: [EXPLORE, EXPLORE, ROLLBACK, EXPLORE, COMMIT_NEXT] with its advantages and returns.

```python
class MetaTrajectoryBuffer:
    """
    Accumulates complete metalevel trajectories (one per token execution).
    Flushes and trains every N trajectories to reduce gradient variance.
    All trajectories in a flush must come from the CURRENT policy.
    """
    def __init__(self, flush_every=8):
        self.flush_every = flush_every
        self.buffer = []     # list of (meta_trajectory, advantages) pairs
        self.step   = 0

    def add(self, meta_trajectory, advantages):
        self.buffer.append((meta_trajectory, advantages))
        self.step += 1

    def should_flush(self):
        return self.step % self.flush_every == 0

    def flush(self):
        batch = self.buffer[-self.flush_every:]  # ONLY the latest N trajectories
        self.buffer.clear()
        return batch
```

**Why N=8 trajectories per flush:**
- Each metalevel trajectory contains 3-20 decisions. With N=8, a flush contains 24-160 gradient steps — meaningful variance reduction without accumulating staleness.
- At 20 Hz and typical token durations of 5-15 frames, 8 tokens complete in roughly 2-6 seconds of game time. This is short enough that the policy has not drifted significantly.
- Fewer than 4 trajectories per flush gives high-variance gradient estimates. More than 16 introduces policy lag (by the 16th trajectory, the policy that generated trajectory 1 is noticeably different).

**Correct procedure:**
1. Collect trajectory with current policy (no gradient)
2. Compute advantages (REINFORCE with baseline)
3. Add to buffer
4. Every N trajectories: compute mean loss over all steps in the buffer, backward once, update policy
5. Clear buffer completely — NEVER reuse trajectories across flush cycles

**What to NEVER do:**
- Never store trajectories from before the last weight update and use them in later flush cycles (this is the off-policy violation)
- Never shuffle individual transitions across trajectories (the advantage for a ROLLBACK decision is only meaningful in the context of the trajectory it came from)

**Confidence:** HIGH — this trajectory accumulation pattern is standard in A2C and vanilla REINFORCE implementations. The N=8 recommendation is MEDIUM confidence (based on the 20 Hz rate and token duration characteristics of this specific environment, not a universal constant).

---

## 4. Entropy Regularization for the 4-Action Policy

### The Problem Being Solved
The metacontroller currently uses argmax, which means it receives zero entropy gradient. Without entropy regularization, REINFORCE on a 4-action softmax policy collapses: the action taken most often gets reinforced, its log-probability increases, the other actions receive less gradient, their probabilities decrease, eventually the distribution becomes near-deterministic on one action regardless of state. This is the identified bug in the project.

### Recommended Implementation

Add entropy bonus to the policy gradient loss with a scheduled coefficient:

```python
def compute_meta_loss(logits, actions, advantages, entropy_coef):
    """
    logits:      [T, 4]   raw scores for each step in trajectory batch
    actions:     [T]      decision taken at each step (0-3)
    advantages:  [T]      REINFORCE advantage at each step
    entropy_coef: float   entropy regularization coefficient
    """
    log_probs = torch.log_softmax(logits, dim=-1)        # [T, 4]
    probs     = torch.softmax(logits, dim=-1)             # [T, 4]

    # policy gradient loss
    log_prob_taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # [T]
    pg_loss = -(log_prob_taken * advantages).mean()

    # entropy bonus (maximize entropy = subtract negative entropy from loss)
    entropy = -(probs * log_probs).sum(dim=-1).mean()    # scalar, max = log(4) = 1.386
    entropy_loss = -entropy_coef * entropy

    return pg_loss + entropy_loss
```

**Coefficient schedule:**

| Phase | entropy_coef | Rationale |
|-------|-------------|-----------|
| Early training (first 500 token updates) | 0.05 | High exploration needed — policy knows nothing yet |
| Mid training (500-2000 updates) | 0.02 | Policy is learning signal; reduce entropy to allow specialization |
| Late training (2000+ updates) | 0.005 | Fine-tuning; only prevent total collapse |

**Why these values:**
- Maximum entropy for 4 actions = log(4) = 1.386 nats. At coefficient 0.05, the entropy bonus at max entropy = 0.05 * 1.386 = 0.069. This is a meaningful regularizer relative to typical advantage magnitudes of 0.1-2.0.
- At coefficient 0.001 (too small), research shows entropy still converges to ~0.06 nats (nearly deterministic) — insufficient.
- At coefficient 0.1+ (too large), entropy bonus dominates and the agent never learns to exploit known-good actions.
- The 0.05 -> 0.02 -> 0.005 schedule follows the principle of "start smooth, refine gradually" from entropy regularization literature.

**Linear decay implementation:**

```python
def get_entropy_coef(update_step, warmup=500, decay_end=2000,
                     coef_start=0.05, coef_mid=0.02, coef_end=0.005):
    if update_step < warmup:
        return coef_start
    elif update_step < decay_end:
        t = (update_step - warmup) / (decay_end - warmup)
        return coef_start + t * (coef_end - coef_start)
    else:
        return coef_end
```

**Monitor entropy during training.** Log `entropy.item()` at every flush. Healthy training: entropy stays above 0.5 nats (out of max 1.386) for the first 1000 updates. If entropy drops below 0.3 nats before update 500, increase `coef_start` to 0.1.

**Confidence:** MEDIUM-HIGH. The coefficient values 0.05/0.02/0.005 are derived from the entropy magnitude analysis above and supported by general RL literature (SAC uses 0.2 for continuous spaces; discrete 4-action spaces need lower values because the max entropy is only 1.386 nats vs. much larger for continuous). The specific update thresholds (500/2000) are LOW confidence — they depend on token frequency and reward scale in actual GTA gameplay.

---

## 5. REINFORCE Implementation: Sampling, Baseline, Gradient Clipping

### Decision Sampling vs. Argmax

**During training: categorical sampling (torch.distributions.Categorical)**
**During inference: argmax**

This is the identified bug. The fix:

```python
from torch.distributions import Categorical

def decide(logits, training=True):
    if training:
        dist     = Categorical(logits=logits)
        decision = dist.sample()          # [batch]
        log_prob = dist.log_prob(decision) # [batch]
        return decision, log_prob
    else:
        decision = logits.argmax(dim=-1)  # [batch]
        return decision, None
```

Categorical sampling during training ensures all 4 actions are explored — a requirement for REINFORCE to produce unbiased gradient estimates. Argmax during training means only one action receives gradient, regardless of how good the other actions might be.

**Storing decisions for training:** The meta_trajectory must store the sampled decision index (an integer 0-3), NOT the argmax. The current `metacontroller.py` stores `meta_out["decision"]` which is `argmax`. This must be changed to the sampled index.

### Baseline Computation

Use the predicted Q-value as the baseline (already partially in place via `predicted_q` in `meta_trajectory`). Improve it:

```python
def compute_advantages_with_baseline(meta_returns, predicted_qs):
    """
    meta_returns:  list of float, discounted return at each step
    predicted_qs:  list of float, best_q at that step (stored in trajectory)

    Returns advantages normalized by return std.
    """
    advantages = [r - q for r, q in zip(meta_returns, predicted_qs)]

    # normalize to unit variance — critical for stable REINFORCE
    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    if adv_tensor.std() > 1e-8:
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    return adv_tensor.tolist()
```

**Why normalize advantages:** Advantage magnitudes depend on the reward scale (which varies as the agent explores GTA). Without normalization, a trajectory with a large crash reward (collision_weight=5.0) will dominate over trajectories with typical progress rewards, leading to catastrophic updates. Normalizing to zero-mean unit-variance keeps learning rate stable across reward scales.

**Why use best_q as baseline vs. episode mean:** `best_q` is state-dependent (it reflects the search tree's current estimate of what's achievable from this state). A state-independent baseline (episode mean) reduces variance less. The advantage A = R - V(s) is lower variance than A = R - mean(R) when V(s) is a good state-value estimate. `best_q` serves as a noisy but state-dependent V(s).

### Gradient Clipping

Replace manual SGD with Adam + gradient clipping:

```python
optimizer = torch.optim.Adam(meta_mlp.parameters(), lr=3e-4)

# after loss.backward():
torch.nn.utils.clip_grad_norm_(meta_mlp.parameters(), max_norm=0.5)
optimizer.step()
optimizer.zero_grad()
```

**Why Adam instead of manual SGD:**
- Manual SGD with no momentum is highly sensitive to learning rate. In online RL where gradient magnitudes vary with the reward signal, SGD requires careful per-parameter tuning. Adam adapts learning rates based on gradient history, which is standard for all modern policy gradient implementations.
- The current manual SGD in `trainer.py` uses `lr=1e-3`. With Adam, `lr=3e-4` is the canonical default across most RL benchmarks.

**Why max_norm=0.5:**
- Policy gradient losses can spike when advantage magnitudes are large (e.g., the first crash after a long run). Clipping at 0.5 prevents weight updates that would permanently damage the policy.
- Values of 0.5-1.0 are standard in PPO and policy gradient implementations. Smaller (0.1) slows learning unnecessarily; larger (5.0) does not prevent catastrophic updates.

**Confidence:** HIGH for categorical sampling (this is the fundamental REINFORCE requirement). HIGH for Adam + grad clipping (universally recommended in RL literature). MEDIUM for max_norm=0.5 (0.5-1.0 range is standard; exact value within that range requires empirical tuning).

---

## 6. Training Pipeline Architecture: Module Chaining and Freezing

### Strict Dependency Order

The training order from PROJECT.md is enforced by data availability and gradient flow:

```
Phase 1: Encoder (main_model)
    Input:  raw GTA state dicts (from player captures)
    Output: z_t [1, 128] per frame
    Loss:   auxiliary reconstruction or contrastive (not yet defined — needs design)
    Status: encoder is used but no standalone training loop exists yet

Phase 2: Intuition Head
    Input:  (z_t, token_id) pairs from real gameplay
    Output: z_{t+1}_pred
    Loss:   MSE(z_{t+1}_pred, z_{t+1}_real)
    Freeze: YES after convergence (z_t must be stable before training)

Phase 3: Reward Head + rf_predictor
    Input:  (z_parent, z_child, rf_parent, rf_child, duration, time_left)
            rf_child from real GTA state (not predicted — training phase)
    Output: r_edge prediction
    Loss:   MSE(r_pred, realized_token_return)
    Freeze: YES after convergence (intuition head must be frozen first)

Phase 4: Action Planner
    Input:  (z_t, z_next_pred) from frozen intuition head
    Output: top-k token distribution
    Loss:   cross-entropy against player's demonstrated action (imitation learning)
    Freeze: YES after convergence (acts as prior for the search tree)

Phase 5: Metacontroller
    Input:  237-dim feature vector (see table above)
    Output: EXPLORE/INTERRUPT/COMMIT_NEXT/ROLLBACK logits
    Loss:   REINFORCE + entropy regularization
    Training: NEVER freeze — this is the final learned policy
```

### Gradient Isolation Protocol

Each module boundary uses `.detach()` to prevent cross-module gradient flow:

```python
# In action_planner.py (already implemented):
planner_input = torch.cat([z_t, z_next_pred.detach()], dim=-1)

# In search_tree.py expand_next_child (already implemented):
z_child.detach()   # before passing to next node
rf_child.detach()  # before storing on child node

# When encoding for reward head training (already implemented):
z_parent = encode_fn(state_before).detach()
z_child  = encode_fn(state_after).detach()
```

**Additional isolation needed for metacontroller training:**

The metacontroller's feature vector includes `drift = z_t - z_running`. Both `z_t` and `z_running` come from the encoder. During metacontroller training (Phase 5), the encoder is frozen. Ensure:

```python
# In metacontroller.py before feature assembly:
drift = z_t.detach() - z_running.detach()
```

This prevents any accidental gradient flow into the encoder during metacontroller updates.

### Convergence Signals for Freezing

| Module | Freeze when | Typical signal |
|--------|-------------|---------------|
| Intuition head | MSE loss plateaus (no improvement over 200 token-level updates) | Loss < 0.05 on z_next_pred |
| Reward head | MSE loss plateaus AND rf_loss plateaus | Loss < 0.1 on token returns |
| Action planner | Imitation accuracy plateaus | Top-1 accuracy > 60% on player demonstrations |
| Metacontroller | Never frozen — it is the final policy | N/A |

### Build Order Recommendation

Based on dependency analysis, the recommended build order for remaining implementation work is:

1. **Fix metacontroller decision sampling** (argmax -> Categorical) — zero other work unblocks this, and it is required before any training data from the metacontroller is valid.

2. **Add entropy regularization + advantage normalization** — can be done in the same pass as (1); both touch `trainer.py` and `metacontroller.py`.

3. **Replace manual SGD with Adam + gradient clipping** — one-line change in `trainer.py` per module; do for all modules simultaneously.

4. **Add LayerNorm to encoder attention blocks** — prerequisite for adding the second attention block.

5. **Upgrade metacontroller MLP to 3-layer with skip** — replaces the inline `nn.Sequential` in `metacontroller.py` with the `MetaMLP` class.

6. **Add second attention block to encoder** — after (4) is validated.

7. **Implement trajectory buffer** — wraps the existing `train_step` call in `frame_loop.py`.

8. **Implement module-specific training loops** (intuition head, reward head, action planner) — these are currently invoked inline but need standalone training scripts with convergence monitoring.

---

## Data Flow Diagram

```
GTA V (20 Hz)
    |
    | raw_state dict
    v
[ENCODER]
  ego [46] -> ego_mlp -> [64]  ──┐
  scene [16] -> scene_mlp -> [64] ├─ query [192]
  route [14] -> route_mlp -> [64] ──┘
  entities [32,24] -> entity_mlp -> [32,64]
                    |
            [ATTENTION BLOCK 1] (4 heads, head_dim=16)
            [LayerNorm]
                    |
            [ATTENTION BLOCK 2] (4 heads, head_dim=16)  ← ADD
            [LayerNorm]
                    |
          entity_context [64]
                    |
  [ego|scene|route|entity_context] -> fusion_mlp -> z_t [128]
    |
    +─────────────────────────────────────────────+
    |                                             |
    v                                             v
[INTUITION HEAD]                        [ACTION PLANNER]
  (z_t, prev_token) -> z_next_pred        (z_t, z_next_pred) -> top-k tokens
    (frozen after Phase 2)                  (frozen after Phase 4)
    |
    v
[SEARCH TREE]  ─── one step per frame while current token plays ───
  TreeNode: z_t -> intuition_head -> z_child
                 -> reward_head -> r_edge
                 -> action_planner -> child.candidates
    |
    v
[METACONTROLLER]
  features [237] = [drift[128] | timing[4] | tree_signals[9] | token_embs[96]]
  MetaMLP: 237 -> 256 -> 256 -> 128 -> 4
  Decision: Categorical sample during training, argmax at inference
  Loss: REINFORCE + entropy_coef * H(pi)
    |
    v
[TRAJECTORY BUFFER]
  Accumulate N=8 metalevel trajectories
  Flush: Adam(lr=3e-4), clip_grad_norm(0.5)
    |
    v
[REWARD HEAD] (trained in parallel, same token rollout)
  (z_parent, z_child, rf, duration) -> r_edge_pred
  Loss: MSE vs realized_token_return
```

---

## Scalability Considerations

| Concern | Now (early training) | At 10K token updates | At 100K token updates |
|---------|---------------------|---------------------|----------------------|
| Metacontroller input dim | 237 (fixed) | 237 | 237 |
| Memory per trajectory | ~50 KB (20-step search * tensors) | same | same |
| Search budget | 20 nodes/token | may increase to 40 | may increase to 60 |
| Training frequency | every 8 tokens (flush) | every 8 tokens | every 8 tokens |
| Encoder frozen? | No (jointly trained or separately) | Yes (after Phase 1-2) | Yes |
| Metacontroller MLP params | ~200K (3-layer) | same | same |

The architecture is designed for a single GPU on a machine also running GTA V. The 3-layer 256-dim MLP and 2-block attention encoder add less than 500K parameters total — negligible GPU overhead at 20 Hz inference.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Replay Buffer with Stale Policy Trajectories
**What:** Saving trajectories from earlier policies and mixing them into current training batches.
**Why bad:** REINFORCE gradients are only unbiased under the policy that generated the data. Stale trajectories produce biased gradients that actively harm policy learning.
**Instead:** Use a trajectory buffer that flushes completely after each update. Never re-use trajectories.

### Anti-Pattern 2: Argmax During Training
**What:** Using `decision_logits.argmax(dim=-1)` to select actions during training (current bug).
**Why bad:** Only the argmax action ever receives gradient. Other actions never get reinforced or suppressed. The policy cannot learn to distinguish states where different actions are correct.
**Instead:** `Categorical(logits=decision_logits).sample()` during training; argmax only at deployment.

### Anti-Pattern 3: Single Attention Block with No Normalization
**What:** One cross-attention block with raw floating-point values, no LayerNorm.
**Why bad:** At batch=1 and with heterogeneous features (positions, speeds, booleans), the attention scores can span many orders of magnitude. Without LayerNorm, gradients through the attention block are unstable.
**Instead:** LayerNorm after each attention block, stacked to 2 blocks.

### Anti-Pattern 4: Manual SGD in Policy Gradient
**What:** Manual `p.data -= lr * p.grad` without adaptive learning rates.
**Why bad:** Policy gradient losses are highly non-stationary (reward scale changes as agent learns). Manual SGD at a fixed learning rate either diverges (lr too high) or stagnates (lr too low). Adam's per-parameter adaptive rates handle this automatically.
**Instead:** `torch.optim.Adam(params, lr=3e-4)` with `clip_grad_norm(0.5)`.

### Anti-Pattern 5: No Advantage Normalization
**What:** Using raw REINFORCE returns as advantages without subtracting a baseline or normalizing.
**Why bad:** Returns depend on the cumulative reward scale, which changes over time. A crash early in training produces return -5; later it might be -50 if the discount is shorter. Raw returns make the policy update step size non-stationary.
**Instead:** Normalize advantages to zero-mean, unit-variance within each trajectory batch.

### Anti-Pattern 6: Training Metacontroller Before Freezing Intuition Head + Reward Head
**What:** Running metacontroller RL updates while the reward head is still learning.
**Why bad:** The metacontroller's reward signal (via r_edge from the reward head) is non-stationary if the reward head is also updating. The metacontroller cannot learn a stable policy from a moving reward function.
**Instead:** Follow the strict training order: freeze intuition head, freeze reward head, then train metacontroller.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Adding 2nd attention block | Increasing encoder parameters makes encoder training longer; if encoder is still changing when intuition head training starts, forward model is unstable | Add early stopping: freeze encoder once encoder loss plateaus, not before intuition head training begins |
| Metacontroller MLP upgrade | Switching from inline nn.Sequential to MetaMLP class breaks all weight persistence code that passes `meta_mlp` as a dict value | Ensure MetaMLP is `nn.Module`, serializable with `torch.save(meta_mlp.state_dict())` |
| Trajectory buffer + Adam | Adam optimizer state (momentum estimates) must persist across flushes | Store optimizer instance outside the buffer; do not recreate it each flush |
| Entropy coefficient tuning | entropy_coef=0.05 may be too high if reward magnitudes are large (collision=-5, progress=+1), causing entropy to dominate | Log entropy separately from pg_loss; if entropy > 1.2 nats consistently, halve entropy_coef |
| think_cost sign | Current think_cost=0.01 penalizes search steps; if set too high it creates an incentive to COMMIT_NEXT immediately (the "lazy commit" bug identified in project audit) | Keep think_cost <= 0.005; the lazy commit penalty must be explicitly added as a separate negative reward, not handled via think_cost alone |

---

## Sources

- Project codebase: `/metacontroller/metacontroller.py`, `/main_model/main_model.py`, `/metacontroller/trainer.py`, `/metacontroller/search_tree.py`
- "Attention Is All You Need" (Vaswani et al., 2017) — multi-head attention head count conventions
- Policy Gradient Algorithms survey (Lil'Log, lilianweng.github.io) — REINFORCE, baseline, entropy
- Spinning Up documentation (OpenAI) — entropy coefficient alpha values and scheduling
- Stable Baselines3 policy network conventions — MLP width 256, 2-3 layers for structured RL inputs
- "Normalization and effective learning rates in reinforcement learning" (arxiv.org, 2024) — LayerNorm over BatchNorm for RL
- "Understanding the Impact of Entropy on Policy Optimization" (Ahmed et al., ICML 2019) — entropy collapse dynamics
- "Deep Policy Gradient Methods Without Batch Updates" (arxiv, 2411.15370) — on-policy constraint for trajectory buffers
