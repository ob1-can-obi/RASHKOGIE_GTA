# Domain Pitfalls

**Domain:** MCTS-based RL driving agent with learned metacontroller (RASHKOGIE GTA)
**Researched:** 2026-04-30
**Confidence:** HIGH — pitfalls derived from direct code audit + verified research sources

---

## Critical Pitfalls

Mistakes that cause rewrites or prevent training from converging at all.

---

### Pitfall 1: Argmax During Training Kills Exploration

**What goes wrong:**
`metacontroller.py` line 166 uses `decision_logits.argmax(dim=-1)` during both training and inference. With argmax, the policy is always deterministic — it always picks the highest-logit action from the first step. The policy gradient update on a deterministic action has zero signal: if the agent always commits immediately (COMMIT_NEXT) the reward can still be positive, so there is no gradient incentive to ever EXPLORE.

**Why it happens:**
Argmax was used as a safe default to avoid random behavior. The cost is that exploration never happens, so the policy never observes the outcome of ROLLBACK or depth-2 EXPLORE paths. REINFORCE requires a stochastic policy to generate the variance that gradient estimates correct.

**Consequences:**
- Training converges to COMMIT_NEXT on every step regardless of state — the lazy degenerate
- No search tree diversity: every token execution explores zero children beyond the root
- Policy gradient loss may decrease (agent "learns" to commit confidently) while behavior gets worse
- Tree backup values are never corrected because only root children are ever committed

**Prevention:**
Use `torch.distributions.Categorical(logits=decision_logits).sample()` during training. Use `argmax` only at inference time (add an `explore` flag to the metacontroller call). This is already flagged in PROJECT.md as an active requirement.

**Detection:**
- All `meta_trajectory` entries have `decision == 2` (COMMIT_NEXT) every episode
- `nodes_expanded` == 1 in every search log
- Policy entropy metric near zero from the start

**Phase:** Metacontroller training setup — fix before any RL loop runs.

---

### Pitfall 2: Entropy Collapse — Metacontroller Collapses to One Decision

**What goes wrong:**
Without entropy regularization, a softmax policy over four decisions (EXPLORE=0, INTERRUPT=1, COMMIT_NEXT=2, ROLLBACK=3) will collapse to a single action within tens of gradient steps. The policy finds one decision that produces slightly above-average reward and reinforces it in a feedback loop: that decision gets selected, its advantage is positive, its logit increases, its probability approaches 1.0, then all other decisions are starved of signal forever.

**Why it happens:**
REINFORCE has no explicit diversity objective. Entropy is not in the loss by default. The think_cost penalty (currently 0.01 per search step) already incentivizes quick commits, so the path of least resistance is COMMIT_NEXT, which receives the full realized return without paying think_cost. The policy finds this immediately and never escapes.

**Consequences:**
- Metacontroller always outputs COMMIT_NEXT — no tree search ever happens
- The MCTS-with-learned-deliberation research contribution is lost; system degrades to a planner top-1 policy
- Loss looks fine (low and stable) while behavior is degenerate

**Prevention:**
Add entropy regularization to the policy gradient loss:
```python
# in update_metapolicy (trainer.py)
entropy = -(probs * log_probs).sum(dim=-1).mean()
step_loss = -(log_prob_taken * advantage) - entropy_coeff * entropy
```
Start with `entropy_coeff = 0.01`. If the agent spends too long exploring and never commits (token duration expires without a commit), increase think_cost or reduce entropy_coeff. Adaptive entropy targets (monitor target entropy of ~1.0 nat for 4-way decision) are more robust than fixed coefficients.

**Too-high coefficient warning:** If `entropy_coeff > 0.1`, decisions become nearly uniform random and gradient signal from advantages is overwhelmed. The agent will explore indefinitely, always hitting the budget limit without committing.

**Detection:**
- Softmax probabilities over decisions: all four near 0.25 = coefficient too high
- One probability near 1.0 = collapsed, coefficient too low or missing
- Healthy range: highest decision probability 0.5–0.75, others non-negligible

**Phase:** Metacontroller training setup — add before first training run.

---

### Pitfall 3: Think-Cost Incentivizes Lazy Commits — Reward Hacking the Meta-Reward

**What goes wrong:**
The metalevel reward structure in `trainer.py::compute_metalevel_advantages` assigns `-think_cost` per search step and `realized_return` only to the final commit. If `think_cost = 0.01` and `realized_return` is typically in the range `[-1, +3]`, then each search step costs 1% of a modest positive return. The optimal policy under this structure is to commit immediately: pay zero think_cost, collect the full return. This is reward hacking at the meta-level.

**Why it happens:**
Think_cost was designed to prevent infinite deliberation. But with no minimum search requirement and no penalty for committing before any alternative is evaluated, the minimum-cost policy is always the zero-search policy.

**Consequences:**
- Agent learns to issue COMMIT_NEXT at the very first search step before evaluating any children
- The intuition head, reward head, and tree structure are never exercised during training
- All modules downstream of the search tree never receive training signal from real search outcomes
- The "rational cognition" contribution is nullified

**Prevention:**
Add two complementary penalties:
1. **Not-ready penalty:** If the token ends and `search_state.chosen_token_id is None` (no commit was made), apply a large negative reward (e.g., -2.0) to the final meta_trajectory step. This penalizes failing to commit before the token expires.
2. **Lazy-commit penalty:** If `nodes_expanded == 0` or `nodes_expanded == 1` when COMMIT_NEXT is issued, subtract a penalty (e.g., -0.5) from `realized_return`. Forces at least some exploration before committing.

Both are flagged in PROJECT.md as active requirements.

**Detection:**
- Average `nodes_expanded` per token execution < 2 in training logs
- `search_state.chosen_token_id` is always set on the very first search step
- Token return is close to fallback planner top-1 return (no improvement from search)

**Phase:** Metacontroller reward shaping — before first RL training run.

---

### Pitfall 4: MLP Capacity Insufficient for 237-Dimensional Metacontroller Input

**What goes wrong:**
The metacontroller input is `[drift(128) + elapsed_ratio(1) + token_frames_left(1) + best_q(1) + mean_q(1) + urgency(1) + parent_unexplored(1) + current_path_value(1) + best_path_value(1) + best_current_q(1) + mean_current_q(1) + current_candidate_durations(top_k) + current_candidate_emb(top_k * embed_dim)]`. With top_k=3 and embed_dim=32, this is 128 + 10 + 3 + 96 = 237 dimensions. The current MLP has one hidden layer of 128 units, giving the architecture `237 → 128 → 4`. This compresses a 237-dimensional space through a 128-unit bottleneck to produce 4 logits.

**Why it happens:**
128 hidden units is a reasonable default for quick prototyping. But the metacontroller must learn non-trivial relationships: when drift is high AND urgency is high AND best_q > mean_q, prefer COMMIT_NEXT. These are conditional interactions that a single-layer linear bottleneck cannot represent without enormous hidden width.

**Consequences:**
- Metacontroller underfits — makes approximately random decisions regardless of input quality
- Advantages computed by REINFORCE are correct but the policy cannot express the optimal function
- Training loss plateaus early; adding entropy helps exploration but cannot fix capacity

**Prevention:**
Use at least 2–3 hidden layers with wider first layer:
```python
meta_mlp = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 4),
)
```
The drift vector alone (128 dims) should ideally project to a space larger than 128 before mixing with other signals.

**Detection:**
- Training accuracy on held-out episodes (what decision was actually beneficial?) barely above 25% (random chance for 4-way)
- Ablation: replace metacontroller with a handcrafted rule (commit if urgency > 0.8, else explore) and measure if rule outperforms the learned policy after 1000 steps

**Phase:** Architecture sizing milestone — before extended training begins.

---

### Pitfall 5: Single Online Sample Per Update — REINFORCE Variance Is Unbounded

**What goes wrong:**
The current training loop calls `update_metapolicy` with a single trajectory (one token execution, typically 3–15 search steps). REINFORCE's gradient variance scales as `O(1/N)` where N is the number of samples. With N=1 trajectory per update, variance is at its maximum. The baseline (`predicted_q`) partially reduces variance but cannot eliminate it when individual episodes have high return variance due to GTA stochasticity (traffic, pedestrians, physics).

**Why it happens:**
Online single-sample training is simple and avoids replay buffer complexity. It is labeled as a known limitation in PROJECT.md ("No batch training — everything is online, one sample at a time").

**Consequences:**
- Gradient updates have opposite sign on ~40% of steps (wrong direction by chance)
- Policy oscillates: learns COMMIT_NEXT, is accidentally penalized, briefly explores, is accidentally rewarded for random behavior, collapses back
- Very small learning rate required to compensate, slowing convergence by 10–100x
- Manual SGD with no momentum accumulates these noisy updates without smoothing

**Prevention:**
Implement a trajectory replay buffer (minimum 32 trajectories before first update, sample 16 per batch). This amortizes variance across more episodes and makes the baseline subtraction more meaningful. PROJECT.md flags this as an active requirement. For the manual SGD, switch to Adam (PyTorch `torch.optim.Adam`) which provides adaptive learning rates and implicit momentum — critical when gradient signal is noisy.

**Detection:**
- Per-step advantage sign flips frequently (check sign of advantage list from `compute_metalevel_advantages`)
- `total_loss` zigzags without trend for hundreds of steps
- Gradient norm computed before and after update: if consistently above 5.0, variance is too high for SGD

**Phase:** Batch training infrastructure milestone.

---

## Moderate Pitfalls

---

### Pitfall 6: Reward Head Trained Concurrently With Metacontroller — Moving Target Problem

**What goes wrong:**
`frame_loop.py::drive_token` calls both `train_step` (metacontroller REINFORCE update) and `train_reward_head` (reward head MSE update) on every token execution. The metacontroller's metalevel credit assignment uses `best_q` from the search tree as a baseline — `best_q` comes from the reward head's predictions. If the reward head updates after every token, the metacontroller's baseline is a moving target: the same node evaluated 10 tokens ago would receive a different `r_edge` today.

**Why it happens:**
Concurrent training is convenient and avoids the complexity of staged training. The architectural intent (from PROJECT.md) is to freeze the intuition head and reward head before training the metacontroller, but the current frame_loop bypasses this by training both simultaneously.

**Consequences:**
- Metalevel advantages computed against a reward head that will give different scores tomorrow
- Bootstrapping instability: the policy gradient update reinforces decisions made with one reward model, then the reward model shifts, making those advantages wrong
- Classic "deadly triad" problem (function approximation + bootstrapping + off-policy = instability)

**Prevention:**
Follow the strict training order in PROJECT.md: train intuition head → train reward head until MSE plateaus → freeze both → train metacontroller via REINFORCE. During metacontroller training, the reward head should have `.requires_grad_(False)` applied to all parameters. Monitor reward head MSE before freezing; target MSE < 0.1 (normalized).

**Detection:**
- Reward head MSE oscillates instead of decreasing monotonically
- `r_edge` values for identical transitions change sign between epochs
- Metacontroller loss spikes after reward head makes large updates

**Phase:** Training order enforcement — architecture decision to validate before RL begins.

---

### Pitfall 7: Reward Double-Counting from Distance + Progress Terms

**What goes wrong:**
The reward formula in `reward.py` includes both `-w_d * d_t` (distance penalty) and `+w_p * (d_{t-1} - d_t)` (progress reward). These are partially redundant: if the agent moves from 100m to 90m, the progress reward fires (+1.0 * 10) and the distance penalty also improves (from -0.1*100 to -0.1*90, a net gain of +1.0). The agent receives ~2.0 reward for moving 10m instead of ~1.0, creating a biased signal.

**Why it happens:**
Both terms serve different conceptual purposes (absolute position vs. relative progress) but the reward_method.txt itself notes: "if you already use progress reward, keep the distance term small, otherwise you may double-count." The distance weight (0.1) is intentionally small, but over long episodes this double-counting compounds.

**Consequences:**
- Agent overestimates the value of being close to the goal vs. making progress toward it
- Stationary agent at distance 5m receives better ongoing reward than a moving agent at 10m
- Reward head learns a biased r_edge that inflates Q-values near the goal

**Prevention:**
During reward head training, log individual reward components. If `distance_term` and `progress_term` are strongly correlated across episodes (Pearson r > 0.8), reduce `distance_weight` further (try 0.01) or remove it entirely during the metacontroller training phase. The progress term alone is sufficient for shaping.

**Detection:**
- Agent hovers near a goal waypoint rather than completing the route (trapped in local optimum)
- `distance_term` and `progress_term` in reward components have correlation > 0.8

**Phase:** Reward calibration — check during reward head training phase.

---

### Pitfall 8: Stale Training Data from Frozen Intuition Head — Distribution Shift

**What goes wrong:**
The intuition head trains on (z_t, token_id) → z_{t+1} pairs from player driving data. After freezing, the encoder (main_model) may continue to be fine-tuned or updated during metacontroller RL training, causing the encoder's z_t representations to drift. The frozen intuition head was trained on the old z_t distribution; now it predicts z_child from a different z_t distribution, and its predictions become increasingly inaccurate.

**Why it happens:**
If the encoder's weights are not frozen during metacontroller training, its output distribution changes. The intuition head's input distribution has shifted, but its weights cannot adapt.

**Consequences:**
- Tree expansion uses increasingly wrong z_child predictions
- Reward head scores transitions using wrong embeddings
- Metacontroller sees a feature vector (`drift = z_t - z_running`) that grows artificially large because z_running was computed with old encoder weights

**Prevention:**
Freeze the encoder (main_model) before training the metacontroller. Apply `encoder.requires_grad_(False)` during the metacontroller training phase. If the encoder must continue training (e.g., for representation improvement), the intuition head and reward head must be retrained from scratch with the new encoder.

**Detection:**
- `drift` tensor norm (`||z_t - z_running||`) systematically increasing over training
- Intuition head MSE on held-out rollouts increasing after being frozen (encoder has drifted)

**Phase:** Module freeze management — validate at start of metacontroller training phase.

---

### Pitfall 9: BPE Token Length Mismatch in Credit Assignment

**What goes wrong:**
BPE merged tokens have variable duration (a merged token of 3 base tokens at 2 frames each = 6 frames; a single base token = 1–3 frames). The metalevel trajectory records one decision entry per search step, but the realized return `token_return` is computed over the full token duration. A short token (1 frame) and a long token (15 frames) receive the same single realized_return value, but the discounted return calculation treats them identically. The long token's return is spread over 15 frames of discounting; the short token's return is essentially undiscounted.

**Why it happens:**
The metalevel reward structure treats the token as an atomic unit. The variable duration is not factored into the advantage computation in `compute_metalevel_advantages`.

**Consequences:**
- Short tokens appear systematically more valuable than long tokens (less discounting)
- Metacontroller learns to prefer recommending short tokens regardless of actual return quality
- Vocabulary skews toward base tokens over time if imitation learning reinforces this bias

**Prevention:**
Normalize the realized return by token duration before using it in advantage computation:
```python
# in compute_metalevel_advantages or train_step
normalized_return = token_return / max(1, len(rollout["rewards"]))
```
Alternatively, use per-frame returns and assign them to the single final meta-trajectory step using the token's actual duration as a scale factor. Log token duration alongside token_return in every training step.

**Detection:**
- Average token_return for long tokens (>5 frames) significantly lower than for short tokens of comparable quality
- Token selection drift: metacontroller increasingly selects the planner's base-token candidates over merged tokens
- Histogram of committed token durations shifts toward 1–2 frames over training

**Phase:** Metacontroller training — check before 50 training episodes complete.

---

### Pitfall 10: Manual SGD Without Gradient Clipping — Single Catastrophic Update

**What goes wrong:**
`update_metapolicy` in `trainer.py` uses manual SGD: `p.data -= lr * p.grad`. There is no gradient clipping. In REINFORCE, if a single episode produces an unusually large `realized_return` (e.g., goal reached early = +20 bonus + progress + time bonus = ~25), the advantage can be large, the log_prob gradient can be large, and `lr * p.grad` can exceed the scale of existing weights. One large update can overwrite weeks of learned behavior.

**Why it happens:**
Manual SGD is simpler than using PyTorch's optimizer API. Gradient clipping was not included in the initial implementation.

**Consequences:**
- Single lucky or unlucky episode destroys training progress
- Weights can jump to large magnitudes causing NaN in softmax
- No momentum means subsequent updates cannot correct the large step

**Prevention:**
Add gradient clipping before the parameter update:
```python
torch.nn.utils.clip_grad_norm_(meta_mlp.parameters(), max_norm=0.5)
```
Switch to `torch.optim.Adam` for momentum and adaptive step sizes. If manual SGD is kept for simplicity, add explicit norm logging (`total_grad_norm = sum(p.grad.norm()**2 for ...) ** 0.5`) and skip the update if norm > threshold.

**Detection:**
- NaN appearing in `decision_logits` after a high-reward episode
- `total_loss` in training logs jumping by 100x in a single step
- Softmax over decisions becoming uniform (all 0.25) which indicates NaN propagation

**Phase:** Training infrastructure — add at the same time as the batch training milestone.

---

## Minor Pitfalls

---

### Pitfall 11: GTA + Training on Same GPU — Memory Fragmentation Under Load

**What goes wrong:**
The project runs GTA V and PyTorch training on the same Windows PC with one GPU. GTA V's DirectX usage and PyTorch's CUDA memory management can fragment the GPU memory. PyTorch's CUDA allocator reserves memory in chunks; after a long GTA session, small fragmented blocks may prevent large tensor allocation even when total free memory appears sufficient.

**Prevention:**
- Call `torch.cuda.empty_cache()` before each training session
- Use `torch.cuda.memory_summary()` to monitor fragmentation
- Keep training batch sizes small (16–32 samples) so peak allocation stays well below GTA's headroom
- Consider training on CPU for this project since training batches are small and latency is not critical during offline training phases

**Detection:**
- `CUDA out of memory` errors that appear random (sometimes training works, sometimes not)
- Free VRAM reported as 2 GB but `torch.cuda.memory_allocated()` shows only 200 MB used

**Phase:** Online training loop — monitor from first full training run.

---

### Pitfall 12: Dashboard Loss Curves Are Misleading for Policy Gradient

**What goes wrong:**
In policy gradient training, the policy gradient loss going down does NOT mean the policy is improving. The loss is `-(log_prob * advantage)` and its magnitude depends on how confident the policy is AND on advantage magnitude, both of which change as the policy evolves. A stable or decreasing loss curve is consistent with a policy that is getting worse if the advantages are shrinking (due to baseline convergence) while the policy is not actually improving.

**Why it happens:**
Standard supervised learning intuition (lower loss = better model) does not apply to policy gradient.

**Prevention:**
Primary dashboard metrics should be:
1. **Episode total return** (sum of all token_returns per GTA episode) — this is the only ground-truth metric
2. **Average nodes_expanded per token** — measures whether deliberation is actually happening (target: 3–8)
3. **Decision distribution** (fraction of EXPLORE/INTERRUPT/COMMIT_NEXT/ROLLBACK per episode) — collapse is visible immediately
4. **Policy entropy** — should stay in range [0.8, 1.5] nats for a 4-way decision
5. **Reward head MSE** — measures prediction quality separately from policy quality

Secondary: token_return mean and std per episode, advantage mean and std.

Do NOT treat `total_loss` from `update_metapolicy` as a primary health indicator.

**Detection:**
- Loss decreasing but episode return flat or declining
- Loss oscillating but decision distribution healthy — this is normal

**Phase:** Dashboard build — document this clearly in the monitoring dashboard UI.

---

### Pitfall 13: Token Embedding Table Not Shared Consistently Across Modules

**What goes wrong:**
`token_embed` (an `nn.Embedding`) is created lazily in `intuition_head` and then passed around as a return value. It is reused in `search_tree.search_step` for the metacontroller's candidate embedding feature. If any call site fails to pass back the same `token_embed` instance, a fresh one is created — silently resetting the learned token representations. This can happen if the `search_state.token_embed` reference is not updated after a call to `expand_next_child`.

**Prevention:**
At search initialization, verify that `state.token_embed` is non-None and points to the same object throughout a training session. Log `id(state.token_embed)` at search init and at the end of each token execution to catch silent resets.

**Detection:**
- Token embeddings always look like random init (embedding norm close to `sqrt(embed_dim)`)
- Intuition head MSE not decreasing despite training

**Phase:** Integration testing — before first full end-to-end episode.

---

### Pitfall 14: INTERRUPT Decision Creates Training Data Gaps

**What goes wrong:**
When INTERRUPT fires, the current token stops early (fewer frames than its full duration). The rollout returned by `get_rollout` has fewer rewards than expected. `compute_token_return` handles this correctly, but the reward head training (`train_reward_head`) uses `state_before` and `state_after` to train on the full token duration — the `duration` field in the rollout is `state.frame_idx` (actual frames played), not the original token duration. If `train_reward_head` does not use the actual duration, it trains on a mismatch between the embedding delta and the supposed duration cost.

**Prevention:**
Verify that `rollout["duration"]` reflects actual frames played (it does in the current `get_rollout`). Confirm `train_reward_head` passes `rollout["duration"]` — not `token_table[token_id]["duration"]` — to the `duration_tensor`.

**Detection:**
- Reward head MSE higher for interrupted tokens than for naturally completing tokens
- `duration_tensor` values in reward head training larger than actual frames played

**Phase:** Integration testing — check during first mixed INTERRUPT/COMMIT_NEXT episode.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Metacontroller sampling fix | Argmax during training (Pitfall 1) | Categorical sample in training, argmax at inference |
| Entropy regularization | Coefficient too high → random; too low → collapse (Pitfall 2) | Start at 0.01, monitor decision distribution |
| Reward shaping for metacontroller | Think-cost creates lazy-commit degenerate (Pitfall 3) | Add not-ready penalty + minimum exploration requirement |
| Architecture sizing | Single-layer MLP cannot represent conditional interactions (Pitfall 4) | 2–3 hidden layers, first layer wider than input |
| Batch training infrastructure | Single-sample REINFORCE variance too high for convergence (Pitfall 5) | Replay buffer, batch size 16–32, switch to Adam |
| Training order enforcement | Reward head moving target during metacontroller training (Pitfall 6) | Freeze reward head + encoder before RL phase |
| Reward head calibration | Distance + progress double-counting biases r_edge (Pitfall 7) | Reduce distance_weight to 0.01 or remove |
| Intuition head freeze | Encoder drift invalidates frozen intuition head (Pitfall 8) | Freeze encoder during metacontroller RL phase |
| BPE token credit | Variable token duration creates duration-length bias (Pitfall 9) | Normalize return by token duration |
| Manual SGD | No gradient clipping risks catastrophic single update (Pitfall 10) | Add clip_grad_norm(0.5), consider Adam |
| Online training GPU | Memory fragmentation on shared GPU (Pitfall 11) | empty_cache(), small batch sizes |
| Dashboard build | Policy gradient loss curves are misleading (Pitfall 12) | Track episode return + decision distribution as primaries |
| Integration testing | token_embed not shared → silent resets (Pitfall 13) | Log embedding object identity at search init |
| INTERRUPT handling | Interrupted rollout duration mismatch (Pitfall 14) | Verify actual vs. expected duration in rollout |

---

## Sources

- Direct code audit: `metacontroller/metacontroller.py`, `metacontroller/trainer.py`, `metacontroller/search_tree.py`, `metacontroller/frame_loop.py`, `reward_head/reward_head.py`, `metacontroller/reward.py`
- Project requirements: `.planning/PROJECT.md`
- REINFORCE variance and baselines: [Policy Gradients — Lil'Log](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/), [RLHF Book — Policy Gradients](https://rlhfbook.com/c/06-policy-gradients)
- Entropy regularization: [Entropy-Regularized RL Explained — Towards Data Science](https://towardsdatascience.com/entropy-regularized-reinforcement-learning-explained-2ba959c92aad/), [On Entropy Control in LLM-RL Algorithms](https://arxiv.org/html/2509.03493)
- Metacognitive laziness: [Metacognitive Laziness — Emergent Mind](https://www.emergentmind.com/topics/metacognitive-laziness), [Meta-R1 paper](https://arxiv.org/pdf/2508.17291)
- BPE action tokenization: [Subwords as Skills — NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/7d0c7a899224cd178b2e0cbecf39b5a5-Paper-Conference.pdf), [Credit Assignment in Long-Horizon RL](https://rljclub.github.io/posts/credit-assignment-long-horizon/)
- MCTS collapse: [MCTS as Regularized Policy Optimization](https://arxiv.org/abs/2007.12509), [Systematic Review of Neural MCTS](https://arxiv.org/pdf/2303.08060)
- RL training dashboard: [RL Tracking: Metrics That Matter When Loss Lies — Medium](https://medium.com/@Modexa/rl-tracking-metrics-that-matter-when-loss-lies-44c47a41d321), [Debugging RL Systems — Andy L. Jones](https://andyljones.com/posts/rl-debugging.html)
- Reward hacking: [Reward Hacking in RL — Lil'Log](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- Online RL single GPU: [Laminar RL Framework](https://arxiv.org/pdf/2510.12633), [Performance Optimization in RL](https://apxml.com/courses/advanced-reinforcement-learning/chapter-8-rl-implementation-optimization/rl-performance-optimization)
- Gradient stability: [Troubleshooting PPO Instability](https://apxml.com/courses/rlhf-reinforcement-learning-human-feedback/chapter-4-rl-ppo-fine-tuning/troubleshooting-ppo-instability)
