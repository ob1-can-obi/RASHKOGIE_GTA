# Feature Landscape

**Domain:** MCTS + RL autonomous driving agent with learned metacontroller (Value of Cognition)
**Project:** RASHKOGIE GTA
**Researched:** 2026-04-30
**Confidence:** HIGH (project code audited directly; research verified against standard RL literature)

---

## Table Stakes

Features where absence causes training to fail, produce garbage, or be undebuggable.

### Training Infrastructure

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Replay buffer (episodic, FIFO, fixed capacity) | Online one-sample-at-a-time training is unstable; batch sampling breaks temporal correlation and is required for any gradient method to converge reliably | Medium | Plain circular buffer of (features, decision, reward) tuples; no need for TorchRL — keep it simple per coding preferences |
| Batch sampling from buffer | Policy gradient on single samples has catastrophic variance; batching is the standard fix | Low | Sample random minibatch of ~32–128 from buffer per gradient step |
| Module checkpointing (save/load weights per module) | Training is multi-session on one machine; crash = total loss without checkpoints | Low | Each module saves to its own folder (already designed this way); save every N steps and at session end |
| Separate optimizer per module | Each module trains at a different phase and must be frozen independently; shared optimizer state causes cross-contamination | Low | Already partially designed; must confirm Adam state is saved/loaded alongside weights |
| Gradient clipping | Policy gradient with large input features (237-dim metacontroller input) is prone to exploding gradients | Low | `torch.nn.utils.clip_grad_norm_` with max_norm ~1.0 |
| Training session log (human-readable) | Cannot debug a black box; need to know what happened each session | Low | Append-only .txt or .jsonl per session; already specified in instruction.txt for each module folder |

### Metacontroller-Specific Training

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Categorical sampling during training (not argmax) | Current code uses argmax — this kills exploration; the metacontroller will collapse to one decision after a few steps and never recover | Low | `torch.distributions.Categorical(logits=decision_logits).sample()` during training; argmax only at inference |
| Entropy regularization on decision logits | Without it the policy entropy collapses to near-zero; the agent learns "always COMMIT_NEXT" or "always EXPLORE" and stops improving | Low | Add `−β * entropy` to the loss; β starts at 0.01–0.05; anneal slowly; this is the most critical single fix in the codebase |
| Not-ready penalty (penalty when token expires without commit) | Current reward structure incentivizes procrastination; agent must learn to commit before token ends | Low | Detect token-expired-without-commit event in frame loop; apply fixed negative reward (e.g. −2.0) to that trajectory step |
| Lazy-commit penalty (penalty for immediate COMMIT_NEXT without search) | Think-cost alone does not prevent zero-search commits; need explicit signal that some search is expected | Low | Detect COMMIT_NEXT at elapsed_ratio < 0.1 with tree depth == 0; apply small negative reward (e.g. −0.5) |
| REINFORCE with baseline (policy gradient) | Pure REINFORCE has high variance; a simple running mean baseline halves variance with near-zero complexity | Low | Subtract exponential moving average of recent returns from each reward before computing policy gradient loss |

### Action Planner Training

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Imitation learning from player captures | Planner has no ground truth without demonstration data; random initialization trains nothing useful | Low | BCE on active_prob, MSE on strength and duration_s vs captured player inputs; already specified in instruction.txt |
| Duration label computation from future frames | Duration labels cannot be computed at capture time; need look-ahead window over capture buffer | Medium | Sliding window over .jsonl capture data; label = number of frames until control changes significantly |

### Intuition Head + Reward Head Training

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Automated online training during gameplay | Both heads must train continuously from live GTA data; no external dataset exists | Medium | Capture (z_t, action, z_{t+1}) pairs during play; run MSE backward after each frame or every N frames |
| Convergence detection + freeze signal | Metacontroller training must not start until these heads are stable; no freeze signal = contaminated gradient | Medium | Track rolling loss over last 200 steps; declare converged when delta < threshold for 3 consecutive windows |

### Data Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Central training_data folder structure | Currently each module has ad-hoc paths; cross-module training (e.g. reward head reading encoder outputs) will break without consistent layout | Low | Already called out in PROJECT.md Active requirements |
| Per-session CSV + SVG graphs (refresh at session end) | Already specified in instruction.txt; absence means zero visibility into training health | Low | Existing pattern from reward_head/stats.py — extend to all modules |

---

## Differentiators

Features that deliver the research contribution or competitive advantage. Not expected by standard RL tooling — specific to this agent's Value of Cognition thesis.

### Value of Cognition Metrics

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Search depth distribution per episode | Measures whether the metacontroller is learning to think deeper in hard situations; flat distribution = no learning | Medium | Log tree depth at each COMMIT decision; plot histogram per episode in dashboard |
| Decision distribution over time (EXPLORE / INTERRUPT / COMMIT_NEXT / ROLLBACK) | The core diagnostic for metacontroller health; if one decision dominates, training has collapsed | Low | Log decision ID at each metacontroller call; rolling 100-step histogram; alert if any decision > 80% |
| Entropy of decision distribution (per episode) | Single scalar summarizing decision diversity; should be moderate (not max, not min); trend is the signal | Low | Compute `−sum(p * log(p))` over decision counts per episode; plot as curve |
| Value of Cognition estimate per frame | The research thesis metric: expected reward improvement from this search cycle vs. cost paid | High | Estimate as (best_path_value − mean_q_at_commit) / think_cost_paid; noisy but meaningful over many frames |
| Think-cost vs. reward-improvement scatter plot | Visual evidence that the metacontroller learned the cost-benefit tradeoff | Medium | Per episode: x = cumulative think_cost, y = realized episode reward; scatter over training sessions |

### Search Efficiency Metrics

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Nodes expanded per commit | How much search the agent does before committing; should increase in uncertain states, decrease in obvious ones | Low | Count tree nodes visited between each COMMIT; log per frame |
| Search budget utilization | Fraction of allocated search budget used before commit; low utilization = lazy commit, high = good use of budget | Low | elapsed_ratio at commit time is a proxy; already in metacontroller input |
| Rollback rate per episode | High rollback rate signals the search is finding dead ends — useful for tree health and reward head quality | Low | Count ROLLBACK decisions per episode |
| Pre-commit Q improvement (delta Q) | Measures how much the search actually improved on the planner's top-1; this is the signal that search is worth doing | Medium | At each COMMIT: log (committed_q − initial_top1_q); positive trend = search is working |

### Training Dashboard (Custom Flask/FastAPI)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Live loss curves per module (action planner, intuition head, reward head, metacontroller) | Standard in research workflows; custom dashboard gives full layout control vs TensorBoard | Medium | WebSocket push from trainer to browser; Chart.js or Plotly for rendering |
| Live reward curve (episode reward + rolling mean) | Most important single signal for RL training health | Low | Emit after each episode end |
| Decision distribution live histogram | Unique to this project; visualizes metacontroller exploration health in real time | Medium | Histogram auto-refreshes every 10s |
| Hyperparameter control panel | Tune entropy coefficient β, think-cost λ, commit threshold θ, learning rates, batch size from browser without restarting | High | POST endpoint updates a shared config dict; trainer reads on next step; no restart needed |
| Training session management (start/stop/pause, session history, before/after weight comparison) | Multi-session training on one machine needs clean session boundaries | Medium | Session ID, start/stop timestamps, checkpoint paths per session stored in SQLite or JSON |
| Module freeze/unfreeze controls | The strict training order (intuition → reward → planner → metacontroller) needs UI controls to manage freeze state | Low | Toggle per module; calls `requires_grad_(False/True)` on module params |

---

## Anti-Features

Things to deliberately NOT build. Each one has a concrete reason.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| TensorBoard integration | Adds a dependency with its own server, log format, and UI that duplicates the custom dashboard work; two monitoring systems creates confusion; PROJECT.md explicitly chose custom dashboard | Custom Flask/FastAPI dashboard with WebSocket push — already the stated plan |
| Weights & Biases / MLflow / ClearML | Cloud MLOps tools designed for multi-experiment teams; this is one researcher on one machine with no cloud; adds auth, network dependency, and abstraction overhead | Local CSV + SVG graphs per module + custom dashboard |
| Prioritized Experience Replay (PER) | Adds significant complexity (priority trees, importance sampling weights, hyperparameter β_PER); marginal benefit for a single-machine online RL loop where data is cheap to collect | Plain uniform random sampling from a circular buffer; get the basics right first |
| Multi-step returns / n-step TD | Adds trajectory bookkeeping complexity; REINFORCE already uses full-episode returns which is equivalent or better for the metacontroller's episodic structure | Single-step REINFORCE with running baseline |
| Separate target network | Standard in DQN for stability; not applicable here because the metacontroller uses policy gradient (REINFORCE), not Q-learning; adds dead weight | Gradient clipping + entropy regularization achieves stability without target networks |
| Pixel/image-based perception | Out of scope per PROJECT.md; SHVDN provides structured game state | Structured state via named pipe — already validated |
| Multi-agent training | Out of scope for v1 per PROJECT.md | Single vehicle only |
| Hyperparameter search (random search, Bayesian optimization, population-based training) | Overkill for a research prototype on one machine; adds infrastructure without research value at this stage | Manual tuning via dashboard control panel; log each session; iterate manually |
| Distributed/async training | GTA + training share one PC; async workers would compete for GPU with GTA itself | Sequential synchronous training between gameplay episodes |
| Automatic curriculum learning | No curriculum mechanism exists in GTA V without a custom scenario designer; not feasible | Route-based difficulty progression is implicit in the game's map |

---

## Feature Dependencies

```
Replay buffer
  → Batch training (requires buffer to sample from)
    → Metacontroller RL training (requires batched policy gradient)

Categorical sampling fix
  → Entropy regularization (sampling without entropy reg still collapses, just slower)
    → Not-ready penalty + lazy-commit penalty (meaningful only when exploration exists)
      → Metacontroller convergence (all four depend on each other for valid training signal)

Intuition head training loop (converged)
  → Reward head training loop (uses intuition head outputs for rollout simulation)
    → Freeze both heads
      → Action planner imitation learning
        → Metacontroller RL (all upstream heads must be stable)

Per-module checkpointing
  → Training session management (sessions are only resumable with checkpoints)
    → Module freeze/unfreeze UI (UI needs to know which checkpoints exist)

Decision distribution logging
  → Decision distribution live histogram in dashboard
    → Value of Cognition estimate (requires distribution to be healthy first)
```

---

## MVP Recommendation

For the current milestone (brownfield — completing training pipelines and fixing metacontroller issues), prioritize in this order:

1. **Categorical sampling fix** — one line change; unblocks all other metacontroller training work
2. **Entropy regularization** — one loss term; prevents collapse during all subsequent training
3. **Not-ready + lazy-commit penalties** — two reward signals; make the training objective coherent
4. **Replay buffer + batch training** — ~100 lines of plain Python; required for stable gradient updates
5. **REINFORCE with baseline** — reduces variance; needed before long training runs
6. **Module checkpointing** — already partially there; verify save/load of optimizer state too
7. **Decision distribution logging + entropy metric** — first differentiator; minimal complexity, high diagnostic value
8. **Intuition head + reward head training loops** — required before metacontroller RL can start
9. **Custom training dashboard (Flask + WebSocket)** — build after core training works; loss curves and decision histogram first

Defer until core training is validated:
- Value of Cognition estimate (too noisy until metacontroller is trained)
- Hyperparameter control panel (tune manually first; build UI when iteration speed matters)
- Search depth distribution plots (meaningful only after hundreds of training episodes)

---

## Sources

- [How to Make Sense of the Reinforcement Learning Agents? What and Why I Log During Training and Debug](https://neptune.ai/blog/reinforcement-learning-agents-training-debug)
- [RL Tracking: Metrics That Matter When Loss Lies](https://medium.com/@Modexa/rl-tracking-metrics-that-matter-when-loss-lies-44c47a41d321)
- [Understanding PPO Plots in TensorBoard](https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2)
- [Rational Metareasoning for Large Language Models](https://arxiv.org/pdf/2410.05563)
- [Stop! Planner Time: Metareasoning for Probabilistic Planning](https://ojs.aaai.org/index.php/AAAI/article/view/29983/31725)
- [Rational Metareasoning in Problem-Solving Search](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=430b6f7eeb28c2ca3f7d6058f15f9faac60ff024)
- [Policy Gradient Algorithms — Lil'Log](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- [Soft Actor-Critic — Spinning Up documentation](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [Using Replay Buffers — TorchRL documentation](https://docs.pytorch.org/rl/main/tutorials/rb_tutorial.html)
- [Increasing Entropy to Boost Policy Gradient Performance](https://arxiv.org/html/2310.05324)
- Project source: `/Users/jishnuraviprolu/Desktop/RASHKOGIE_GTA/.planning/PROJECT.md`
- Project source: `/Users/jishnuraviprolu/Desktop/RASHKOGIE_GTA/metacontroller/metacontroller.py`
- Project source: `/Users/jishnuraviprolu/Desktop/RASHKOGIE_GTA/instruction.txt`
