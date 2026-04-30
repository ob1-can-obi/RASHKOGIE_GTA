# Project Research Summary

**Project:** RASHKOGIE GTA — MCTS-based RL Driving Agent with Rational Cognition Metacontroller
**Domain:** MCTS + Policy Gradient RL with Learned Metalevel Deliberation Controller (Value of Cognition)
**Researched:** 2026-04-30
**Confidence:** HIGH

---

## Executive Summary

RASHKOGIE GTA is a brownfield research project implementing a Value of Cognition metacontroller on top of a BPE-tokenized MCTS driving agent in GTA V. The core research thesis is that a neural metacontroller can learn when deliberation (tree search) is worth its cost — deciding at each step whether to EXPLORE the search tree further, COMMIT to the current best path, INTERRUPT early, or ROLLBACK. All major modules exist; the remaining work is completing training infrastructure, fixing a critical exploration bug, and building a monitoring dashboard. The recommended approach is: fix the argmax-during-training bug immediately, add entropy regularization and reward shaping penalties, build a staged training pipeline that freezes modules in strict order, and expose all training controls via a custom FastAPI dashboard.

The biggest risk to this project is not architectural — it is that three compounding bugs currently prevent any meaningful training signal from reaching the metacontroller. First, the metacontroller uses argmax instead of categorical sampling during training, which eliminates all exploration. Second, there is no entropy regularization, so the policy would collapse to a single action even if sampling were fixed. Third, the reward structure incentivizes zero-search commits via the think_cost mechanism without a counterbalancing not-ready penalty. These three issues must be resolved in sequence before any RL training runs are valid. Until fixed, every training step produces corrupted data.

The architecture is well-designed for its research goals, with a clean 5-phase training dependency chain (encoder → intuition head → reward head → action planner → metacontroller). The main architectural upgrades needed are: deeper MLPs (3-layer with skip connections for the metacontroller, which has a 237-dim heterogeneous input), LayerNorm throughout (batch=1 at 20 Hz invalidates BatchNorm), and a second attention block in the encoder. The stack requires only three new dependencies (FastAPI, uvicorn, sse-starlette) on top of the existing PyTorch 2.11 environment. Keep everything else as-is — no RL frameworks, no cloud tools.

---

## Key Findings

### Recommended Stack

The project is already on the correct core stack: PyTorch 2.11 + Python 3.12 with a custom architecture. The training infrastructure needs three targeted upgrades: (1) replace manual SGD with `torch.optim.Adam(lr=3e-4)` and add `clip_grad_norm(max_norm=0.5)` across all module optimizers — manual SGD is documented to fail on noisy policy gradient objectives; (2) add a custom deque-based trajectory buffer (no external dependency needed) that flushes every 8 complete metalevel trajectories to reduce REINFORCE variance from unbounded single-sample to meaningful batch estimates; (3) build the training dashboard with FastAPI + SSE + vanilla JS + Chart.js, avoiding any Node.js/React/TensorBoard dependencies that would conflict with the Windows gaming PC environment.

**Core technologies:**
- PyTorch 2.11 + `torch.distributions.Categorical`: all NN ops and stochastic sampling — already in use, add Categorical for REINFORCE
- `torch.optim.Adam` (lr=3e-4, eps=1e-5): replace manual SGD — adaptive lr is required for non-stationary PG gradients
- `torch.nn.utils.clip_grad_norm_(max_norm=0.5)`: catastrophic update prevention — required given GTA's large reward variance
- Custom `deque`-based trajectory buffer (stdlib): variance reduction — variable-length metalevel trajectories don't fit TorchRL
- FastAPI 0.115 + sse-starlette 2.1 + uvicorn: training dashboard — native async SSE, no Node.js required
- Chart.js (CDN): live loss curves — 60KB, no npm, handles streaming updates natively
- `torch.save`/`torch.load` + JSON sidecars: checkpointing — offline single-machine; no cloud tools needed

**Explicitly excluded:** stable-baselines3, torchrl, wandb, mlflow, tensorboard, React/Vue/Node.js, numpy (all tensors stay in PyTorch).

See `.planning/research/STACK.md` for full rationale and installation commands.

### Expected Features

The feature priority list is unusually clear for a brownfield project because the bugs and missing pieces are already identified in PROJECT.md. The must-have items are not new features — they are correctness fixes that enable any training to happen at all.

**Must have (table stakes — training correctness):**
- Categorical sampling fix (`argmax` → `dist.sample()` during training) — current code produces zero exploration
- Entropy regularization (`-entropy_coeff * H(pi)` in loss) — prevents policy collapse to single action
- Not-ready penalty (token expires without commit → large negative reward) — required for coherent reward signal
- Lazy-commit penalty (COMMIT_NEXT at search depth 0 → negative reward) — counterbalances think_cost hacking
- REINFORCE with advantage normalization (zero-mean, unit-variance per batch) — prevents reward-scale instability
- Trajectory buffer + batch gradient updates (N=8 trajectories per flush) — REINFORCE variance is unbounded at N=1
- Adam optimizer + gradient clipping for all modules — manual SGD cannot handle noisy PG gradients
- Per-module checkpointing with optimizer state saved — multi-session training requires resumable state
- Module freeze/unfreeze enforcement with convergence detection — training order is a correctness requirement, not a preference

**Should have (training visibility and research diagnostics):**
- Decision distribution logging (EXPLORE/INTERRUPT/COMMIT_NEXT/ROLLBACK histogram) — primary health indicator
- Policy entropy metric per step (target: 0.5–1.5 nats; alert below 0.3) — collapse detector
- Custom FastAPI training dashboard (live loss curves, episode return, decision histogram) — loop iteration speed
- Hyperparameter control panel (entropy_coeff, think_cost, lr, batch_size tunable from browser without restart)
- Per-session CSV + SVG session graphs for each module
- Search depth distribution per episode histogram — validates that metacontroller is learning to deliberate contextually
- Pre-commit Q improvement (delta Q = committed_q − initial_top1_q) — proves search is worth doing

**Defer to post-validation:**
- Value of Cognition estimate per frame (too noisy until metacontroller is trained; high complexity)
- Think-cost vs. reward-improvement scatter plot (meaningful only after hundreds of training sessions)
- Hyperparameter auto-search (manual tuning via dashboard is sufficient at research prototype stage)

See `.planning/research/FEATURES.md` for full feature dependency graph.

### Architecture Approach

The 5-phase training pipeline (encoder → intuition head → reward head → action planner → metacontroller) is correctly designed in PROJECT.md and is enforced by data availability. The key architectural upgrades needed are all about capacity and normalization at the individual component level, not about restructuring the pipeline. Every `.detach()` boundary between modules is already in place; the remaining gap is that metacontroller training (Phase 5) currently runs concurrently with reward head training — they must be sequenced with explicit freeze enforcement.

**Major components and responsibilities:**

1. **Encoder (main_model)** — raw GTA state dict → z_t [128]; ego/scene/route MLP projections + 2-block cross-attention over 32 entities; upgrade to 2 stacked attention blocks with LayerNorm; freeze before metacontroller RL phase to prevent z_t distribution drift

2. **Intuition Head** — (z_t, token_id) → z_next_pred; forward model for tree rollout simulation; train until MSE < 0.05, then freeze; token_embed must be a shared singleton — not recreated between calls

3. **Reward Head** — (z_parent, z_child, rf, duration) → r_edge_pred; edge value estimator for MCTS node scoring; train until MSE < 0.1, then freeze; must use actual rollout duration, not token table duration, for interrupted tokens

4. **Action Planner** — (z_t, z_next_pred) → top-k token candidates; imitation learning from player captures; freeze after top-1 accuracy > 60%; upgrade to 2-layer 256-unit MLP (input is 256-dim concat)

5. **MetaMLP (Metacontroller)** — 237-dim heterogeneous features → EXPLORE/INTERRUPT/COMMIT_NEXT/ROLLBACK logits; upgrade to 3-layer 256-256-128 with skip connection from input to layer 2 and LayerNorm at every hidden layer; Categorical sampling during training, argmax at inference; never frozen — this is the final learned policy

6. **Trajectory Buffer** — accumulates N=8 complete metalevel trajectories, flushes completely on update; never re-uses stale trajectories (on-policy constraint); Adam optimizer instance lives outside the buffer and persists across flushes

7. **FastAPI Dashboard** — SSE streaming of training metrics; POST endpoints for live hyperparameter updates; module freeze/unfreeze toggles; session management with checkpoint history

See `.planning/research/ARCHITECTURE.md` for full data flow diagram and component sizing.

### Critical Pitfalls

1. **Argmax during training kills all exploration** — Use `torch.distributions.Categorical(logits=decision_logits).sample()` during training, argmax only at inference. This is the highest-priority single fix in the codebase; all other metacontroller work is invalid without it.

2. **Entropy collapse to a single decision** — Add `-entropy_coeff * H(pi)` to the policy gradient loss before the first training run. Start at entropy_coeff=0.05 for the first 500 updates, decay to 0.005 by update 2000. Monitor: if policy entropy drops below 0.3 nats before update 500, double the coefficient.

3. **Think-cost creates zero-search reward hacking** — Add a not-ready penalty (-2.0) when a token expires without a commit, and a lazy-commit penalty (-0.5) when COMMIT_NEXT fires with zero nodes expanded. Think_cost alone incentivizes the degenerate zero-search policy.

4. **Training metacontroller with a live reward head** — The reward head must be frozen (`.requires_grad_(False)`) before metacontroller RL begins. Concurrent updates create a moving reward target, which is the "deadly triad" instability in practice. Do not skip the strict training order.

5. **Manual SGD without gradient clipping risks a single catastrophic update** — A large GTA reward (goal reached = +25) combined with no gradient clipping can overwrite the entire policy in one step. Add `clip_grad_norm_(max_norm=0.5)` and switch to Adam before any training run.

See `.planning/research/PITFALLS.md` for 14 pitfalls with detection signals.

---

## Implications for Roadmap

Based on combined research, the correct phase structure follows the strict data dependency chain in the project's own architecture. Phases 1-2 are unblocking correctness fixes. Phases 3-5 are the staged training pipeline. Phase 6 is the research contribution layer on top.

### Phase 1: Training Infrastructure Hardening
**Rationale:** Three compounding bugs (argmax sampling, no entropy reg, broken SGD) make every subsequent training run invalid. These are one-line or few-line fixes that must happen before any other work produces meaningful signal. No new modules — only fixes to existing code.
**Delivers:** A valid REINFORCE training loop that can produce exploration, gradient stability, and a coherent reward signal.
**Addresses:** Categorical sampling fix, entropy regularization, not-ready penalty, lazy-commit penalty, Adam + gradient clipping, advantage normalization.
**Avoids:** Pitfalls 1, 2, 3, 5, 10 (the five that prevent convergence entirely).

### Phase 2: Batch Training and Checkpointing
**Rationale:** Single-sample REINFORCE has unbounded variance. The replay buffer and checkpoint system are prerequisites for any extended training session. Adam optimizer state must persist across flushes or it resets to cold start.
**Delivers:** Trajectory buffer (N=8 flush), per-module checkpointing with optimizer state, module freeze/unfreeze logic with convergence detection, per-session training logs.
**Uses:** Custom deque buffer (stdlib), `torch.save`/`torch.load` + JSON sidecars, Adam optimizer as persistent instance.
**Avoids:** Pitfall 5 (single-sample variance), Pitfall 6 (reward head concurrent training), Pitfall 8 (encoder drift during metacontroller RL).

### Phase 3: Architecture Upgrades
**Rationale:** MLP capacity and normalization upgrades are prerequisites for the metacontroller to express non-trivial decision functions over its 237-dim input. LayerNorm must precede adding the second attention block. MetaMLP class must be `nn.Module` compatible for checkpointing before it is deployed in training.
**Delivers:** 3-layer MetaMLP with skip connection and LayerNorm; 2-block stacked encoder attention with LayerNorm; 2-layer action planner MLP. All modules verifiably serializable.
**Implements:** MetaMLP class replacing inline nn.Sequential, second attention block in encoder, LayerNorm after every hidden layer and attention block.
**Avoids:** Pitfall 4 (insufficient MLP capacity), ARCHITECTURE anti-patterns 3 and 5.

### Phase 4: Staged Module Training Pipeline
**Rationale:** The training order (intuition → reward → planner → metacontroller) is a correctness constraint, not a preference. Each upstream module must converge and freeze before the downstream module's training data is valid. This phase implements the full pipeline with convergence monitoring.
**Delivers:** Working training loops for intuition head (MSE on z_next_pred), reward head (MSE on token returns), action planner (imitation learning BCE + MSE), metacontroller (REINFORCE + entropy). Convergence-triggered freeze protocol for each. Encoder frozen during metacontroller RL phase.
**Avoids:** Pitfall 6 (moving reward target), Pitfall 7 (reward double-counting calibration), Pitfall 8 (encoder drift), Pitfall 9 (BPE token duration normalization), Pitfall 13 (token_embed singleton), Pitfall 14 (INTERRUPT duration mismatch).

### Phase 5: FastAPI Training Dashboard
**Rationale:** The dashboard is a force-multiplier for iteration speed and the research contribution layer requires live visibility into decision distributions and entropy. Build after core training works so the dashboard shows real data from the start.
**Delivers:** FastAPI SSE server, single-HTML frontend with Chart.js, live loss curves for all 4 modules, episode return curve, decision distribution histogram, policy entropy curve, hyperparameter control panel (entropy_coeff, think_cost, lr, batch_size, penalty weights tunable without restart), module freeze/unfreeze controls, session history table with checkpoint paths.
**Uses:** FastAPI 0.115, sse-starlette 2.1, uvicorn, Chart.js (CDN), asyncio.Queue thread bridge.
**Avoids:** Pitfall 12 (policy gradient loss curves are misleading — primary metrics are episode return + decision distribution, not total_loss).

### Phase 6: Research Metrics and Validation
**Rationale:** Value of Cognition metrics are meaningful only after the metacontroller has completed significant training. These are the research contribution deliverables, deferred until upstream training is validated.
**Delivers:** Search depth distribution per episode histogram, pre-commit Q improvement (delta Q) curve, Value of Cognition estimate per frame (noisy but trending), think-cost vs. reward-improvement scatter plot, multi-session comparison tooling.
**Addresses:** All "differentiator" features from FEATURES.md.

### Phase Ordering Rationale

- Phases 1-2 must precede all other phases because the training loop is currently broken in ways that corrupt every gradient update. No architecture work has value until the training loop produces valid signal.
- Phase 3 (architecture upgrades) can proceed in parallel with the end of Phase 2 (checkpointing), since the MetaMLP class can be designed and unit-tested before it is plugged into live training.
- Phase 4 follows Phase 3 because the upgraded MetaMLP must be the version trained, not the old single-layer net.
- Phase 5 (dashboard) is intentionally after Phase 4 because a dashboard showing garbage data from an untrained model wastes iteration time.
- Phase 6 (research metrics) is last because those metrics are only scientifically meaningful after substantial training — putting them earlier creates false signals.

### Research Flags

Phases where deeper research may be needed during planning:
- **Phase 4:** Convergence thresholds (MSE < 0.05 for intuition head, MSE < 0.1 for reward head, top-1 accuracy > 60% for planner) are heuristics — the actual thresholds depend on GTA reward scale in practice. May need empirical calibration after 1-2 training sessions.
- **Phase 4:** BPE token duration normalization (Pitfall 9) — the exact normalization formula and its interaction with the discount factor needs validation against actual token length distributions in GTA V.
- **Phase 6:** Value of Cognition estimate formula — the specific estimator `(best_path_value - mean_q_at_commit) / think_cost_paid` is a first-order approximation; the literature on rational metareasoning suggests more rigorous VOC estimators but they require stable reward head predictions first.

Phases with well-documented patterns (standard implementation, skip deep research):
- **Phase 1:** All fixes are documented PyTorch patterns (Categorical, entropy loss, Adam) — no ambiguity.
- **Phase 2:** Deque buffer and `torch.save` checkpoint patterns are fully documented — straightforward implementation.
- **Phase 3:** MLP sizing (256-256-128) and LayerNorm placement follow established SB3 and transformer conventions.
- **Phase 5:** FastAPI SSE + Chart.js streaming is a well-documented pattern with working examples.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | PyTorch 2.11, Adam, gradient clipping — all verified against PyTorch release notes and RL literature. FastAPI SSE pattern is documented. No experimental choices. |
| Features | HIGH | Features derived from direct code audit of existing modules. Training correctness issues are observable, not inferred. |
| Architecture | HIGH | Input dimensions confirmed by tracing live code (237-dim verified). MLP sizing follows SB3 conventions. LayerNorm vs. BatchNorm reasoning is batch-size-independence logic, not opinion. |
| Pitfalls | HIGH | 5 of 14 pitfalls are observable current bugs in the codebase (verified via code audit). Remaining pitfalls are standard RL failure modes with documented detection signals. |

**Overall confidence:** HIGH

### Gaps to Address

- **Optimal entropy_coeff schedule thresholds (500/2000 update cutoffs):** These depend on actual GTA token frequency and reward scale in live play. Treat the 0.05 → 0.02 → 0.005 schedule as a starting point; log entropy at every flush and adjust manually via the dashboard control panel. Do not hard-code the schedule before observing real entropy curves.
- **Trajectory buffer flush frequency (N=8):** The N=8 recommendation is derived from 20 Hz game rate and 5-15 frame token durations. If game framerate drops (GTA + training on same GPU), the staleness-variance tradeoff shifts. Monitor: if training instability appears at N=8, reduce to N=4. If variance is high, increase to N=16.
- **Reward double-counting (Pitfall 7):** The distance_weight=0.1 may need empirical reduction to 0.01 or elimination. Cannot determine the right value without seeing actual r_edge distributions from the reward head during training. Flag for calibration in Phase 4.
- **Encoder training loop (Phase 4):** ARCHITECTURE.md notes "auxiliary reconstruction or contrastive loss — not yet defined — needs design." The encoder is used but has no standalone training loss defined. This must be resolved before Phase 4 begins; the choice of encoder pretraining objective (reconstruction vs. contrastive vs. joint training with downstream head) affects z_t quality for all downstream modules.

---

## Sources

### Primary (HIGH confidence)
- PyTorch 2.11.0 release notes — https://github.com/pytorch/pytorch/releases
- PyTorch distributions (Categorical) — https://pytorch.org/docs/stable/distributions.html
- FastAPI SSE documentation — https://fastapi.tiangolo.com/tutorial/server-sent-events/
- Stable Baselines3 policy network conventions — https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
- "Attention Is All You Need" (Vaswani et al., 2017) — multi-head attention head count conventions
- Project source code (direct audit): `metacontroller/metacontroller.py`, `metacontroller/trainer.py`, `metacontroller/search_tree.py`, `metacontroller/frame_loop.py`, `reward_head/reward_head.py`
- `.planning/PROJECT.md` — requirements and active flags

### Secondary (MEDIUM confidence)
- Policy Gradient Algorithms — Lil'Log (lilianweng.github.io) — REINFORCE, baseline, entropy regularization
- Spinning Up documentation (OpenAI) — entropy coefficient scheduling
- "Understanding the Impact of Entropy on Policy Optimization" (Ahmed et al., ICML 2019) — entropy collapse dynamics
- "Normalization and effective learning rates in reinforcement learning" (arxiv, 2024) — LayerNorm over BatchNorm for RL
- "Deep Policy Gradient Methods Without Batch Updates" (arxiv, 2411.15370) — on-policy constraint for trajectory buffers
- "Subwords as Skills" NeurIPS 2024 — BPE token credit assignment
- Set Transformer, Relational Deep RL — stacked attention encoder patterns

### Tertiary (LOW confidence — needs empirical validation)
- Specific entropy_coeff thresholds (0.05/0.02/0.005) — derived from entropy magnitude analysis, needs tuning against actual GTA reward scale
- Trajectory buffer N=8 flush frequency — based on 20 Hz rate and token duration estimates
- Convergence thresholds for freeze decisions — heuristics from literature, not calibrated to this specific environment

---
*Research completed: 2026-04-30*
*Ready for roadmap: yes*
