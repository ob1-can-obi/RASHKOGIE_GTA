# Technology Stack

**Project:** RASHKOGIE GTA — MCTS-based RL Driving Agent with Rational Cognition Metacontroller
**Researched:** 2026-04-30
**Context:** Brownfield — all modules exist, now completing training infrastructure and dashboard

---

## Recommended Stack

### Core ML Framework

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch | 2.11.0 | All tensor ops, NN definitions, autograd | Already in use; 2.11.0 is current stable as of March 2026. torch.distributions.Categorical for sampling, autograd for REINFORCE. No reason to switch. |
| Python | 3.12 | Runtime | Already locked in. 3.12 is fully supported by PyTorch 2.11.0. |

**Confidence:** HIGH — verified against PyTorch release notes (pytorch.org/pytorch/releases).

---

### Policy Gradient Training (REINFORCE + Entropy Regularization)

**Recommendation: Pure PyTorch — no RL framework.**

The metacontroller trains via custom REINFORCE, not DQN/PPO/SAC. The trajectory structure (metalevel decisions during search, not env episodes) is too non-standard for any RL library to handle without fighting the abstractions.

#### Key Implementation Decisions

**1. Replace argmax with categorical sampling (highest priority fix)**

```python
# WRONG (current code — kills exploration):
decision = decision_logits.argmax(dim=-1)

# CORRECT (training):
dist = torch.distributions.Categorical(logits=decision_logits)
decision = dist.sample()
log_prob = dist.log_prob(decision)

# CORRECT (inference only):
decision = decision_logits.argmax(dim=-1)
```

Use `torch.distributions.Categorical` — it is part of PyTorch core, no extra dependency.

**2. Entropy regularization**

```python
dist = torch.distributions.Categorical(logits=decision_logits)
entropy = dist.entropy()  # [batch] — maximize this

# Loss:
pg_loss = -(log_prob * advantage)
entropy_loss = -entropy_coeff * entropy   # subtract entropy to maximize it
loss = pg_loss + entropy_loss
```

`entropy_coeff` should start at `0.01` and be tunable from the dashboard. If the metacontroller collapses to one action (the current bug), increase to `0.05` or `0.1`. Log `entropy.mean()` every step — if it drops below `0.3` nats for a 4-way distribution, something is collapsing.

**3. Penalty for not being ready**

Add to meta_rewards in `compute_metalevel_advantages`:

```python
# If the token ended before metacontroller committed:
if token_ended_without_commit:
    meta_rewards[-1] += NOT_READY_PENALTY  # large negative, e.g. -5.0
```

**4. Penalty for lazy commits (immediate COMMIT_NEXT without searching)**

```python
if decision == COMMIT_NEXT and search_steps_taken == 0:
    meta_rewards[-1] += LAZY_COMMIT_PENALTY  # e.g. -1.0
```

**Confidence:** HIGH — standard REINFORCE math, verified against PyTorch distributions docs.

---

### Optimizer

**Recommendation: Adam (torch.optim.Adam), lr=3e-4**

Replace the current manual SGD in `update_metapolicy`. Reasons:

| Criterion | Manual SGD (current) | Adam (recommended) |
|-----------|---------------------|-------------------|
| Non-stationary gradients in RL | Poor — fixed lr on noisy PG gradients | Good — per-parameter adaptive lr |
| Implementation effort | Already broken (no momentum, no grad clipping) | One line |
| Stability with sparse rewards | Poor | Good |
| Convergence speed | Slow | ~3-5x faster on policy networks |

```python
optimizer = torch.optim.Adam(meta_mlp.parameters(), lr=3e-4, eps=1e-5)
```

Use `eps=1e-5` (slightly larger than default 1e-8) — standard for RL to prevent numerical issues on low-reward episodes.

**Also add gradient clipping** (critical for REINFORCE stability):

```python
torch.nn.utils.clip_grad_norm_(meta_mlp.parameters(), max_norm=0.5)
```

Apply the same upgrade to the reward head optimizer and action planner optimizer.

**Confidence:** HIGH — Adam + grad clipping is the standard across all major RL frameworks (SB3, TorchRL, CleanRL).

---

### Replay Buffer (Batch Training Infrastructure)

**Recommendation: Custom deque-based circular buffer — no extra dependency.**

TorchRL's `ReplayBuffer` is the right tool in general, but adds a dependency and import overhead on a machine already running GTA V. The metacontroller's trajectory structure (variable-length metalevel rollouts, not fixed (s,a,r,s') tuples) also fits custom code better.

**Implementation:**

```python
from collections import deque
import random

class MetaReplayBuffer:
    """Circular buffer for metalevel trajectories."""
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)   # auto-evicts oldest

    def push(self, trajectory_dict):
        """Store one completed metalevel trajectory."""
        self.buffer.append(trajectory_dict)

    def sample(self, batch_size):
        """Random batch of trajectories for batch training."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)
```

`trajectory_dict` stores: `features` tensor, `decision` int, `advantage` float, `meta_return` float, `entropy` float, `step_in_trajectory` int.

**Buffer size recommendation:** 10,000 trajectory steps. At ~20 Hz with average 5 metalevel steps per token, this is ~100 tokens = ~5 minutes of gameplay. Large enough for stable batches; small enough to not dominate RAM on a gaming PC.

**Batch size:** 64 trajectories per update. Smaller (16-32) works fine early in training.

**When to use the buffer:** The current code updates online (one step at a time). Add a secondary batch update loop: after every N online steps (e.g. N=20), sample a batch from the buffer and run one Adam step. This stabilizes training without requiring off-policy corrections (REINFORCE on a small buffer is a mild approximation, acceptable here).

**Confidence:** MEDIUM — deque buffer is established pattern; the batch frequency tuning is empirical.

---

### Training Dashboard

**Recommendation: FastAPI 0.115+ with SSE + vanilla JS frontend (no React/Node required)**

**Why FastAPI over Flask:**
- Native async — the training loop runs in a background thread/process; FastAPI's async handlers + `asyncio.Queue` bridge them cleanly
- SSE is first-class in FastAPI via `sse-starlette` — no WebSocket handshake complexity
- Auto-generated API docs (Swagger UI) useful for the hyperparameter control panel

**Dependencies:**

```bash
pip install fastapi>=0.115.0 uvicorn[standard]>=0.32.0 sse-starlette>=2.1.0
```

**Architecture:**

```
Training process
    │  pushes metrics to asyncio.Queue (thread-safe via asyncio.run_coroutine_threadsafe)
    │
FastAPI server (same Python process, different thread)
    ├── GET /stream          → SSE endpoint, yields from queue
    ├── POST /hyperparams    → update lr, entropy_coeff, think_cost etc. live
    ├── GET /checkpoints     → list saved checkpoints
    ├── POST /start          → start training session
    ├── POST /stop           → stop training session
    └── GET /                → serves the single HTML page

Frontend (single HTML file, no build step)
    ├── EventSource('/stream')  → receives metric updates
    ├── Chart.js (CDN)          → renders loss curves, entropy, return
    └── HTML forms              → send hyperparameter updates to /hyperparams
```

**Why Chart.js over Plotly/D3:**
- Chart.js loads from CDN (~60KB minified), no npm, no build tool
- Handles live streaming updates with `chart.data.datasets[0].data.push(point); chart.update('none')`
- Sufficient for loss curves, entropy, token return per session

**Why vanilla JS over React/Vue:**
- This is a single-page tool on localhost; the complexity overhead of a JS framework is unjustified
- No build step means no Node.js dependency on the Windows gaming PC

**Dashboard panels to implement:**
1. Loss curve (meta PG loss, reward head MSE, rf predictor MSE)
2. Entropy per step (watch for collapse below 0.3 nats)
3. Token return per episode (running mean)
4. Decision distribution histogram (EXPLORE / INTERRUPT / COMMIT_NEXT / ROLLBACK counts)
5. Hyperparameter panel: `lr`, `entropy_coeff`, `think_cost`, `not_ready_penalty`, `lazy_commit_penalty`, `batch_size`
6. Session log table: session ID, start time, total steps, final loss, checkpoint path

**Confidence:** HIGH — FastAPI SSE pattern is well-established; Chart.js streaming is documented.

---

### Architecture Sizing — MLPs and Attention

#### Metacontroller MLP

Current: single hidden layer 128 units. This is insufficient for the 237-dim input.

**Recommended:**

```python
meta_mlp = nn.Sequential(
    nn.Linear(input_dim, 256),   # input_dim ≈ 237
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 4),
)
```

Rationale:
- 237-dim input with learned structure requires at least 2 hidden layers to form useful intermediate representations
- Width 256 → 256 → 128 follows the "pyramid" pattern standard in SB3's default policy nets
- 4-way output (EXPLORE / INTERRUPT / COMMIT_NEXT / ROLLBACK) is low-dimensional; the last hidden layer can shrink
- Total parameters: ~130K — fast to forward on GPU at 20 Hz

#### Encoder MLPs (ego, scene, route, entity projections)

Current: 2-layer MLPs projecting to 64-dim. These are fine. Keep as-is.

#### Encoder Multi-Head Attention

Current: 4 heads, 1 attention block, embed_dim=64.

**Recommended: 4 heads, 2 stacked attention blocks**

```python
# Stack two cross-attention blocks:
# Block 1: query=[ego|scene|route], K/V=entity_embs → entity_context_1
# Block 2: query=[ego|scene|route|entity_context_1], K/V=entity_embs → entity_context_2
```

Rationale:
- 32 entities × 24 features is a moderately complex set; two passes allow the query to attend conditionally (first pass: find relevant entities, second pass: refine)
- 4 heads at embed_dim=64 means head_dim=16 — adequate for spatial relations. Do not increase heads; head_dim below 8 becomes noise.
- Adding a third block is unlikely to help; entity relationships in driving are not deeply compositional

**head_dim rule of thumb:** `embed_dim / num_heads >= 16`. At 64/4=16, we are at the minimum. If embed_dim is increased, num_heads can increase proportionally.

#### Action Planner MLP

Current: single hidden layer. Recommend 2 hidden layers at 256 units (same logic as metacontroller — input is concatenated z_t + z_next_pred = 256-dim).

#### Reward Head MLP

Already has 3 layers (input → 128 → 64 → 1). This is correct. Keep as-is.

#### Intuition Head MLP

Not reviewed in this pass. Similar to action planner — if single-layer, upgrade to 2 layers at 256.

**Confidence:** MEDIUM — sizing is based on established heuristics (SB3 defaults, attention head_dim rule) applied to the specific input dims of this codebase. Actual optimal sizes require empirical validation.

---

### Checkpoint Management

**Recommendation: torch.save / torch.load with JSON sidecar files. No external experiment tracking service.**

Do not use Weights & Biases, MLflow, or DVC. The project constraint is single offline Windows PC; cloud services add friction and network dependency. The custom dashboard already covers what W&B would provide.

**Checkpoint format:**

```python
import json, time
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints")

def save_checkpoint(session_id, step, modules, hyperparams, metrics):
    """Save all trainable modules + metadata."""
    ckpt_dir = CHECKPOINT_DIR / session_id / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save each module independently (enables partial loading)
    torch.save(modules["meta_mlp"].state_dict(),     ckpt_dir / "meta_mlp.pt")
    torch.save(modules["reward_mlp"].state_dict(),   ckpt_dir / "reward_mlp.pt")
    torch.save(modules["rf_predictor"].state_dict(), ckpt_dir / "rf_predictor.pt")
    torch.save(modules["planner_mlp"].state_dict(),  ckpt_dir / "planner_mlp.pt")
    torch.save(modules["encoder"].state_dict(),      ckpt_dir / "encoder.pt")

    # JSON sidecar: hyperparams + metrics snapshot
    meta = {
        "session_id": session_id,
        "step": step,
        "timestamp": time.time(),
        "hyperparams": hyperparams,
        "metrics": metrics,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
```

**Checkpoint frequency:** Every 500 steps, or on manual trigger from dashboard.

**Module freeze protocol:** When a module is frozen (intuition head, reward head after convergence), save a `frozen=true` flag in the sidecar. The training loop checks this flag before running optimizer steps.

**Session management:**
- Each training run = one session ID (timestamp-based: `2026-04-30_14-23`)
- Sessions stored under `checkpoints/<session_id>/`
- Dashboard reads `meta.json` files to populate session history table

**Confidence:** HIGH — standard PyTorch pattern; no verification needed.

---

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `sse-starlette` | >=2.1.0 | SSE streaming in FastAPI | Dashboard only |
| `uvicorn[standard]` | >=0.32.0 | ASGI server for FastAPI | Dashboard only |
| `websockets` | >=12.0 | Already in use for GTA bridge | Keep as-is |
| `pynput` | >=1.7.6 | Already in use for input capture | Keep as-is |
| `pytest` | >=7.0.0 | Already in use | Keep as-is |
| `collections.deque` | stdlib | Replay buffer | No install needed |
| `torch.distributions` | part of torch | Categorical sampling for REINFORCE | No install needed |

**Explicitly do NOT add:**
- `stable-baselines3` — policy gradient interface incompatible with metalevel trajectory structure
- `torchrl` — adds ~200MB dependency for features already implemented; replay buffer overkill for this scale
- `wandb` / `mlflow` — cloud-centric; project is offline single-machine
- `tensorboard` — custom dashboard supersedes it per PROJECT.md key decision
- `numpy` — avoid introducing it; all tensors should stay in PyTorch to avoid CPU copies
- `React` / `Vue` / `Node.js` — no build tools on the Windows gaming PC

---

## Full Installation

```bash
# Upgrade PyTorch to current stable (if not already on 2.11+)
pip install torch>=2.11.0 --index-url https://download.pytorch.org/whl/cu124

# Dashboard server
pip install fastapi>=0.115.0 uvicorn[standard]>=0.32.0 sse-starlette>=2.1.0

# Already installed (keep versions)
pip install websockets>=12.0 pynput>=1.7.6 pytest>=7.0.0
```

No other changes to requirements.txt needed.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| RL framework | Pure PyTorch | TorchRL, SB3 | Metalevel trajectory structure doesn't map to standard episode-based APIs |
| Optimizer | Adam | Manual SGD (current) | SGD with fixed lr is unstable on noisy PG gradients; no momentum state |
| Replay buffer | Custom deque | TorchRL ReplayBuffer | Adds ~200MB dep; variable-length metalevel trajectories need custom push logic anyway |
| Dashboard server | FastAPI + SSE | Flask + SocketIO | FastAPI native async better for streaming; SocketIO adds handshake complexity |
| Frontend | Vanilla JS + Chart.js | React + recharts | No Node/npm on gaming PC; single-page tool doesn't justify framework overhead |
| Experiment tracking | JSON sidecars | W&B / MLflow | Offline constraint; custom dashboard already covers visualization needs |
| Sampling | torch.distributions.Categorical | custom softmax + multinomial | Categorical is optimized, handles temperature, includes entropy() method |

---

## Sources

- PyTorch 2.11.0 release: [https://github.com/pytorch/pytorch/releases](https://github.com/pytorch/pytorch/releases)
- PyTorch distributions (Categorical): [https://pytorch.org/docs/stable/distributions.html](https://pytorch.org/docs/stable/distributions.html)
- FastAPI SSE: [https://fastapi.tiangolo.com/tutorial/server-sent-events/](https://fastapi.tiangolo.com/tutorial/server-sent-events/)
- TorchRL replay buffers: [https://docs.pytorch.org/rl/stable/tutorials/rb_tutorial.html](https://docs.pytorch.org/rl/stable/tutorials/rb_tutorial.html)
- SB3 policy network defaults: [https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- sse-starlette PyPI: [https://pypi.org/project/fastapi-sse/](https://pypi.org/project/fastapi-sse/)
- Adam in RL (adaptive lr for non-stationary objectives): multiple sources consistent with PyTorch optim docs
