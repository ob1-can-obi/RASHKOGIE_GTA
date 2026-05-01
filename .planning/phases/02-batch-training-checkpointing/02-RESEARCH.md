# Phase 2: Batch Training and Checkpointing - Research

**Researched:** 2026-04-30
**Domain:** PyTorch training infrastructure -- replay buffers, optimizers, gradient clipping, checkpointing
**Confidence:** HIGH

## Summary

Phase 2 replaces the single-sample online REINFORCE training loop (completed in Phase 1) with batch training infrastructure. The current `trainer.py` performs a weight update after every single token trajectory, using manual SGD (`p.data -= lr * p.grad`). This produces high variance gradients and discards optimizer momentum. The phase introduces three interconnected changes: (1) a trajectory replay buffer that accumulates complete metalevel trajectories before triggering a batch update, (2) Adam optimizer with gradient clipping replacing manual SGD for all trainable modules, and (3) per-module checkpoint save/load so training survives process interruption.

The codebase has a clear module structure with six independently trainable nn.Module components: `meta_mlp` (metacontroller decision MLP), `reward_mlp` (reward head), `rf_predictor` (reward feature predictor), `intuition_mlp` (intuition head), `token_embed` (token embedding table), and `planner_mlp` (action planner). Currently all six are lazily created (constructed on first call if None is passed) and all use the same manual SGD pattern. Phase 2 must replace manual SGD in both `update_metapolicy()` and `train_reward_head()` with Adam, and add checkpointing for all modules that carry trainable state.

**Primary recommendation:** Introduce a `TrainingState` class in trainer.py that owns the replay buffer (collections.deque), Adam optimizers (one per module group), gradient step counter, and checkpoint save/load. The `train_step()` function changes from "update immediately" to "buffer trajectory, update when batch is full." This is a refactor of trainer.py with integration changes in frame_loop.py.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BATCH-01 | Trajectory replay buffer (deque, ~10K capacity) collecting full metalevel trajectories | collections.deque with maxlen=10000; store trajectory dict with meta_trajectory + realized_return + metadata |
| BATCH-02 | Batch updates every N=8 trajectories instead of single-sample online updates | TrainingState tracks buffer count, triggers update_from_batch() when len(buffer) % 8 == 0 |
| BATCH-03 | Adam optimizer (lr=3e-4, eps=1e-5) replacing manual SGD for all modules | torch.optim.Adam with per-module parameter groups; replaces manual SGD in update_metapolicy() and train_reward_head() |
| BATCH-04 | Gradient clipping (max_norm=0.5) on all policy gradient updates | torch.nn.utils.clip_grad_norm_ called after loss.backward(), before optimizer.step(); log clip events |
| BATCH-05 | Per-module checkpoint saving (one .pt per module per session) | torch.save() with model_state_dict + optimizer_state_dict + metadata per module |
| BATCH-06 | Checkpoint loading and resume for interrupted training sessions | torch.load() with weights_only=True where possible; restore optimizer state to preserve momentum/variance |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Trajectory buffering | Training Logic (trainer.py) | -- | Buffer is pure training concern, not inference |
| Batch update logic | Training Logic (trainer.py) | -- | When to flush buffer and compute gradients |
| Adam optimizer management | Training Logic (trainer.py) | -- | Optimizer wraps module parameters |
| Gradient clipping | Training Logic (trainer.py) | -- | Applied between backward() and step() |
| Checkpoint save | Training Logic (trainer.py) | Frame Loop (frame_loop.py) | Trainer owns the data; frame_loop triggers save at session boundaries |
| Checkpoint load | Training Logic (trainer.py) | Main Model (main_model.py) | Trainer loads; main_model may need to pass loaded modules to drive_token |
| Module weight passing | Frame Loop (frame_loop.py) | Main Model (main_model.py) | Currently weights are passed through function args; this pattern continues |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.11.0 | Neural network framework, optimizer, checkpoint I/O | Already installed in project .venv [VERIFIED: .venv/bin/pip list] |
| torch.optim.Adam | (part of torch) | Adaptive learning rate optimizer with momentum/variance | Specified in BATCH-03 requirements; standard for RL policy gradients |
| torch.nn.utils.clip_grad_norm_ | (part of torch) | In-place gradient norm clipping | Specified in BATCH-04 requirements; standard for RL gradient stabilization [VERIFIED: Context7 PyTorch docs] |
| collections.deque | (stdlib) | Fixed-capacity FIFO buffer with O(1) append/popleft | Specified in BATCH-01 requirements for trajectory replay buffer |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib.Path | (stdlib) | Checkpoint directory creation and path management | Checkpoint save/load paths |
| datetime | (stdlib) | Session ID generation (timestamp-based) | Naming checkpoint files per session |
| json | (stdlib) | Checkpoint metadata (hyperparameters, step counts) | Optional: store training config alongside .pt files |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| collections.deque | list with manual trimming | deque has O(1) popleft and automatic maxlen eviction; list would need manual slicing |
| Per-module .pt files | Single monolithic checkpoint | Per-module is required by BATCH-05; also enables independent module loading for Phase 4 pipeline |
| Adam | AdamW (decoupled weight decay) | AdamW is newer best practice for supervised learning, but requirements explicitly specify Adam with lr=3e-4, eps=1e-5; no weight decay specified |

**Installation:**
No new packages needed. All dependencies (torch 2.11.0, Python stdlib) are already available.

**Version verification:**
- torch 2.11.0 confirmed installed in .venv [VERIFIED: .venv/bin/pip list]
- Python 3.12.9 confirmed [VERIFIED: python3 --version]
- pytest 9.0.3 confirmed for testing [VERIFIED: .venv/bin/pip list]

## Architecture Patterns

### System Architecture Diagram

```
Token Execution (frame_loop.py)
        |
        v
   drive_token() completes one token
        |
        v
   train_step()  -----> TrainingState.add_trajectory(trajectory_dict)
        |                       |
        |                  buffer full? (len % 8 == 0)
        |                       |
        |              YES: update_from_batch()
        |                       |
        |          +------------+------------+
        |          |                         |
        |    Meta Policy Update       Reward Head Update
        |    (Adam + clip_grad)       (Adam + clip_grad)
        |          |                         |
        |          v                         v
        |    optimizer_meta.step()    optimizer_reward.step()
        |          |                         |
        |          +------------+------------+
        |                       |
        |              increment step_count
        |              save checkpoint if session boundary
        |                       |
        v                       v
   Return train_result     TrainingState persists across calls
```

### Recommended Project Structure

```
metacontroller/
    trainer.py           # Modified: TrainingState class, batch update logic, checkpoint I/O
    frame_loop.py        # Modified: pass TrainingState to train_step, trigger checkpoint saves
    metacontroller.py    # Unchanged
    search_tree.py       # Unchanged
    executor.py          # Unchanged
    reward.py            # Unchanged (pure math, no NN)
    time_context.py      # Unchanged
checkpoints/             # NEW: top-level checkpoint directory
    session_YYYYMMDD_HHMMSS/
        meta_mlp.pt
        reward_mlp.pt
        rf_predictor.pt
        intuition_mlp.pt
        token_embed.pt
        planner_mlp.pt
        training_state.pt   # step count, buffer state, hyperparams
tests/
    test_training_correctness.py  # Existing (20 tests, all passing)
    test_batch_training.py        # NEW: Phase 2 tests
    conftest.py                   # Existing + new fixtures
```

### Pattern 1: TrainingState Class

**What:** A stateful object that owns the replay buffer, optimizers, step counter, and checkpoint logic. Replaces the current stateless function-based training approach.

**When to use:** Created once at training session start, passed to every `train_step()` call, persisted across the entire session.

**Example:**
```python
# Source: PyTorch checkpoint docs [VERIFIED: Context7]
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from collections import deque

class TrainingState:
    """Owns replay buffer, optimizers, and checkpoint logic."""

    def __init__(
        self,
        meta_mlp,
        reward_mlp,
        rf_predictor,
        lr=3e-4,
        eps=1e-5,
        max_grad_norm=0.5,
        batch_size=8,
        buffer_capacity=10000,
    ):
        self.buffer = deque(maxlen=buffer_capacity)
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

        # One Adam optimizer per module group
        self.optimizer_meta = Adam(
            meta_mlp.parameters(), lr=lr, eps=eps
        )
        self.optimizer_reward = Adam(
            list(reward_mlp.parameters()) + list(rf_predictor.parameters()),
            lr=lr, eps=eps,
        )

    def add_trajectory(self, trajectory_dict):
        """Buffer a complete trajectory. Returns True if batch update triggered."""
        self.buffer.append(trajectory_dict)
        if len(self.buffer) >= self.batch_size and len(self.buffer) % self.batch_size == 0:
            return True  # caller should call update_from_batch()
        return False
```

### Pattern 2: Batch Update with Gradient Clipping

**What:** Accumulate gradients over N trajectories, clip, then step the optimizer.

**When to use:** Every time the buffer fills a batch of 8 trajectories.

**Example:**
```python
# Source: PyTorch gradient clipping docs [VERIFIED: Context7]
def update_from_batch(self, meta_mlp, meta_trajectories, advantages_list, entropy_coeff):
    """Batch update over N trajectories with gradient clipping."""
    self.optimizer_meta.zero_grad()

    total_loss = torch.tensor(0.0)
    for traj, advantages in zip(meta_trajectories, advantages_list):
        # Same REINFORCE logic as update_metapolicy, but accumulate grads
        for step, advantage in zip(traj, advantages):
            features = step["features"]
            decision = step["decision"]
            logits = meta_mlp(features)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(decision))
            entropy = dist.entropy()
            step_loss = -(log_prob * advantage)
            entropy_loss = -entropy_coeff * entropy
            total_loss = total_loss + step_loss + entropy_loss

    total_loss = total_loss / len(meta_trajectories)  # mean over batch
    total_loss.backward()

    # Gradient clipping (BATCH-04)
    grad_norm = clip_grad_norm_(meta_mlp.parameters(), self.max_grad_norm)
    clipped = grad_norm > self.max_grad_norm

    self.optimizer_meta.step()
    self.step_count += 1

    return {"loss": total_loss.item(), "grad_norm": grad_norm.item(), "clipped": clipped}
```

### Pattern 3: Per-Module Checkpoint Save/Load

**What:** Each module saves its own .pt file containing model state_dict + optimizer state_dict + metadata.

**When to use:** At session boundaries (end of training, periodic saves, process shutdown).

**Example:**
```python
# Source: PyTorch checkpoint docs [VERIFIED: Context7]
def save_checkpoint(self, checkpoint_dir, meta_mlp, reward_mlp, rf_predictor, session_id):
    """Save per-module checkpoints (BATCH-05)."""
    checkpoint_dir = Path(checkpoint_dir) / f"session_{session_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Meta MLP checkpoint
    torch.save({
        "model_state_dict": meta_mlp.state_dict(),
        "optimizer_state_dict": self.optimizer_meta.state_dict(),
        "step_count": self.step_count,
    }, checkpoint_dir / "meta_mlp.pt")

    # Reward head checkpoint (reward_mlp + rf_predictor share optimizer)
    torch.save({
        "reward_mlp_state_dict": reward_mlp.state_dict(),
        "rf_predictor_state_dict": rf_predictor.state_dict(),
        "optimizer_state_dict": self.optimizer_reward.state_dict(),
        "step_count": self.step_count,
    }, checkpoint_dir / "reward_head.pt")

def load_checkpoint(self, checkpoint_dir, meta_mlp, reward_mlp, rf_predictor):
    """Load from latest checkpoint (BATCH-06). Returns True if loaded."""
    meta_path = Path(checkpoint_dir) / "meta_mlp.pt"
    if meta_path.exists():
        ckpt = torch.load(meta_path, weights_only=False)
        meta_mlp.load_state_dict(ckpt["model_state_dict"])
        self.optimizer_meta.load_state_dict(ckpt["optimizer_state_dict"])
        self.step_count = ckpt["step_count"]
        return True
    return False
```

### Anti-Patterns to Avoid

- **Re-creating optimizers on every batch:** Adam stores per-parameter momentum (m) and variance (v) estimates. If you create a new Adam() each time you call update, you lose all accumulated state and Adam degrades to plain SGD. The optimizer must persist across batches. This is the core of BATCH-03.
- **Clipping before backward():** `clip_grad_norm_` must be called AFTER `loss.backward()` and BEFORE `optimizer.step()`. Calling it before backward has no effect because gradients do not exist yet.
- **Saving model but not optimizer state:** A checkpoint that only saves `model.state_dict()` loses Adam momentum/variance. On resume, the optimizer cold-starts and the first few updates will be noisy. Always save both together (BATCH-06 requirement).
- **Using a single monolithic checkpoint file:** BATCH-05 explicitly requires per-module checkpoint files. This enables Phase 4's staged training pipeline where individual modules are loaded/frozen independently.
- **Accumulating gradients without zeroing:** When processing a batch of trajectories, `optimizer.zero_grad()` must be called once before the loop, NOT inside the loop. Gradients accumulate across loop iterations, then one `optimizer.step()` applies the batch update.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Adaptive learning rate with momentum | Manual momentum tracking (`m = beta1 * m + ...`) | `torch.optim.Adam(params, lr=3e-4, eps=1e-5)` | Adam tracks per-parameter first and second moment estimates; manual implementation is error-prone and misses bias correction [VERIFIED: Context7 PyTorch docs] |
| Gradient norm clipping | Manual norm computation + scaling | `torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)` | Handles multi-parameter norm correctly, returns the actual norm for logging [VERIFIED: Context7 PyTorch docs] |
| Checkpoint serialization | Custom pickle/JSON serialization of tensors | `torch.save()` / `torch.load()` with state_dict | Handles tensor device mapping, dtype preservation, and cross-platform compatibility [VERIFIED: Context7 PyTorch docs] |
| Fixed-capacity FIFO buffer | List with manual `if len > N: pop(0)` | `collections.deque(maxlen=N)` | O(1) append and auto-eviction; list.pop(0) is O(n) [ASSUMED] |

**Key insight:** The entire phase is about replacing hand-rolled training infrastructure (manual SGD, no buffering, no checkpoints) with PyTorch's built-in optimizer and serialization APIs. The complexity is in the integration points (where to create optimizers, when to trigger batch updates, how to wire checkpoint save/load into the session lifecycle), not in the algorithms themselves.

## Common Pitfalls

### Pitfall 1: Optimizer Parameter Identity After Module Re-creation

**What goes wrong:** The current codebase lazily creates modules (`if meta_mlp is None: meta_mlp = nn.Sequential(...)`). If a module is accidentally re-created after the optimizer is constructed, the optimizer's parameter references become stale -- it optimizes the old (discarded) parameters while the forward pass uses new ones. Gradients flow into the new parameters but the optimizer steps on the old ones.

**Why it happens:** The lazy-init pattern in metacontroller.py, reward_head.py, intuition_head.py, and action_planner.py creates fresh modules when None is passed. If any caller forgets to pass the existing module, a new one is created and the optimizer is not updated.

**How to avoid:** Create all modules ONCE at session start (before constructing optimizers). Never pass `None` for any module after initialization. The TrainingState constructor should receive all modules and immediately wrap them in Adam optimizers. Add an assertion that optimizer.param_groups[0]["params"][0].data_ptr() matches the module's first parameter to detect staleness.

**Warning signs:** Loss does not decrease despite many gradient steps. Optimizer step count increases but model predictions do not change.

### Pitfall 2: Buffer Flush Logic Off-by-One

**What goes wrong:** If the batch update triggers when `len(buffer) % batch_size == 0`, the first update happens after trajectory #8. But if the buffer is also being sampled from (not just drained), the trigger condition can fire multiple times or skip entirely.

**Why it happens:** Confusion between "buffer length" and "trajectories since last update."

**How to avoid:** Track a separate `trajectories_since_update` counter. Increment on every `add_trajectory()`, reset to 0 after `update_from_batch()`. Trigger when `trajectories_since_update >= batch_size`.

**Warning signs:** Updates happen at inconsistent intervals or not at all.

### Pitfall 3: Gradient Accumulation Memory Growth

**What goes wrong:** When computing gradients over a batch of 8 trajectories by looping through them and summing losses, all intermediate computation graphs are retained in memory until `backward()` is called. For 8 trajectories with ~10 search steps each, this is 80 forward passes through the MLP before any memory is freed.

**Why it happens:** PyTorch retains the computation graph for every operation until backward().

**How to avoid:** For small MLPs and batch size 8, this is not a problem (total memory is trivial). But if batch_size is increased later, consider calling `.backward()` per trajectory and accumulating gradients, or using `loss.item()` detach pattern. For Phase 2's parameters (batch_size=8, ~237-dim input MLP), memory is not a concern.

**Warning signs:** GPU/CPU memory usage increases linearly with batch_size.

### Pitfall 4: Checkpoint Resume Device Mismatch

**What goes wrong:** Saving a checkpoint on GPU and loading on CPU (or vice versa) can cause device mismatch errors or silent performance degradation.

**Why it happens:** torch.save serializes tensor device metadata. torch.load restores to the same device by default.

**How to avoid:** Use `map_location` parameter: `torch.load(path, map_location='cpu')`. This project runs on a single Windows PC with GPU, so the device is likely consistent, but `map_location` is a best practice for robustness.

**Warning signs:** RuntimeError about tensor expected on cuda:0 but found on cpu (or reverse).

### Pitfall 5: Reward Head Optimizer Grouping

**What goes wrong:** `reward_mlp` and `rf_predictor` are currently trained together in `train_reward_head()` using the same manual SGD step. If they are given separate Adam optimizers, the rf_predictor's gradients might not be stepped, or they might be stepped twice.

**Why it happens:** The two modules are logically coupled (rf_predictor feeds into reward_mlp) but are separate nn.Modules. The current code explicitly lists `all_params = list(reward_mlp.parameters()) + list(rf_predictor.parameters())` and steps them together.

**How to avoid:** Put both modules' parameters into a single Adam optimizer, matching the current behavior. One optimizer, one zero_grad(), one backward(), one step(). This is the cleanest mapping from the existing manual SGD pattern.

**Warning signs:** rf_predictor loss does not decrease, or NaN in rf_predictor outputs.

### Pitfall 6: weights_only=True Incompatibility

**What goes wrong:** PyTorch 2.0+ defaults to `weights_only=False` for `torch.load()` but warns about security. Using `weights_only=True` is safer but only works for state_dict loading -- it fails if the checkpoint contains arbitrary Python objects (like the deque buffer or custom classes).

**Why it happens:** `weights_only=True` restricts deserialization to tensor-safe types. State dicts are safe. Custom objects are not.

**How to avoid:** Use `weights_only=True` for model/optimizer state_dict loading. Save the replay buffer and training metadata separately if needed, or accept `weights_only=False` for the training_state.pt file since it is locally generated (not from an untrusted source).

**Warning signs:** `_pickle.UnpicklingError` or `torch.load` security warning.

## Code Examples

### Example 1: Replacing Manual SGD with Adam in update_metapolicy

```python
# Current code (trainer.py lines 328-338):
for p in meta_mlp.parameters():
    if p.grad is not None:
        p.grad.zero_()
total_loss.backward()
with torch.no_grad():
    for p in meta_mlp.parameters():
        if p.grad is not None:
            p.data -= lr * p.grad

# Replacement with Adam + gradient clipping:
# (optimizer_meta is created ONCE in TrainingState.__init__)
optimizer_meta.zero_grad()
total_loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(meta_mlp.parameters(), max_norm=0.5)
optimizer_meta.step()
```

### Example 2: Trajectory Dict Structure for Buffer

```python
# What gets stored in the deque buffer per trajectory:
trajectory_dict = {
    "meta_trajectory": meta_trajectory,  # list of step dicts with features, decisions
    "realized_return": token_return,     # float
    "is_fallback": is_fallback,          # bool
    "nodes_expanded": nodes_expanded,    # int
    "token_duration_frames": token_duration_frames,  # int
    "rollout": rollout,                  # dict with rewards, states
    "committed_token_id": committed_token_id,  # int
}
```

### Example 3: Batch-Averaged Policy Gradient

```python
# Source: Standard REINFORCE with batch averaging [ASSUMED]
def update_metapolicy_batch(meta_mlp, batch, optimizer, max_grad_norm, entropy_coeff):
    """Process a batch of trajectories in one optimizer step."""
    optimizer.zero_grad()

    batch_loss = torch.tensor(0.0)
    total_steps = 0

    for traj_dict in batch:
        meta_trajectory = traj_dict["meta_trajectory"]
        advantages = traj_dict["advantages"]  # pre-computed

        for step, advantage in zip(meta_trajectory, advantages):
            logits = meta_mlp(step["features"])
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(step["decision"]))
            entropy = dist.entropy()

            batch_loss = batch_loss - log_prob * advantage - entropy_coeff * entropy
            total_steps += 1

    # Mean over all steps in batch
    if total_steps > 0:
        batch_loss = batch_loss / total_steps

    batch_loss.backward()
    grad_norm = clip_grad_norm_(meta_mlp.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": batch_loss.item(),
        "grad_norm": grad_norm.item(),
        "clipped": grad_norm.item() > max_grad_norm,
        "n_steps": total_steps,
    }
```

### Example 4: Complete Checkpoint Save/Load Cycle

```python
# Source: PyTorch checkpoint tutorial [VERIFIED: Context7]
# Save
checkpoint = {
    "model_state_dict": meta_mlp.state_dict(),
    "optimizer_state_dict": optimizer_meta.state_dict(),
    "step_count": step_count,
    "entropy_coeff": get_entropy_coeff(step_count),
    "buffer_size": len(buffer),
}
torch.save(checkpoint, checkpoint_dir / "meta_mlp.pt")

# Load (resume)
ckpt = torch.load(checkpoint_dir / "meta_mlp.pt", map_location="cpu")
meta_mlp.load_state_dict(ckpt["model_state_dict"])
optimizer_meta.load_state_dict(ckpt["optimizer_state_dict"])
step_count = ckpt["step_count"]
# Entropy annealing resumes from the saved step_count automatically
# because get_entropy_coeff(step_count) is a pure function of step
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual SGD (`p.data -= lr * p.grad`) | Adam optimizer with state_dict persistence | Current in codebase (Phase 1) -> Phase 2 target | Eliminates variance from forgetting momentum between updates |
| Online single-sample updates | Mini-batch REINFORCE with trajectory buffer | Current in codebase -> Phase 2 target | Reduces gradient variance by factor of ~sqrt(batch_size) |
| No gradient clipping | `clip_grad_norm_(max_norm=0.5)` | Current -> Phase 2 target | Prevents catastrophic updates from reward spikes |
| No checkpointing | Per-module .pt checkpoint files | Current -> Phase 2 target | Training survives interruption |
| `torch.save(obj)` | `torch.save(obj, weights_only=True)` for state dicts | PyTorch 2.0+ (2023) | Security improvement, prevents arbitrary code execution on load [VERIFIED: Context7] |

**Deprecated/outdated:**
- Manual SGD pattern: Not deprecated in PyTorch but not recommended when optimizer state matters. Phase 2 replaces it with `torch.optim.Adam`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Batch size of 8 is appropriate for the trajectory lengths in this system (typically ~3-15 search steps per trajectory) | Architecture Patterns | Low -- batch_size is a configurable hyperparameter, easily tuned later |
| A2 | collections.deque.pop(0) equivalent (popleft) is O(1) while list.pop(0) is O(n) | Don't Hand-Roll | Very low -- this is well-documented Python behavior |
| A3 | Reward head and rf_predictor should share a single Adam optimizer (matching current manual SGD grouping) | Pitfall 5 | Medium -- separate optimizers could allow different learning rates, but the requirement says "Adam for all modules" without specifying per-module LR |
| A4 | The replay buffer does not need prioritized sampling for trajectory-level REINFORCE | Summary | Low -- uniform sampling is standard for policy gradient methods; prioritized replay is mainly for value-based methods (DQN). Also explicitly out of scope per REQUIREMENTS.md |
| A5 | Checkpoint directory should be at project root (`checkpoints/`) rather than inside `metacontroller/` | Architecture Patterns | Low -- directory location is a convention choice; Phase 4 will need cross-module checkpoint access |

## Open Questions (RESOLVED)

1. **Should the replay buffer persist across sessions via checkpoint?**
   - What we know: The buffer holds trajectory dicts with tensor references. Serializing tensors in a deque is possible but increases checkpoint size.
   - What's unclear: Whether resuming from checkpoint should also resume the buffer contents or start with an empty buffer.
   - Recommendation: Start with empty buffer on resume. The buffer is only ~8 trajectories deep before flushing, so cold-starting it adds minimal overhead. Save buffer_size in the checkpoint metadata for logging purposes only.
   - **RESOLVED: No** -- buffer starts empty on resume. Plans implement this: load_checkpoint does not restore buffer contents.

2. **Should encoder weights (main_model) also get an Adam optimizer and checkpoint?**
   - What we know: The encoder (ego_mlp, scene_mlp, route_mlp, entity_mlp, fusion_mlp, attention weights) is a separate module in main_model.py. Phase 2 requirements mention "all modules" but the current training only updates meta_mlp, reward_mlp, and rf_predictor.
   - What's unclear: Whether BATCH-05 "per-module checkpoint" includes encoder weights that are not currently trained in the metacontroller training loop.
   - Recommendation: Include encoder weights in checkpointing (save state_dict) but do NOT create an Adam optimizer for them yet -- they are not updated in the current training loop. Phase 4 will add encoder training. Save their state_dict for completeness so a full model can be reconstructed from checkpoints.
   - **RESOLVED: Yes** -- save all 6 module state_dicts for completeness (meta_mlp, reward_mlp, rf_predictor, intuition_mlp, token_embed, planner_mlp). No Adam optimizer created for untrained modules; only state_dict saved. Plan 02 updated to implement this.

3. **Should advantage normalization happen per-batch or per-trajectory?**
   - What we know: Phase 1 normalizes advantages within a single trajectory. With batch training, we could normalize across all steps in the batch for better statistics.
   - What's unclear: Whether cross-trajectory normalization changes the learning dynamics.
   - Recommendation: Normalize per-batch (across all steps in the 8-trajectory batch). This gives a better mean/std estimate and was the original intent of TRAIN-05 ("across metalevel trajectory batches").
   - **RESOLVED: Per-batch** -- Plan 01 implements cross-batch normalization in update_metapolicy_batch.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | .pytest_cache (implicit) |
| Quick run command | `.venv/bin/python -m pytest tests/ -x -q` |
| Full suite command | `.venv/bin/python -m pytest tests/ -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BATCH-01 | Trajectory buffer accumulates entries up to capacity | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_buffer_accumulation -x` | Wave 0 |
| BATCH-01 | Buffer evicts oldest when over capacity | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_buffer_eviction -x` | Wave 0 |
| BATCH-02 | Update triggers after every 8th trajectory | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_batch_trigger_every_n -x` | Wave 0 |
| BATCH-02 | No update before 8 trajectories accumulated | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_no_premature_update -x` | Wave 0 |
| BATCH-03 | Adam optimizer updates model weights | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_adam_updates_weights -x` | Wave 0 |
| BATCH-03 | Adam momentum persists across batch flushes | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_adam_state_persists -x` | Wave 0 |
| BATCH-04 | Gradient norm is clipped to max_norm=0.5 | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_gradient_clipping -x` | Wave 0 |
| BATCH-04 | Clip event is logged/returned when norm exceeds threshold | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_clip_event_reported -x` | Wave 0 |
| BATCH-05 | Per-module .pt files saved with correct state_dict keys | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_checkpoint_save_structure -x` | Wave 0 |
| BATCH-05 | Each module file loadable independently | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_independent_module_load -x` | Wave 0 |
| BATCH-06 | Resume from checkpoint restores model weights exactly | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_checkpoint_resume_weights -x` | Wave 0 |
| BATCH-06 | Resume from checkpoint restores optimizer state (momentum/variance) | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_checkpoint_resume_optimizer -x` | Wave 0 |
| BATCH-06 | Resume from checkpoint restores step count (entropy annealing continues) | unit | `.venv/bin/python -m pytest tests/test_batch_training.py::test_checkpoint_resume_step_count -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `.venv/bin/python -m pytest tests/ -x -q`
- **Per wave merge:** `.venv/bin/python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_batch_training.py` -- covers BATCH-01 through BATCH-06
- [ ] Update `tests/conftest.py` with new fixtures for TrainingState, mock modules with parameters

*(No framework install needed -- pytest 9.0.3 already available)*

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | -- |
| V3 Session Management | no | -- |
| V4 Access Control | no | -- |
| V5 Input Validation | no | No external input; all data from internal training loop |
| V6 Cryptography | no | -- |

### Known Threat Patterns for PyTorch Checkpointing

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Pickle deserialization attack via malicious .pt file | Tampering | `torch.load(path, weights_only=True)` for state_dict-only loads; only load locally generated checkpoints [VERIFIED: Context7 PyTorch docs] |
| Checkpoint file corruption during save (crash mid-write) | Tampering | Save to temp file then atomic rename (`Path.rename()`); verify load after save in tests |

Note: This is a local research project on a single machine. Checkpoints are generated and consumed locally. The primary risk is data integrity (corruption), not adversarial attacks.

## Sources

### Primary (HIGH confidence)
- Context7 `/pytorch/pytorch` -- checkpoint save/load patterns, Adam optimizer API, clip_grad_norm_ API
- Context7 `/websites/pytorch_stable` -- torch.optim.Adam parameters and algorithm details, optimizer state_dict structure
- Codebase analysis: `metacontroller/trainer.py` (current manual SGD at lines 328-338, 537-549)
- Codebase analysis: `metacontroller/frame_loop.py` (train_step integration at lines 267-276)
- Codebase analysis: `metacontroller/metacontroller.py` (lazy module creation pattern)
- Codebase analysis: `reward_head/reward_head.py` (module structure, RF_DIM=6)
- Codebase analysis: `main_model/main_model.py` (encoder weights dict structure)
- Codebase analysis: `action_planner/action_planner.py` (planner_mlp lazy creation)
- Codebase analysis: `intuition_head/intuition_head.py` (token_embed + intuition_mlp lazy creation)

### Secondary (MEDIUM confidence)
- PyTorch stable documentation: Adam optimizer default values and bias correction

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries are already installed, APIs verified via Context7
- Architecture: HIGH -- Patterns are standard PyTorch checkpoint/optimizer usage applied to the existing codebase structure
- Pitfalls: HIGH -- Identified from analyzing the specific lazy-init module pattern in this codebase + standard PyTorch checkpointing pitfalls

**Research date:** 2026-04-30
**Valid until:** 2026-05-30 (stable domain -- PyTorch checkpoint APIs do not change frequently)
