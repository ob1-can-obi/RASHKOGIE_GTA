"""
Trainer — computes returns, performs tree backup, and updates the
metacontroller from realized outcomes.

The executor plays a token and returns a rollout (per-frame rewards).
The trainer takes that rollout and does the actual learning:

1. Compute the realized token return (discounted sum of per-frame rewards,
   optionally bootstrapped with a value estimate of the next state).
2. Back up the realized return into the search tree (update N, W, Q on
   the committed branch).
3. Compute metalevel advantages over the full search trajectory
   (each search action gets credit, not just the final commit).
4. Update metacontroller weights via policy gradient on the metalevel
   trajectory.

Conceptual split:
    executor  = "play the token, collect raw data"
    trainer   = "learn from what happened"
"""

import os
import sys
import json
import math
import logging
import threading
from pathlib import Path

import torch
from collections import deque
from datetime import datetime
from torch.distributions import Categorical
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

REWARD_HEAD_DIR = Path(__file__).resolve().parent.parent / "reward_head"
if str(REWARD_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_HEAD_DIR))

from reward_head import reward_head, extract_reward_features, predict_reward_features


# =========================================================================
# 1. Compute realized token return
# =========================================================================

# Penalty constants (Claude's discretion per CONTEXT.md)
# C: not-ready penalty multiplier. Penalty = -C * token_duration_frames
# K: lazy-commit penalty multiplier. Penalty = -K * token_duration_frames * scaling
# MIN_SEARCH_NODES: threshold below which a commit is considered "lazy"
NOT_READY_C = 0.1
LAZY_K = 0.05
MIN_SEARCH_NODES = 2


# =========================================================================
# Entropy annealing schedule (TRAIN-02, per D-10/D-11)
# =========================================================================

# Default anneal_steps=5000 (Claude's discretion per CONTEXT.md)
# Exposed as a configurable parameter for later tuning per D-11
DEFAULT_ENTROPY_ANNEAL_STEPS = 5000

# Batch training constants (BATCH-01 through BATCH-04)
DEFAULT_BATCH_SIZE = 8
DEFAULT_BUFFER_CAPACITY = 10000
DEFAULT_LR = 3e-4
DEFAULT_EPS = 1e-5
DEFAULT_MAX_GRAD_NORM = 0.5


def get_entropy_coeff(step, start=0.05, end=0.005, anneal_steps=DEFAULT_ENTROPY_ANNEAL_STEPS):
    """
    Linear decay of entropy coefficient from start to end over anneal_steps.

    Per D-10: linear decay (not exponential).
    Per D-11: anneal_steps is configurable, default chosen by Claude.

    Input:
    - step: int, current gradient step count
    - start: float, initial entropy coefficient (0.05)
    - end: float, final entropy coefficient (0.005)
    - anneal_steps: int, steps over which to anneal

    Output:
    - coeff: float, current entropy coefficient
    """
    if anneal_steps <= 0:
        return end
    frac = min(1.0, step / anneal_steps)
    return start + frac * (end - start)


def compute_token_return(rollout, gamma=0.99, bootstrap_value=None):
    """
    Compute the discounted return over a full token rollout.

    R_token = r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^(k-1)*r_{k-1}

    If bootstrap_value is provided (a value estimate of the state AFTER
    the token finishes), we add it as a terminal value:

    R_token = sum of discounted rewards + gamma^k * V(next_real_state)

    Input:
    - rollout: dict from executor.execute_token, must have "rewards" list
    - gamma: discount factor
    - bootstrap_value: float or None, V(state after token) for bootstrapping

    Output:
    - token_return: float, the realized discounted return
    """
    rewards = rollout["rewards"]
    k = len(rewards)

    token_return = 0.0
    for i, r in enumerate(rewards):
        token_return += (gamma ** i) * r

    if bootstrap_value is not None:
        token_return += (gamma ** k) * bootstrap_value

    # -----------------------------------------------------------------
    # duration normalization (TRAIN-06, per D-08/D-09)
    # sqrt preserves short-burst surprise signals while dampening
    # variable-length BPE bias
    # -----------------------------------------------------------------

    if k > 0:
        token_return = token_return / math.sqrt(k)

    return token_return


# =========================================================================
# 2. Tree backup
# =========================================================================

def backup_tree(root, committed_token_id, realized_return):
    """
    Update the search tree with the realized return from the committed token.

    Finds the root child that was committed and updates its N/W/Q with the
    real outcome.  This corrects the tree's value estimates so that if the
    tree is reused or inspected, it reflects reality.

    Input:
    - root: TreeNode, the search tree root
    - committed_token_id: int, which token was committed
    - realized_return: float, the discounted return from the rollout
    """
    for child in root.children:
        if int(child.token_id) == int(committed_token_id):
            child.n += 1
            child.w += realized_return
            child.q = child.w / child.n

            # also update root stats
            root.n += 1
            root.w += realized_return
            root.q = root.w / root.n
            break


# =========================================================================
# 3. Metalevel trajectory and credit assignment
# =========================================================================

def compute_metalevel_advantages(
    meta_trajectory,
    realized_return,
    think_cost=0.01,
    gamma=0.99,
    is_fallback=False,
    nodes_expanded=0,
    token_duration_frames=1,
):
    """
    Assign credit to every search decision the metacontroller made,
    not just the final commit.

    The metalevel trajectory is the sequence of decisions during search:
    KEEP, KEEP, ROLLBACK, KEEP, COMMIT_NEXT, etc.  Each search step costs
    think_cost (time spent thinking instead of acting).  The final commit
    gets the realized token return.

    Metalevel rewards:
    - search actions (KEEP, ROLLBACK): -think_cost  (cost of deliberation)
    - final action (COMMIT_NEXT, INTERRUPT): realized_return

    We compute discounted returns backward from the end, so early search
    decisions that led to a good final commit get positive advantage, and
    excessive thinking before a bad commit gets penalized.

    Input:
    - meta_trajectory: list of dicts, one per search step:
        {"decision": int, "decision_logits": tensor [1,4], "predicted_q": float}
    - realized_return: float, the discounted return from the token rollout
    - think_cost: float, penalty per search step (cost of deliberation)
    - gamma: discount factor for metalevel returns
    - is_fallback: bool, True if metacontroller failed to commit (D-04)
    - nodes_expanded: int, how many nodes were expanded during search
    - token_duration_frames: int, duration of the token in frames

    Output:
    - advantages: list of floats, one per trajectory step
    - meta_returns: list of floats, discounted metalevel return at each step
    """
    n = len(meta_trajectory)
    if n == 0:
        return [], []

    # -----------------------------------------------------------------
    # assign metalevel rewards
    # -----------------------------------------------------------------

    meta_rewards = []
    for i, step in enumerate(meta_trajectory):
        if i == n - 1:
            # final step: the commit/interrupt gets the realized return
            meta_rewards.append(realized_return)
        else:
            # search steps: pay the think cost
            meta_rewards.append(-think_cost)

    # -----------------------------------------------------------------
    # inject penalty signals (TRAIN-03, TRAIN-04)
    # penalties are already duration-scaled per D-02/D-06/D-07
    # applied AFTER duration normalization of token_return (Pitfall 5)
    # -----------------------------------------------------------------

    if is_fallback and n > 0:
        # TRAIN-03 / D-01 / D-02: not-ready penalty replaces final reward
        # (the commit was not the agent's decision, no credit)
        not_ready_penalty = -NOT_READY_C * token_duration_frames
        meta_rewards[-1] = not_ready_penalty

    elif n > 0 and nodes_expanded < MIN_SEARCH_NODES:
        # TRAIN-04 / D-05 / D-06: lazy-commit penalty added to final reward
        # scaling: max(0, 1 - nodes_expanded/threshold) so 0 nodes = full penalty
        # short tokens get near-zero penalty because token_duration_frames is small
        scaling = max(0.0, 1.0 - nodes_expanded / MIN_SEARCH_NODES)
        lazy_penalty = -LAZY_K * token_duration_frames * scaling
        meta_rewards[-1] += lazy_penalty

    # -----------------------------------------------------------------
    # compute discounted returns backward
    # -----------------------------------------------------------------

    meta_returns = [0.0] * n
    running = 0.0
    for i in range(n - 1, -1, -1):
        running = meta_rewards[i] + gamma * running
        meta_returns[i] = running

    # -----------------------------------------------------------------
    # advantages: return - baseline (predicted_q at that step)
    # -----------------------------------------------------------------

    advantages = []
    for i, step in enumerate(meta_trajectory):
        baseline = step.get("predicted_q", 0.0)
        advantages.append(meta_returns[i] - baseline)

    return advantages, meta_returns


# =========================================================================
# 4. Update metacontroller weights
# =========================================================================

def update_metapolicy(meta_mlp, meta_trajectory, advantages, lr=1e-3, entropy_coeff=0.05):
    """
    Policy gradient update over the full metalevel trajectory.

    Each search step's decision gets reinforced or penalized based on
    its advantage.  Positive advantage = good decision, strengthen it.
    Negative advantage = bad decision, weaken it.

    TRAIN-02: Adds entropy regularization to prevent decision collapse.
    TRAIN-05: Normalizes advantages to zero-mean unit-variance.

    Input:
    - meta_mlp: the metacontroller MLP (nn.Sequential)
    - meta_trajectory: list of dicts with "decision" and "features"
    - advantages: list of floats from compute_metalevel_advantages
    - lr: learning rate
    - entropy_coeff: float, current entropy regularization coefficient

    Output:
    - total_loss: float, sum of per-step policy + entropy losses (for logging)
    """

    # -----------------------------------------------------------------
    # TRAIN-05: advantage normalization (zero-mean, unit-variance)
    # Skip for single-step trajectories (std is undefined)
    # -----------------------------------------------------------------

    if len(advantages) > 1:
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        adv_mean = adv_t.mean()
        adv_std = adv_t.std()
        adv_t = (adv_t - adv_mean) / (adv_std + 1e-8)
        advantages = adv_t.tolist()

    # -----------------------------------------------------------------
    # TRAIN-02 + policy gradient: compute loss with entropy bonus
    # Entropy is computed from CURRENT logits (re-run through MLP),
    # NOT from stale stored logits (Pitfall 2 from RESEARCH.md)
    # -----------------------------------------------------------------

    step_losses = []
    entropy_sum = torch.tensor(0.0)

    for step, advantage in zip(meta_trajectory, advantages):
        features = step["features"]    # [1, fused_dim + 6] -- rerun to get live graph
        decision = step["decision"]    # int

        logits    = meta_mlp(features)                        # [1, 4]
        dist      = Categorical(logits=logits)
        log_prob  = dist.log_prob(torch.tensor(decision))     # scalar
        entropy   = dist.entropy()                            # scalar

        # policy gradient loss: -log_prob * advantage (REINFORCE)
        step_losses.append(-(log_prob * advantage))
        entropy_sum = entropy_sum + entropy

    if not step_losses:
        return 0.0

    policy_loss = torch.stack(step_losses).sum()

    # TRAIN-02: entropy bonus -- SUBTRACT to MAXIMIZE entropy
    # (Pitfall 1: adding would MINIMIZE entropy, causing collapse)
    entropy_loss = -entropy_coeff * entropy_sum
    total_loss = policy_loss + entropy_loss

    # -----------------------------------------------------------------
    # backward + manual SGD step
    # -----------------------------------------------------------------

    for p in meta_mlp.parameters():
        if p.grad is not None:
            p.grad.zero_()

    total_loss.backward()

    with torch.no_grad():
        for p in meta_mlp.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad

    return total_loss.item()


# =========================================================================
# 5. Full train step (convenience wrapper)
# =========================================================================

def train_step(
    rollout,
    root,
    committed_token_id,
    meta_trajectory,
    meta_mlp,
    gamma=0.99,
    think_cost=0.01,
    lr=1e-3,
    bootstrap_value=None,
    is_fallback=False,
    nodes_expanded=0,
    token_duration_frames=1,
    entropy_coeff=0.05,
    training_state=None,  # TrainingState for batch mode (Phase 2)
):
    """
    Full learning cycle after one token execution.

    1. Compute realized token return from the rollout
    2. Back up the return into the search tree
    3. Compute metalevel advantages over the search trajectory
    4. Update metacontroller weights

    Input:
    - rollout: dict from executor.execute_token
    - root: TreeNode, the search tree root
    - committed_token_id: int, which token was committed
    - meta_trajectory: list of search step dicts (from the orchestrator)
    - meta_mlp: metacontroller MLP to update
    - gamma: discount factor
    - think_cost: penalty per search step
    - lr: learning rate
    - bootstrap_value: float or None, V(next state) for bootstrapping

    Output dict:
    - token_return: float, realized discounted return
    - advantages: list of floats, one per search step
    - meta_returns: list of floats, discounted metalevel return at each step
    - total_loss: float, policy gradient loss
    - n_search_steps: int, how many metalevel decisions were made
    """

    # step 1: realized return
    token_return = compute_token_return(
        rollout, gamma=gamma, bootstrap_value=bootstrap_value,
    )

    # step 2: tree backup
    backup_tree(root, committed_token_id, token_return)

    # step 3: metalevel credit assignment
    advantages, meta_returns = compute_metalevel_advantages(
        meta_trajectory,
        realized_return=token_return,
        think_cost=think_cost,
        gamma=gamma,
        is_fallback=is_fallback,
        nodes_expanded=nodes_expanded,
        token_duration_frames=token_duration_frames,
    )

    # step 4: weight update (batch mode vs legacy single-sample mode)
    if training_state is not None:
        # Batch mode: buffer the trajectory, defer update
        trajectory_dict = {
            "meta_trajectory": meta_trajectory,
            "realized_return": token_return,
            "is_fallback": is_fallback,
            "nodes_expanded": nodes_expanded,
            "token_duration_frames": token_duration_frames,
            "rollout": rollout,
            "advantages": advantages,
        }
        batch_ready = training_state.add_trajectory(trajectory_dict)
        total_loss = 0.0  # No immediate update
    else:
        # Legacy single-sample mode (backward compatible)
        total_loss = update_metapolicy(
            meta_mlp, meta_trajectory, advantages, lr=lr,
            entropy_coeff=entropy_coeff,
        )
        batch_ready = False

    return {
        "token_return":   token_return,
        "advantages":     advantages,
        "meta_returns":   meta_returns,
        "total_loss":     total_loss,
        "n_search_steps": len(meta_trajectory),
        "is_fallback":    is_fallback,
        "entropy_coeff":  entropy_coeff,
        "batch_ready":    batch_ready,  # True when buffer is full (Phase 2)
    }


# =========================================================================
# 6. Train reward head
# =========================================================================

def train_reward_head(
    rollout,
    token_return,
    encode_fn,
    reward_mlp,
    rf_predictor,
    hidden_dim=128,
    lr=1e-3,
):
    """
    Train the reward head NN against the realized token return.

    The reward head predicts r_edge from embeddings alone.  Here we give it
    the real outcome so it learns to score transitions accurately.

    Training target: token_return (discounted sum of real per-frame rewards)

    Input:
    - rollout:       dict from get_rollout — needs state_before, state_after, duration
    - token_return:  float, realized discounted return from compute_token_return
    - encode_fn:     callable(gta_state) -> z_t  [1, fused_dim]
    - reward_mlp:    reward head MLP to update
    - rf_predictor:  reward feature predictor MLP to update
    - fused_dim, hidden_dim, lr: network and training params

    Output dict:
    - reward_loss:   float, MSE loss
    - reward_mlp:    updated MLP
    - rf_predictor:  updated MLP
    """

    state_before = rollout["state_before"]
    state_after  = rollout["state_after"]
    duration     = rollout["duration"]

    # -------------------------------------------------------------------------
    # Step 1: encode real states into embeddings
    # -------------------------------------------------------------------------

    z_parent = encode_fn(state_before).detach()  # [1, fused_dim]
    z_child  = encode_fn(state_after).detach()  # [1, fused_dim]

    fused_dim = z_parent.shape[-1]

    # -------------------------------------------------------------------------
    # Step 2: extract real reward features from both states
    # -------------------------------------------------------------------------

    rf_parent = extract_reward_features(state_before)   # [1, RF_DIM]
    rf_child  = extract_reward_features(state_after)    # [1, RF_DIM]  (real, not predicted)

    # -------------------------------------------------------------------------
    # Step 3: build duration and time_left tensors
    # token is already done so time_left = 0
    # -------------------------------------------------------------------------

    duration_t  = torch.tensor([[float(duration)]], dtype=torch.float32)
    time_left_t = torch.tensor([[0.0]],             dtype=torch.float32)

    # -------------------------------------------------------------------------
    # Step 4: forward pass through reward head with real rf_child
    # (bypasses rf_predictor — we have the real values here)
    # -------------------------------------------------------------------------

    r_pred, reward_mlp = reward_head(
        z_parent   = z_parent,
        z_child    = z_child,
        rf_parent  = rf_parent,
        rf_child   = rf_child,
        duration   = duration_t,
        time_left  = time_left_t,
        reward_mlp = reward_mlp,
        fused_dim  = fused_dim,
        hidden_dim = hidden_dim,
    )

    # -------------------------------------------------------------------------
    # Step 5: also train rf_predictor — it should have predicted rf_child
    # from z_parent, delta_z, rf_parent
    # -------------------------------------------------------------------------

    delta_z = z_child - z_parent

    rf_pred, rf_predictor = predict_reward_features(
        z_parent     = z_parent,
        delta_z      = delta_z,
        rf_parent    = rf_parent,
        rf_predictor = rf_predictor,
        fused_dim    = fused_dim,
        hidden_dim   = hidden_dim // 2,
    )

    # -------------------------------------------------------------------------
    # Step 6: compute losses
    # reward_mlp loss: predicted r_edge vs realized token_return
    # rf_predictor loss: predicted rf vs real rf_child
    # -------------------------------------------------------------------------

    target_return = torch.tensor([[token_return]], dtype=torch.float32)
    reward_loss   = (r_pred - target_return) ** 2

    rf_loss = ((rf_pred - rf_child.detach()) ** 2).mean()

    total_loss = reward_loss + rf_loss

    # -------------------------------------------------------------------------
    # Step 7: backward + manual SGD on both MLPs
    # -------------------------------------------------------------------------

    all_params = list(reward_mlp.parameters()) + list(rf_predictor.parameters())

    for p in all_params:
        if p.grad is not None:
            p.grad.zero_()

    total_loss.backward()

    with torch.no_grad():
        for p in all_params:
            if p.grad is not None:
                p.data -= lr * p.grad

    return {
        "reward_loss": reward_loss.item(),
        "rf_loss":     rf_loss.item(),
        "reward_mlp":  reward_mlp,
        "rf_predictor": rf_predictor,
    }


# =========================================================================
# 7. TrainingState -- batch training infrastructure (Phase 2)
# =========================================================================

class TrainingState:
    """
    Owns the trajectory replay buffer, Adam optimizers, and batch update logic.

    Replaces single-sample manual SGD with batch-averaged Adam updates.
    This reduces gradient variance by ~sqrt(batch_size) and preserves
    optimizer momentum across updates.

    The buffer accumulates complete trajectory dicts. After every batch_size
    trajectories, the caller triggers a batch update via update_metapolicy_batch
    and/or update_reward_batch.

    Input (constructor):
    - meta_mlp:        metacontroller decision MLP (nn.Module)
    - reward_mlp:      reward head MLP (nn.Module)
    - rf_predictor:    reward feature predictor MLP (nn.Module)
    - lr:              learning rate for Adam (default 3e-4)
    - eps:             Adam epsilon for numerical stability (default 1e-5)
    - max_grad_norm:   gradient clipping threshold (default 0.5)
    - batch_size:      trajectories per batch update (default 8)
    - buffer_capacity: maximum buffer size before oldest evicted (default 10000)
    """

    def __init__(
        self,
        meta_mlp,
        reward_mlp,
        rf_predictor,
        lr=DEFAULT_LR,
        eps=DEFAULT_EPS,
        max_grad_norm=DEFAULT_MAX_GRAD_NORM,
        batch_size=DEFAULT_BATCH_SIZE,
        buffer_capacity=DEFAULT_BUFFER_CAPACITY,
        jsonl_dir=None,
    ):
        self.lr = lr
        self.eps = eps
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity

        # Replay buffer (BATCH-01): fixed-capacity FIFO with O(1) append
        self.buffer = deque(maxlen=buffer_capacity)

        # Separate counter for batch trigger (Pitfall 2 from RESEARCH.md:
        # track count separately, not len(buffer) % batch_size)
        self.trajectories_since_update = 0

        # Gradient step counter for entropy annealing
        self.step_count = 0

        # Adam optimizers (BATCH-03): one per module group
        self.optimizer_meta = Adam(meta_mlp.parameters(), lr=lr, eps=eps)
        self.optimizer_reward = Adam(
            list(reward_mlp.parameters()) + list(rf_predictor.parameters()),
            lr=lr,
            eps=eps,
        )

        # JSONL output for dashboard collector (D-05)
        self._jsonl_file = None
        if jsonl_dir is not None:
            jsonl_dir = Path(jsonl_dir)
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
            jsonl_path = jsonl_dir / f"{session_id}.jsonl"
            self._jsonl_file = open(jsonl_path, "a", encoding="utf-8")
            logging.info("JSONL output: %s", jsonl_path)

        # Dashboard WS param receiver (D-06) -- initialized separately via start_param_receiver()
        self._param_receiver = None

    def add_trajectory(self, trajectory_dict):
        """
        Buffer a complete trajectory. Returns True if batch update should fire.

        Input:
        - trajectory_dict: dict with keys meta_trajectory, realized_return,
          is_fallback, nodes_expanded, token_duration_frames, rollout

        Output:
        - ready: bool, True if trajectories_since_update >= batch_size
        """
        self.buffer.append(trajectory_dict)
        self.trajectories_since_update += 1
        return self.trajectories_since_update >= self.batch_size

    def get_batch(self):
        """
        Return the last batch_size entries from the buffer and reset the counter.

        Output:
        - batch: list of trajectory dicts (length = batch_size)
        """
        batch = list(self.buffer)[-self.batch_size:]
        self.trajectories_since_update = 0
        return batch

    def _write_jsonl(self, row: dict) -> None:
        """Append a JSON line to the JSONL output file for dashboard ingestion."""
        if self._jsonl_file is not None:
            self._jsonl_file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._jsonl_file.flush()

    def update_metapolicy_batch(self, meta_mlp, batch):
        """
        Batch policy gradient update over multiple trajectories.

        Computes advantages for each trajectory, normalizes across the entire
        batch (all steps pooled), then performs a single Adam step with
        gradient clipping.

        Input:
        - meta_mlp: metacontroller MLP (nn.Module)
        - batch:    list of trajectory dicts from get_batch()

        Output dict:
        - loss:          float, batch-mean policy + entropy loss
        - grad_norm:     float, gradient norm before clipping
        - clipped:       bool, True if grad_norm exceeded max_grad_norm
        - entropy_coeff: float, current entropy coefficient
        - step_count:    int, gradient step count after this update
        """

        # -----------------------------------------------------------------
        # Apply any pending hot-reload params (D-06)
        # -----------------------------------------------------------------

        pending = self.get_pending_params()
        if pending:
            if "batch_size" in pending:
                new_bs = int(pending["batch_size"])
                if new_bs >= 1:
                    self.batch_size = new_bs
                    logging.info("Hot-reload applied: batch_size -> %d", new_bs)

        # -----------------------------------------------------------------
        # Step 1: compute advantages for each trajectory in the batch
        # -----------------------------------------------------------------

        all_advantages = []
        all_meta_trajectories = []

        for traj_dict in batch:
            token_return = compute_token_return(traj_dict["rollout"])
            advantages, _ = compute_metalevel_advantages(
                traj_dict["meta_trajectory"],
                realized_return=token_return,
                is_fallback=traj_dict.get("is_fallback", False),
                nodes_expanded=traj_dict.get("nodes_expanded", 0),
                token_duration_frames=traj_dict.get("token_duration_frames", 1),
            )
            all_advantages.extend(advantages)
            all_meta_trajectories.append(
                (traj_dict["meta_trajectory"], advantages)
            )

        # -----------------------------------------------------------------
        # Step 2: normalize advantages ACROSS the entire batch
        # (all steps pooled, not per-trajectory)
        # -----------------------------------------------------------------

        if len(all_advantages) > 1:
            adv_t = torch.tensor(all_advantages, dtype=torch.float32)
            adv_mean = adv_t.mean()
            adv_std = adv_t.std()
            adv_t = (adv_t - adv_mean) / (adv_std + 1e-8)
            all_advantages = adv_t.tolist()

        # Redistribute normalized advantages back to per-trajectory lists
        idx = 0
        normalized_trajectories = []
        for meta_traj, raw_advs in all_meta_trajectories:
            n_steps = len(raw_advs)
            norm_advs = all_advantages[idx : idx + n_steps]
            normalized_trajectories.append((meta_traj, norm_advs))
            idx += n_steps

        # -----------------------------------------------------------------
        # Step 3: compute batch loss with entropy regularization
        # -----------------------------------------------------------------

        entropy_coeff = get_entropy_coeff(self.step_count)

        self.optimizer_meta.zero_grad()

        total_loss = torch.tensor(0.0)
        total_steps = 0

        for meta_traj, advantages in normalized_trajectories:
            for step, advantage in zip(meta_traj, advantages):
                features = step["features"]
                decision = step["decision"]

                logits = meta_mlp(features)
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(decision))
                entropy = dist.entropy()

                step_loss = -(log_prob * advantage)
                entropy_loss = -entropy_coeff * entropy
                total_loss = total_loss + step_loss + entropy_loss
                total_steps += 1

        # Batch-mean: divide by total number of steps across all trajectories
        if total_steps > 0:
            total_loss = total_loss / total_steps

        # -----------------------------------------------------------------
        # Step 4: backward + gradient clipping + optimizer step
        # -----------------------------------------------------------------

        total_loss.backward()

        grad_norm = clip_grad_norm_(meta_mlp.parameters(), self.max_grad_norm)
        clipped = bool(grad_norm.item() > self.max_grad_norm)

        self.optimizer_meta.step()
        self.step_count += 1

        # -----------------------------------------------------------------
        # Step 5: emit JSONL rows for dashboard collector (D-05, DASH-03)
        # -----------------------------------------------------------------

        # Aggregate decision distributions across the batch
        decision_totals = {"explore": 0, "rollback": 0, "interrupt": 0, "commit_next": 0}
        total_nodes = 0
        max_depth = 0
        for traj_dict in batch:
            for step_d in traj_dict["meta_trajectory"]:
                decision = step_d["decision"]
                # Decision mapping: 0=EXPLORE, 1=ROLLBACK, 2=INTERRUPT, 3=COMMIT_NEXT
                if decision == 0:
                    decision_totals["explore"] += 1
                elif decision == 1:
                    decision_totals["rollback"] += 1
                elif decision == 2:
                    decision_totals["interrupt"] += 1
                elif decision == 3:
                    decision_totals["commit_next"] += 1
            total_nodes += traj_dict.get("nodes_expanded", 0)
            max_depth = max(max_depth, len(traj_dict["meta_trajectory"]))

        # Write decision_counts JSONL row
        self._write_jsonl({
            "type": "decision_counts",
            "step": self.step_count,
            "explore": decision_totals["explore"],
            "rollback": decision_totals["rollback"],
            "interrupt": decision_totals["interrupt"],
            "commit_next": decision_totals["commit_next"],
            "nodes_expanded": total_nodes // max(1, len(batch)),
            "search_depth": max_depth,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Compute mean episode return from batch
        batch_returns = [compute_token_return(traj_dict["rollout"]) for traj_dict in batch]
        mean_episode_return = sum(batch_returns) / max(1, len(batch_returns))

        # Write per-batch metric JSONL row
        self._write_jsonl({
            "type": "metric",
            "step": self.step_count,
            "loss": total_loss.item(),
            "reward": mean_episode_return,
            "episode_return": mean_episode_return,
            "grad_norm": grad_norm.item(),
            "clipped": clipped,
            "lr": self.optimizer_meta.param_groups[0]["lr"],
            "timestamp": datetime.utcnow().isoformat(),
        })

        return {
            "loss": total_loss.item(),
            "grad_norm": grad_norm.item(),
            "clipped": clipped,
            "entropy_coeff": entropy_coeff,
            "step_count": self.step_count,
        }

    def update_reward_batch(self, reward_mlp, rf_predictor, batch, encode_fn, hidden_dim=128):
        """
        Batch update for reward head and reward feature predictor.

        Reproduces the forward passes from train_reward_head() for each
        trajectory in the batch, accumulates losses, then performs a single
        Adam step with gradient clipping.

        Input:
        - reward_mlp:    reward head MLP (nn.Module)
        - rf_predictor:  reward feature predictor MLP (nn.Module)
        - batch:         list of trajectory dicts from get_batch()
        - encode_fn:     callable(gta_state) -> z_t [1, fused_dim]
        - hidden_dim:    hidden layer size (default 128)

        Output dict:
        - reward_loss: float, batch-mean reward prediction loss
        - rf_loss:     float, batch-mean reward feature prediction loss
        - grad_norm:   float, gradient norm before clipping
        - clipped:     bool, True if grad_norm exceeded max_grad_norm
        """

        self.optimizer_reward.zero_grad()

        total_reward_loss = torch.tensor(0.0)
        total_rf_loss = torch.tensor(0.0)

        for traj_dict in batch:
            rollout = traj_dict["rollout"]
            token_return = compute_token_return(rollout)

            state_before = rollout["state_before"]
            state_after = rollout["state_after"]
            duration = rollout["duration"]

            # Encode real states
            z_parent = encode_fn(state_before).detach()
            z_child = encode_fn(state_after).detach()

            fused_dim = z_parent.shape[-1]

            # Extract real reward features
            rf_parent = extract_reward_features(state_before)
            rf_child = extract_reward_features(state_after)

            # Build duration and time_left tensors
            duration_t = torch.tensor([[float(duration)]], dtype=torch.float32)
            time_left_t = torch.tensor([[0.0]], dtype=torch.float32)

            # Forward pass through reward head
            r_pred, reward_mlp = reward_head(
                z_parent=z_parent,
                z_child=z_child,
                rf_parent=rf_parent,
                rf_child=rf_child,
                duration=duration_t,
                time_left=time_left_t,
                reward_mlp=reward_mlp,
                fused_dim=fused_dim,
                hidden_dim=hidden_dim,
            )

            # Forward pass through rf_predictor
            delta_z = z_child - z_parent
            rf_pred, rf_predictor = predict_reward_features(
                z_parent=z_parent,
                delta_z=delta_z,
                rf_parent=rf_parent,
                rf_predictor=rf_predictor,
                fused_dim=fused_dim,
                hidden_dim=hidden_dim // 2,
            )

            # Losses
            target_return = torch.tensor([[token_return]], dtype=torch.float32)
            reward_loss = (r_pred - target_return) ** 2
            rf_loss = ((rf_pred - rf_child.detach()) ** 2).mean()

            total_reward_loss = total_reward_loss + reward_loss
            total_rf_loss = total_rf_loss + rf_loss

        # Batch-mean
        n = len(batch)
        total_loss = (total_reward_loss + total_rf_loss) / n

        total_loss.backward()

        all_params = list(reward_mlp.parameters()) + list(rf_predictor.parameters())
        grad_norm = clip_grad_norm_(all_params, self.max_grad_norm)
        clipped = bool(grad_norm.item() > self.max_grad_norm)

        self.optimizer_reward.step()

        return {
            "reward_loss": (total_reward_loss / n).item(),
            "rf_loss": (total_rf_loss / n).item(),
            "grad_norm": grad_norm.item(),
            "clipped": clipped,
        }

    def save_checkpoint(self, checkpoint_dir, meta_mlp, reward_mlp, rf_predictor,
                        intuition_mlp=None, token_embed=None, planner_mlp=None,
                        session_id=None):
        """
        Save per-module checkpoint files (BATCH-05).

        Each module gets its own .pt file. Trained modules (meta_mlp,
        reward_mlp, rf_predictor) include both model_state_dict and
        optimizer_state_dict. Untrained modules (intuition_mlp, token_embed,
        planner_mlp) save state_dict only for model reconstruction.

        Per BATCH-05: "one .pt per module per session" -- all 6 modules
        get separate files.

        Input:
        - checkpoint_dir: str or Path, base checkpoint directory
        - meta_mlp: the metacontroller MLP (required)
        - reward_mlp: reward head MLP (required)
        - rf_predictor: reward feature predictor MLP (required)
        - intuition_mlp: intuition head MLP (optional, state_dict-only save)
        - token_embed: token embedding table (optional, state_dict-only save)
        - planner_mlp: action planner MLP (optional, state_dict-only save)
        - session_id: str or None (auto-generated from timestamp if None)

        Output:
        - dict with "session_dir" path and "files_saved" list
        """
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        session_dir = Path(checkpoint_dir) / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        files_saved = []

        # --- Trained modules: model + optimizer state ---

        # Meta MLP checkpoint
        meta_path = session_dir / "meta_mlp.pt"
        torch.save({
            "model_state_dict": meta_mlp.state_dict(),
            "optimizer_state_dict": self.optimizer_meta.state_dict(),
            "step_count": self.step_count,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
        }, meta_path)
        files_saved.append(str(meta_path))

        # Reward MLP checkpoint (separate file per BATCH-05)
        reward_mlp_path = session_dir / "reward_mlp.pt"
        torch.save({
            "model_state_dict": reward_mlp.state_dict(),
        }, reward_mlp_path)
        files_saved.append(str(reward_mlp_path))

        # RF Predictor checkpoint (separate file per BATCH-05)
        rf_predictor_path = session_dir / "rf_predictor.pt"
        torch.save({
            "model_state_dict": rf_predictor.state_dict(),
        }, rf_predictor_path)
        files_saved.append(str(rf_predictor_path))

        # Shared optimizer for reward_mlp + rf_predictor
        optimizer_reward_path = session_dir / "optimizer_reward.pt"
        torch.save({
            "optimizer_state_dict": self.optimizer_reward.state_dict(),
            "step_count": self.step_count,
        }, optimizer_reward_path)
        files_saved.append(str(optimizer_reward_path))

        # --- Untrained modules: state_dict only (for model reconstruction) ---

        if intuition_mlp is not None:
            intuition_path = session_dir / "intuition_mlp.pt"
            torch.save({
                "model_state_dict": intuition_mlp.state_dict(),
            }, intuition_path)
            files_saved.append(str(intuition_path))

        if token_embed is not None:
            token_embed_path = session_dir / "token_embed.pt"
            torch.save({
                "model_state_dict": token_embed.state_dict(),
            }, token_embed_path)
            files_saved.append(str(token_embed_path))

        if planner_mlp is not None:
            planner_path = session_dir / "planner_mlp.pt"
            torch.save({
                "model_state_dict": planner_mlp.state_dict(),
            }, planner_path)
            files_saved.append(str(planner_path))

        # Training state metadata
        state_path = session_dir / "training_state.pt"
        torch.save({
            "step_count": self.step_count,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "buffer_size": len(self.buffer),
            "trajectories_since_update": self.trajectories_since_update,
        }, state_path)
        files_saved.append(str(state_path))

        return {"session_dir": str(session_dir), "files_saved": files_saved}

    def load_checkpoint(self, session_dir, meta_mlp, reward_mlp, rf_predictor,
                        intuition_mlp=None, token_embed=None, planner_mlp=None):
        """
        Load from a checkpoint directory (BATCH-06).

        Restores model weights, optimizer state (momentum/variance),
        and step count. Buffer is NOT restored (starts empty on resume
        per RESEARCH.md Open Question 1 -- RESOLVED: No).

        Uses map_location='cpu' for robustness (Pitfall 4).

        Input:
        - session_dir: str or Path, path to session_YYYYMMDD_HHMMSS directory
        - meta_mlp: the metacontroller MLP (weights will be overwritten)
        - reward_mlp: reward head MLP (weights will be overwritten)
        - rf_predictor: reward feature predictor MLP (weights will be overwritten)
        - intuition_mlp: intuition head MLP (optional, weights overwritten if file exists)
        - token_embed: token embedding table (optional, weights overwritten if file exists)
        - planner_mlp: action planner MLP (optional, weights overwritten if file exists)

        Output:
        - dict with "loaded" bool, "step_count" int, "files_loaded" list
        """
        session_dir = Path(session_dir)
        files_loaded = []

        # Load meta MLP (model + optimizer)
        meta_path = session_dir / "meta_mlp.pt"
        if meta_path.exists():
            ckpt = torch.load(meta_path, map_location="cpu", weights_only=True)
            meta_mlp.load_state_dict(ckpt["model_state_dict"])
            self.optimizer_meta.load_state_dict(ckpt["optimizer_state_dict"])
            self.step_count = ckpt["step_count"]
            files_loaded.append(str(meta_path))

        # Load reward MLP (model only)
        reward_mlp_path = session_dir / "reward_mlp.pt"
        if reward_mlp_path.exists():
            ckpt = torch.load(reward_mlp_path, map_location="cpu", weights_only=True)
            reward_mlp.load_state_dict(ckpt["model_state_dict"])
            files_loaded.append(str(reward_mlp_path))

        # Load RF Predictor (model only)
        rf_predictor_path = session_dir / "rf_predictor.pt"
        if rf_predictor_path.exists():
            ckpt = torch.load(rf_predictor_path, map_location="cpu", weights_only=True)
            rf_predictor.load_state_dict(ckpt["model_state_dict"])
            files_loaded.append(str(rf_predictor_path))

        # Load shared optimizer for reward_mlp + rf_predictor
        optimizer_reward_path = session_dir / "optimizer_reward.pt"
        if optimizer_reward_path.exists():
            ckpt = torch.load(optimizer_reward_path, map_location="cpu", weights_only=True)
            self.optimizer_reward.load_state_dict(ckpt["optimizer_state_dict"])
            files_loaded.append(str(optimizer_reward_path))

        # Load untrained modules (state_dict only)
        if intuition_mlp is not None:
            intuition_path = session_dir / "intuition_mlp.pt"
            if intuition_path.exists():
                ckpt = torch.load(intuition_path, map_location="cpu", weights_only=True)
                intuition_mlp.load_state_dict(ckpt["model_state_dict"])
                files_loaded.append(str(intuition_path))

        if token_embed is not None:
            token_embed_path = session_dir / "token_embed.pt"
            if token_embed_path.exists():
                ckpt = torch.load(token_embed_path, map_location="cpu", weights_only=True)
                token_embed.load_state_dict(ckpt["model_state_dict"])
                files_loaded.append(str(token_embed_path))

        if planner_mlp is not None:
            planner_path = session_dir / "planner_mlp.pt"
            if planner_path.exists():
                ckpt = torch.load(planner_path, map_location="cpu", weights_only=True)
                planner_mlp.load_state_dict(ckpt["model_state_dict"])
                files_loaded.append(str(planner_path))

        loaded = len(files_loaded) > 0
        return {"loaded": loaded, "step_count": self.step_count, "files_loaded": files_loaded}

    def start_param_receiver(self, module_name: str = "metacontroller"):
        """Start a background WebSocket client for hot-reload from dashboard (D-06).

        Runs in a daemon thread. If the dashboard server is not running,
        silently retries every 10 seconds. When a param_update message
        arrives, it updates the optimizer lr directly and queues other
        params (entropy_coeff, think_cost, batch_size) for the training
        loop to read via get_pending_params().

        T-05-22 mitigation: All param changes logged via logging.info.
        """
        host = os.environ.get("DASHBOARD_HOST", "localhost")
        port = os.environ.get("DASHBOARD_PORT", "8000")
        ws_url = f"ws://{host}:{port}/ws/train"

        self._pending_params = {}
        self._param_lock = threading.Lock()
        self._param_stop = threading.Event()

        def _ws_loop():
            import websockets.sync.client as ws_client
            while not self._param_stop.is_set():
                try:
                    with ws_client.connect(ws_url, close_timeout=5) as ws:
                        ws.send(json.dumps({"type": "register", "module": module_name}))
                        logging.info("Connected to dashboard WS | module=%s", module_name)
                        while not self._param_stop.is_set():
                            try:
                                msg = ws.recv(timeout=2.0)
                                data = json.loads(msg)
                                if data.get("type") == "param_update":
                                    params = data.get("params", {})
                                    target = params.get("module", "")
                                    if target and target != module_name:
                                        continue
                                    with self._param_lock:
                                        if "lr" in params:
                                            new_lr = float(params["lr"])
                                            if new_lr > 0:
                                                for pg in self.optimizer_meta.param_groups:
                                                    pg["lr"] = new_lr
                                                logging.info("Hot-reload: lr -> %g", new_lr)
                                        for k in ("entropy_coeff", "think_cost", "batch_size"):
                                            if k in params:
                                                self._pending_params[k] = params[k]
                                                logging.info("Hot-reload: %s queued = %s", k, params[k])
                            except TimeoutError:
                                continue
                except Exception as exc:
                    if not self._param_stop.is_set():
                        logging.debug("Dashboard WS unavailable: %s (retry 10s)", exc)
                        self._param_stop.wait(10.0)

        t = threading.Thread(target=_ws_loop, daemon=True, name="dashboard-ws-meta")
        t.start()
        self._param_receiver = t
        logging.info("Dashboard param receiver started | url=%s", ws_url)

    def get_pending_params(self) -> dict:
        """Retrieve and clear pending hot-reload params."""
        if not hasattr(self, '_param_lock'):
            return {}
        with self._param_lock:
            params = self._pending_params.copy()
            self._pending_params.clear()
        return params

    def close(self):
        """Cleanup JSONL file and WS receiver."""
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if hasattr(self, '_param_stop'):
            self._param_stop.set()
