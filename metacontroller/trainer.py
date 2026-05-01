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

import sys
import math
from pathlib import Path

import torch
from torch.distributions import Categorical

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

    # step 4: weight update
    total_loss = update_metapolicy(
        meta_mlp, meta_trajectory, advantages, lr=lr,
        entropy_coeff=entropy_coeff,
    )

    return {
        "token_return":   token_return,
        "advantages":     advantages,
        "meta_returns":   meta_returns,
        "total_loss":     total_loss,
        "n_search_steps": len(meta_trajectory),
        "is_fallback":    is_fallback,
        "entropy_coeff":  entropy_coeff,
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
