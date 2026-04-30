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

import torch


# =========================================================================
# 1. Compute realized token return
# =========================================================================

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
        if child.token_id == committed_token_id:
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

def update_metapolicy(meta_mlp, meta_trajectory, advantages, lr=1e-3):
    """
    Policy gradient update over the full metalevel trajectory.

    Each search step's decision gets reinforced or penalized based on
    its advantage.  Positive advantage = good decision, strengthen it.
    Negative advantage = bad decision, weaken it.

    Input:
    - meta_mlp: the metacontroller MLP (nn.Sequential)
    - meta_trajectory: list of dicts with "decision" and "decision_logits"
    - advantages: list of floats from compute_metalevel_advantages
    - lr: learning rate

    Output:
    - total_loss: float, sum of per-step losses (for logging)
    """

    total_loss = torch.tensor(0.0, requires_grad=True)

    for step, advantage in zip(meta_trajectory, advantages):
        logits = step["decision_logits"]       # [1, 4]
        decision = step["decision"]             # int

        log_probs = torch.log_softmax(logits, dim=-1)
        log_prob_taken = log_probs[0, decision]

        step_loss = -(log_prob_taken * advantage)
        total_loss = total_loss + step_loss

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
    )

    # step 4: weight update
    total_loss = update_metapolicy(
        meta_mlp, meta_trajectory, advantages, lr=lr,
    )

    return {
        "token_return": token_return,
        "advantages": advantages,
        "meta_returns": meta_returns,
        "total_loss": total_loss,
        "n_search_steps": len(meta_trajectory),
    }
