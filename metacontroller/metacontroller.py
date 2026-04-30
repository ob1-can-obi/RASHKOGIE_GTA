"""
Metacontroller — decides whether to KEEP, INTERRUPT, COMMIT_NEXT, or ROLLBACK
given the current real state, the running action token, and search tree
results.

Inputs:
- z_t: latest real fused embedding [batch, fused_dim]
- z_running: predicted state when current token was committed [batch, fused_dim]
- running_token_id: token currently being executed [batch]
- elapsed_ratio: how much of the running token's duration has passed [batch, 1]
  (0.0 = just started, 1.0 = fully elapsed)
- candidate_q: Q-values from the search tree for each candidate [batch, k]
- candidate_ids: token ids for each candidate [batch, k]
- urgency: time-pressure scalar [batch, 1]  (0 = no rush, 1 = critical)
- parent_unexplored: fraction of unopened siblings at the parent node [batch, 1]
  (0.0 = all explored, 1.0 = none explored yet)

Output:
- decision: 0 = KEEP, 1 = INTERRUPT, 2 = COMMIT_NEXT, 3 = ROLLBACK  [batch]
- decision_logits: raw scores for the four choices [batch, 4]
- selected_token_id: which token to switch to (only meaningful when
  decision is INTERRUPT or COMMIT_NEXT) [batch]
- meta_mlp: the trainable MLP (pass back in for reuse)
"""

import torch
from torch import nn

KEEP = 0
INTERRUPT = 1
COMMIT_NEXT = 2
ROLLBACK = 3


def metacontroller(
    z_t,
    z_running,
    running_token_id,
    elapsed_ratio,
    candidate_q,
    candidate_ids,
    urgency,
    parent_unexplored,
    meta_mlp=None,
    hidden_dim=128,
):
    """
    Decide whether to keep, interrupt, commit next, or rollback.

    Input:
    - z_t: current real fused embedding, shape [batch, fused_dim]
    - z_running: predicted state when current token was committed,
      shape [batch, fused_dim]
    - running_token_id: currently executing token, shape [batch] (long)
      (unused by MLP — kept for caller context / logging)
    - elapsed_ratio: fraction of token duration elapsed, shape [batch, 1]
    - candidate_q: Q-values from search tree, shape [batch, k]
    - candidate_ids: token ids from search tree, shape [batch, k] (long)
    - urgency: time-pressure scalar, shape [batch, 1]
    - parent_unexplored: fraction of unopened siblings at parent node,
      shape [batch, 1]  (0.0 = all explored, 1.0 = none explored)

    Output dict:
    - decision: KEEP(0) / INTERRUPT(1) / COMMIT_NEXT(2) / ROLLBACK(3), shape [batch]
    - decision_logits: raw scores, shape [batch, 4]
    - selected_token_id: best candidate token id, shape [batch]
    - meta_mlp: the trainable MLP (pass back in for reuse)
    """

    fused_dim = z_t.shape[-1]
    k = candidate_q.shape[-1]

    # -------------------------------------------------------------------------
    # Step 1: compute drift — how much real state diverged from prediction
    # -------------------------------------------------------------------------

    drift = z_t - z_running  # [batch, fused_dim]

    # -------------------------------------------------------------------------
    # Step 2: summarize candidate quality from the search tree
    # -------------------------------------------------------------------------

    best_q, best_idx = candidate_q.max(dim=-1, keepdim=True)  # [batch, 1]
    mean_q = candidate_q.mean(dim=-1, keepdim=True)            # [batch, 1]

    # -------------------------------------------------------------------------
    # Step 3: assemble the feature vector for the decision MLP
    # -------------------------------------------------------------------------

    # drift:              [batch, fused_dim]  — how wrong the prediction was
    # elapsed_ratio:      [batch, 1]          — how far into the current token
    # best_q:             [batch, 1]          — best alternative available
    # mean_q:             [batch, 1]          — average alternative quality
    # urgency:            [batch, 1]          — time pressure
    # parent_unexplored:  [batch, 1]          — unexplored siblings at parent

    features = torch.cat(
        [drift, elapsed_ratio, best_q, mean_q, urgency, parent_unexplored],
        dim=-1,
    )  # [batch, fused_dim + 5]

    input_dim = fused_dim + 5

    # -------------------------------------------------------------------------
    # Step 4: create the decision MLP if one was not passed in
    # -------------------------------------------------------------------------

    if meta_mlp is None:
        meta_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    # -------------------------------------------------------------------------
    # Step 5: produce logits over the three decisions
    # -------------------------------------------------------------------------

    decision_logits = meta_mlp(features)  # [batch, 4]

    # -------------------------------------------------------------------------
    # Step 6: pick the decision with highest score
    # -------------------------------------------------------------------------

    decision = decision_logits.argmax(dim=-1)  # [batch]

    # -------------------------------------------------------------------------
    # Step 7: select the best candidate token from the search tree
    # -------------------------------------------------------------------------

    best_idx_flat = best_idx.squeeze(-1)  # [batch]
    selected_token_id = candidate_ids.gather(
        1, best_idx_flat.unsqueeze(-1)
    ).squeeze(-1)  # [batch]

    return {
        "decision": decision,
        "decision_logits": decision_logits,
        "selected_token_id": selected_token_id,
        "meta_mlp": meta_mlp,
    }
