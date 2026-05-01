"""
Action planner.

Takes the current embedding and the predicted next embedding (from the
intuition head) and produces a softmax distribution over the token vocabulary.

The intuition head is NOT called here — that is the caller's responsibility.
Each module trains independently:

    intuition_head  → trained separately, produces z_next_pred
    action_planner  → trained separately, consumes z_next_pred as a frozen feature

Flow:
    z_t + prev_token  -->  intuition_head  -->  z_next_pred   (caller's job)
    [z_t | z_next_pred]  -->  planner_mlp (2 hidden layers)  -->  logits  -->  softmax  -->  top-k
"""

import torch
from torch import nn


def action_planner(
    z_t,
    z_next_pred,
    vocab_size,
    planner_mlp=None,
    hidden_dim=128,
    top_k=3,
):
    """
    Produce a distribution over the token vocabulary from current + predicted state.

    The intuition head must be run by the caller before this function.
    z_next_pred is detached here so no gradient flows back into the intuition head.

    Input:
    - z_t:         current fused embedding, shape [batch, fused_dim]
    - z_next_pred: predicted next embedding from intuition head, shape [batch, fused_dim]
    - vocab_size:  number of tokens in the tokenizer vocabulary
    - planner_mlp: the trainable planner MLP (None = create fresh)
                   Architecture: Linear(256,256) -> ReLU -> Linear(256,128) -> ReLU -> Linear(128,V)
    - hidden_dim:  hidden layer size
    - top_k:       how many candidates to return

    Output dict:
    - logits:      raw scores before softmax, shape [batch, vocab_size]
    - token_probs: full softmax distribution, shape [batch, vocab_size]
    - top_k_ids:   token ids of top-k candidates, shape [batch, top_k]
    - top_k_probs: probabilities of top-k candidates, shape [batch, top_k]
    - planner_mlp: the trainable planner MLP (pass back in for reuse)
    """

    fused_dim = z_t.shape[-1]

    # -------------------------------------------------------------------------
    # Step 1: combine current and predicted future
    #
    # z_next_pred is detached — the planner treats it as a fixed feature.
    # Gradients must not flow back into the intuition head from here.
    # -------------------------------------------------------------------------

    planner_input = torch.cat([z_t, z_next_pred.detach()], dim=-1)
    # shape: [batch, fused_dim * 2]

    # -------------------------------------------------------------------------
    # Step 2: create the planner MLP if one was not passed in
    # -------------------------------------------------------------------------

    if planner_mlp is None:
        planner_mlp = nn.Sequential(
            nn.Linear(fused_dim * 2, hidden_dim * 2),  # 256 -> 256
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),      # 256 -> 128
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),           # 128 -> V
        )

    # -------------------------------------------------------------------------
    # Step 3: produce logits over the full token vocabulary
    # -------------------------------------------------------------------------

    logits = planner_mlp(planner_input)

    # -------------------------------------------------------------------------
    # Step 4: softmax → top-k
    # -------------------------------------------------------------------------

    token_probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(token_probs, k=top_k, dim=-1)

    return {
        "logits":      logits,
        "token_probs": token_probs,
        "top_k_ids":   top_k_ids,
        "top_k_probs": top_k_probs,
        "planner_mlp": planner_mlp,
    }
