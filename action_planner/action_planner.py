"""
Action planner.

Uses the intuition head to predict the future state, then combines current
and predicted future z_t to produce a softmax distribution over the token
vocabulary.

Flow:
    z_t + prev_token  -->  intuition_head  -->  z_next_pred
    [z_t | z_next_pred]  -->  planner_mlp  -->  logits  -->  softmax  -->  top-k
"""

import sys
from pathlib import Path

import torch
from torch import nn

INTUITION_HEAD_DIR = Path(__file__).resolve().parent.parent / "intuition_head"
if str(INTUITION_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(INTUITION_HEAD_DIR))

from intuition_head import IDLE_TOKEN_ID, intuition_head


def action_planner(
    z_t,
    vocab_size,
    prev_token_id=None,
    token_embed=None,
    intuition_mlp=None,
    planner_mlp=None,
    intuition_embed_dim=32,
    hidden_dim=128,
    top_k=3,
):
    """
    Input:
    - z_t: fused embedding tensor, shape [batch, fused_dim]
    - vocab_size: number of tokens in the tokenizer vocabulary
    - prev_token_id: previous action token, shape [batch] (long)
                     defaults to IDLE_TOKEN_ID on first frame

    Output dict:
    - z_next_pred: predicted next embedding, shape [batch, fused_dim]
    - delta_z: predicted change, shape [batch, fused_dim]
    - logits: raw scores before softmax, shape [batch, vocab_size]
    - token_probs: full softmax distribution, shape [batch, vocab_size]
    - top_k_ids: token ids of top-k candidates, shape [batch, top_k]
    - top_k_probs: probabilities of top-k candidates, shape [batch, top_k]
    - token_embed: the trainable token embedding table (pass back in for reuse)
    - intuition_mlp: the trainable intuition MLP (pass back in for reuse)
    - planner_mlp: the trainable planner MLP (pass back in for reuse)
    """

    fused_dim = z_t.shape[-1]
    batch_size = z_t.shape[0]

    # -------------------------------------------------------------------------
    # Step 1: default to idle token on first frame
    # -------------------------------------------------------------------------

    if prev_token_id is None:
        prev_token_id = torch.full(
            (batch_size,), IDLE_TOKEN_ID, dtype=torch.long, device=z_t.device
        )

    # -------------------------------------------------------------------------
    # Step 2: predict the future state from current state + previous action
    # -------------------------------------------------------------------------

    z_next_pred, delta_z, token_embed, intuition_mlp = intuition_head(
        z_t,
        prev_token_id,
        vocab_size=vocab_size,
        token_embed=token_embed,
        intuition_mlp=intuition_mlp,
        embed_dim=intuition_embed_dim,
        hidden_dim=hidden_dim,
    )

    # -------------------------------------------------------------------------
    # Step 3: combine current and predicted future for the planner
    # -------------------------------------------------------------------------

    planner_input = torch.cat([z_t, z_next_pred], dim=-1)

    # planner_input.shape = [batch, fused_dim * 2]

    # -------------------------------------------------------------------------
    # Step 4: create the planner MLP if one was not passed in
    # -------------------------------------------------------------------------

    if planner_mlp is None:
        planner_mlp = nn.Sequential(
            nn.Linear(fused_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    # -------------------------------------------------------------------------
    # Step 5: produce logits over the full token vocabulary
    # -------------------------------------------------------------------------

    logits = planner_mlp(planner_input)

    # -------------------------------------------------------------------------
    # Step 6: softmax to get the probability distribution
    # -------------------------------------------------------------------------

    token_probs = torch.softmax(logits, dim=-1)

    # -------------------------------------------------------------------------
    # Step 7: pick the top-k most likely tokens
    # -------------------------------------------------------------------------

    top_k_probs, top_k_ids = torch.topk(token_probs, k=top_k, dim=-1)

    return {
        "z_next_pred": z_next_pred,
        "delta_z": delta_z,
        "logits": logits,
        "token_probs": token_probs,
        "top_k_ids": top_k_ids,
        "top_k_probs": top_k_probs,
        "token_embed": token_embed,
        "intuition_mlp": intuition_mlp,
        "planner_mlp": planner_mlp,
    }
