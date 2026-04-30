"""
Intuition head — predicts the next fused embedding from current state
and previous action token.

Used by both the action planner and the metacontroller.

Token 0 (rs0_ls0_ft0_b0) is the idle token — used as the previous action
on the first frame when no action has been taken yet.
"""

import torch
from torch import nn

IDLE_TOKEN_ID = 0


def intuition_head(
    z_t,
    prev_token_id,
    vocab_size,
    token_embed=None,
    intuition_mlp=None,
    embed_dim=32,
    hidden_dim=128,
):
    """
    Predict the next fused embedding from the current state and previous action.

    Input:
    - z_t: current fused embedding, shape [batch, fused_dim]
    - prev_token_id: previous action token, shape [batch] (long)
    - vocab_size: number of tokens in the vocabulary

    Output:
    - z_next_pred: predicted next fused embedding, shape [batch, fused_dim]
    - delta_z: predicted change, shape [batch, fused_dim]
    - token_embed: the trainable token embedding table (pass back in for reuse)
    - intuition_mlp: the trainable MLP (pass back in for reuse)
    """

    fused_dim = z_t.shape[-1]

    if token_embed is None:
        token_embed = nn.Embedding(vocab_size, embed_dim)

    if intuition_mlp is None:
        intuition_mlp = nn.Sequential(
            nn.Linear(fused_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fused_dim),
        )

    prev_emb = token_embed(prev_token_id)
    intuition_input = torch.cat([z_t, prev_emb], dim=-1)

    delta_z = intuition_mlp(intuition_input)
    z_next_pred = z_t + delta_z

    return z_next_pred, delta_z, token_embed, intuition_mlp
