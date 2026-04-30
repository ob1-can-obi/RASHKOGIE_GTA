"""
Small reusable multi-head attention helper for the driving model.

This file is intentionally simple:
- one function to create the trainable weight matrices
- one function to run multi-head attention with explicit matrix multiplies
"""

import math

import torch
from torch import nn


def create_multi_head_attention_weights(query_dim: int, embed_dim: int):
    """
    Create the trainable projection weights used by multi-head attention.

    Shapes:
    - qw: query projection  [query_dim, embed_dim]
    - kw: key projection    [embed_dim, embed_dim]
    - vw: value projection  [embed_dim, embed_dim]
    - ow: output projection [embed_dim, embed_dim]
    """
    qw = nn.Parameter(torch.empty(query_dim, embed_dim))
    kw = nn.Parameter(torch.empty(embed_dim, embed_dim))
    vw = nn.Parameter(torch.empty(embed_dim, embed_dim))
    ow = nn.Parameter(torch.empty(embed_dim, embed_dim))

    nn.init.xavier_uniform_(qw)
    nn.init.xavier_uniform_(kw)
    nn.init.xavier_uniform_(vw)
    nn.init.xavier_uniform_(ow)

    return qw, kw, vw, ow


def multi_head_attention(query_input, entity_embs, mask, qw, kw, vw, ow, num_heads: int):
    """
    Run explicit multi-head attention.

    Inputs:
    - query_input: [batch, query_dim]
    - entity_embs: [batch, num_entities, embed_dim]
    - mask:        [batch, num_entities]

    Returned tensors:
    - q, k, v: projected tensors before head splitting
    - q_heads, k_heads, v_heads: split-by-head tensors
    - scores: attention scores before softmax
    - attention: normalized attention weights
    - merged_context: concatenated head outputs before final output projection
    - entity_context: final attention output after output projection
    """
    batch_size = query_input.shape[0]
    num_entities = entity_embs.shape[1]
    embed_dim = entity_embs.shape[2]

    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        )

    head_dim = embed_dim // num_heads

    # Project into the standard Q, K, V tensors.
    q = query_input @ qw
    k = entity_embs @ kw
    v = entity_embs @ vw

    # Split the full embedding into smaller head-sized chunks.
    #
    # q: [B, D]      -> [B, H, 1, Hd]
    # k: [B, N, D]   -> [B, H, N, Hd]
    # v: [B, N, D]   -> [B, H, N, Hd]
    q_heads = q.view(batch_size, num_heads, head_dim).unsqueeze(2)
    k_heads = k.view(batch_size, num_entities, num_heads, head_dim).permute(0, 2, 1, 3)
    v_heads = v.view(batch_size, num_entities, num_heads, head_dim).permute(0, 2, 1, 3)

    # Standard scaled dot-product attention:
    # scores = Q K^T / sqrt(head_dim)
    scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) / math.sqrt(head_dim)

    # Expand the mask so it can be applied to every head.
    head_mask = mask.unsqueeze(1).unsqueeze(2)

    # Ignore padded entities.
    scores = scores.masked_fill(head_mask <= 0.0, -1e9)

    # Convert scores into probabilities.
    attention = torch.softmax(scores, dim=-1)

    # Re-apply the mask and normalize again for numerical safety.
    attention = attention * head_mask
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    # Weighted sum over values for each head.
    head_context = torch.matmul(attention, v_heads)

    # Merge all heads back into one full embedding.
    merged_context = head_context.squeeze(2).reshape(batch_size, embed_dim)

    # Final output projection.
    entity_context = merged_context @ ow

    return {
        "q": q,
        "k": k,
        "v": v,
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "scores": scores,
        "attention": attention,
        "merged_context": merged_context,
        "entity_context": entity_context,
        "head_dim": head_dim,
    }
