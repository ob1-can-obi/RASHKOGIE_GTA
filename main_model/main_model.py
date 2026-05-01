"""
Main model — GTA state encoder and top-level pipeline integration.

Public API:
    encode_state(raw_state, weights)       → z_t [1, 128]
    get_candidates(z_t, prev_token_id, …)  → top-k token ids + probs
    run_token(token_id, raw_state, …)      → next_token_id + training results

Architecture:
    ego [46]         → ego_mlp    → ego_emb   [64]
    scene [16]       → scene_mlp  → scene_emb [64]
    route [14]       → route_mlp  → route_emb [64]
    entities [32,24] → entity_mlp → entity_embs [32,64]

    [ego_emb | scene_emb | route_emb] → attn_block_1(K,V=entity_embs) → LN → ctx1 [64]
    ctx1 → attn_block_2(K,V=entity_embs) + ctx1 → LN → entity_context [64]

    [ego_emb | scene_emb | route_emb | entity_context] → fusion_mlp → z_t [128]

    z_t + prev_token_id  →  intuition_head  →  z_next_pred [128]
    [z_t | z_next_pred]  →  action_planner  →  top-k token ids + probs
    top-k + z_t          →  drive_token     →  next_token_id  (metacontroller loop)
"""

import sys
from pathlib import Path

import torch
from torch import nn

# ---------------------------------------------------------------------------
# resolve sibling module paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent

for _d in ("intuition_head", "action_planner", "metacontroller"):
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from intuition_head import intuition_head
from action_planner import action_planner
from frame_loop import drive_token
from multi_head_attention import create_multi_head_attention_weights, multi_head_attention


# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------

ego_fields = [
    "px", "py", "pz",
    "vx", "vy", "vz",
    "rx", "ry", "rz",
    "hp", "hp_max", "armor", "dead",
    "v_speed", "v_throttle", "v_brake", "v_steer",
    "u_throttle", "u_brake", "u_steer", "u_handbrake",
    "agent_throttle", "agent_brake", "agent_steer", "agent_handbrake",
    "v_rpm", "v_gear", "v_max_gear", "v_steer_angle", "v_clutch",
    "v_wheel_speed", "v_accel", "v_heading",
    "v_engine_hp", "v_body_hp", "v_engine_on", "v_turbo", "v_class",
    "v_ax", "v_ay", "v_az",
    "v_right_x", "v_right_y",
    "v_dim_w", "v_dim_l", "v_dim_h",
]

scene_fields = [
    "clock_h", "clock_m", "clock_s",
    "weather", "rain", "wind_speed", "wind_x", "wind_y",
    "wanted",
    "near_total", "near_entities_kept",
    "near_vehicle_count", "near_ped_count", "near_object_count",
    "agent_active", "agent_age_ms",
]

route_fields = [
    "road_heading", "road_dist", "lane_offset", "lane_heading_delta",
    "road_lanes", "road_node2_heading", "road_node2_dist", "road_curve",
    "wp_x", "wp_y", "wp_dist",
    "v_heading", "v_fwd_x", "v_fwd_y",
]

entity_fields = [
    "type_id", "bucket_id",
    "dist", "dist3d",
    "rel_fwd", "rel_lat", "rel_z",
    "speed", "rel_v_fwd", "rel_v_lat",
    "heading", "hdiff",
    "dim_w", "dim_l", "dim_h",
    "is_static", "has_collision", "is_visible", "is_on_roadside",
    "ttc",
    "vx", "vy", "vz",
    "attached",
]

# Derived dims — read once so the rest of the file can reference them directly.
EGO_DIM      = len(ego_fields)     # 46
SCENE_DIM    = len(scene_fields)   # 16
ROUTE_DIM    = len(route_fields)   # 14
ENTITY_DIM   = len(entity_fields)  # 24
MAX_ENTITIES = 32

# Default network dims
EMBED_DIM  = 64
HIDDEN_DIM = 128
FUSED_DIM  = 128
NUM_HEADS  = 4


# ---------------------------------------------------------------------------
# build_state_tensors
# ---------------------------------------------------------------------------

def _to_float(v):
    if v is True:
        return 1.0
    if v is False or v is None:
        return 0.0
    return float(v)


def build_state_tensors(raw_state):
    """
    Convert a raw GTA state dict into batched tensors.

    Input:
    - raw_state: dict from GTA stream

    Output dict (all batched [1, ...]):
    - ego:      [1, 46]
    - scene:    [1, 16]
    - route:    [1, 14]
    - entities: [1, 32, 24]
    - mask:     [1, 32]
    """
    ego = torch.tensor(
        [_to_float(raw_state.get(f, 0.0)) for f in ego_fields],
        dtype=torch.float32,
    ).unsqueeze(0)

    scene_counts = {
        "near_vehicle_count": len(raw_state.get("near_vehs",    [])),
        "near_ped_count":     len(raw_state.get("near_peds",    [])),
        "near_object_count":  len(raw_state.get("near_objects", [])),
    }
    scene = torch.tensor(
        [_to_float(scene_counts[f] if f in scene_counts else raw_state.get(f, 0.0))
         for f in scene_fields],
        dtype=torch.float32,
    ).unsqueeze(0)

    route = torch.tensor(
        [_to_float(raw_state.get(f, 0.0)) for f in route_fields],
        dtype=torch.float32,
    ).unsqueeze(0)

    entity_rows, mask_values = [], []
    for entity in raw_state.get("near_entities", [])[:MAX_ENTITIES]:
        entity_rows.append([_to_float(entity.get(f, 0.0)) for f in entity_fields])
        mask_values.append(1.0)
    while len(entity_rows) < MAX_ENTITIES:
        entity_rows.append([0.0] * ENTITY_DIM)
        mask_values.append(0.0)

    entities = torch.tensor(entity_rows, dtype=torch.float32).unsqueeze(0)
    mask     = torch.tensor(mask_values, dtype=torch.float32).unsqueeze(0)

    return {"ego": ego, "scene": scene, "route": route, "entities": entities, "mask": mask}


# ---------------------------------------------------------------------------
# create_encoder_weights
# ---------------------------------------------------------------------------

def create_encoder_weights(
    embed_dim  = EMBED_DIM,
    hidden_dim = HIDDEN_DIM,
    fused_dim  = FUSED_DIM,
    num_heads  = NUM_HEADS,
):
    """
    Create fresh encoder weight modules.

    Returns a plain dict; pass it back in to encode_state() on every call.
    """
    ego_mlp = nn.Sequential(
        nn.Linear(EGO_DIM,   hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim),
    )
    scene_mlp = nn.Sequential(
        nn.Linear(SCENE_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim),
    )
    route_mlp = nn.Sequential(
        nn.Linear(ROUTE_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim),
    )
    entity_mlp = nn.Sequential(
        nn.Linear(ENTITY_DIM, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim),
    )
    fusion_mlp = nn.Sequential(
        nn.Linear(embed_dim * 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, fused_dim),
    )
    # Attention block 1 (query from concatenated ego/scene/route, dim = embed_dim * 3 = 192)
    qw1, kw1, vw1, ow1 = create_multi_head_attention_weights(embed_dim * 3, embed_dim)
    ln_attn1 = nn.LayerNorm(embed_dim)

    # Attention block 2 (query from block 1 output, dim = embed_dim = 64)
    qw2, kw2, vw2, ow2 = create_multi_head_attention_weights(embed_dim, embed_dim)
    ln_attn2 = nn.LayerNorm(embed_dim)

    return {
        "ego_mlp":    ego_mlp,
        "scene_mlp":  scene_mlp,
        "route_mlp":  route_mlp,
        "entity_mlp": entity_mlp,
        "fusion_mlp": fusion_mlp,
        "qw1": qw1, "kw1": kw1, "vw1": vw1, "ow1": ow1,
        "ln_attn1": ln_attn1,
        "qw2": qw2, "kw2": kw2, "vw2": vw2, "ow2": ow2,
        "ln_attn2": ln_attn2,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
    }


# ---------------------------------------------------------------------------
# encode_state
# ---------------------------------------------------------------------------

def encode_state(raw_state, weights):
    """
    Encode a raw GTA state dict into z_t [1, fused_dim].

    Input:
    - raw_state: dict from GTA stream
    - weights:   dict from create_encoder_weights()

    Output:
    - z_t: fused state embedding, shape [1, fused_dim]
    """
    t = build_state_tensors(raw_state)

    ego_emb     = weights["ego_mlp"](t["ego"])
    scene_emb   = weights["scene_mlp"](t["scene"])
    route_emb   = weights["route_mlp"](t["route"])
    entity_embs = weights["entity_mlp"](t["entities"])

    query_input = torch.cat([ego_emb, scene_emb, route_emb], dim=-1)

    # Block 1: cross-attention from ego/scene/route query to entity K/V
    attn1 = multi_head_attention(
        query_input = query_input,
        entity_embs = entity_embs,
        mask        = t["mask"],
        qw          = weights["qw1"],
        kw          = weights["kw1"],
        vw          = weights["vw1"],
        ow          = weights["ow1"],
        num_heads   = weights["num_heads"],
    )
    ctx1 = weights["ln_attn1"](attn1["entity_context"])  # [1, 64]

    # Block 2: refine using block 1 output as query, same entity K/V
    attn2 = multi_head_attention(
        query_input = ctx1,
        entity_embs = entity_embs,
        mask        = t["mask"],
        qw          = weights["qw2"],
        kw          = weights["kw2"],
        vw          = weights["vw2"],
        ow          = weights["ow2"],
        num_heads   = weights["num_heads"],
    )
    entity_context = weights["ln_attn2"](attn2["entity_context"] + ctx1)  # residual + LN

    fusion_input = torch.cat([ego_emb, scene_emb, route_emb, entity_context], dim=-1)
    z_t = weights["fusion_mlp"](fusion_input)
    return z_t


# ---------------------------------------------------------------------------
# get_candidates
# ---------------------------------------------------------------------------

def get_candidates(
    z_t,
    prev_token_id,
    vocab_size,
    token_embed   = None,
    intuition_mlp = None,
    planner_mlp   = None,
    token_embed_dim = 32,
    hidden_dim    = HIDDEN_DIM,
    top_k         = 3,
):
    """
    Run intuition head + action planner to get top-k candidates from z_t.

    The intuition head predicts where the world will be after the previous
    action continues.  The action planner uses that prediction alongside z_t
    to rank the token vocabulary and return the top-k most promising tokens.

    Input:
    - z_t:            current fused embedding, shape [1, fused_dim]
    - prev_token_id:  last action taken, int or shape [1] long tensor
    - vocab_size:     token vocabulary size
    - token_embed:    intuition head token embedding table (None = create fresh)
    - intuition_mlp:  intuition head MLP (None = create fresh)
    - planner_mlp:    action planner MLP (None = create fresh)
    - token_embed_dim: embedding dim for tokens inside the intuition head
    - hidden_dim, top_k: network dimensions

    Output dict:
    - z_next_pred:   predicted next embedding [1, fused_dim]
    - top_k_ids:     top-k token ids [1, top_k]
    - top_k_probs:   top-k probabilities [1, top_k]
    - token_embed:   updated token embedding table (pass back in next call)
    - intuition_mlp: updated intuition MLP (pass back in next call)
    - planner_mlp:   updated planner MLP (pass back in next call)
    """
    if isinstance(prev_token_id, int):
        prev_token_id = torch.tensor([prev_token_id], dtype=torch.long)

    z_next_pred, _, token_embed, intuition_mlp = intuition_head(
        z_t,
        prev_token_id,
        vocab_size    = vocab_size,
        token_embed   = token_embed,
        intuition_mlp = intuition_mlp,
        embed_dim     = token_embed_dim,
        hidden_dim    = hidden_dim,
    )

    planner_out = action_planner(
        z_t         = z_t,
        z_next_pred = z_next_pred,
        vocab_size  = vocab_size,
        planner_mlp = planner_mlp,
        hidden_dim  = hidden_dim,
        top_k       = top_k,
    )

    return {
        "z_next_pred":   z_next_pred,
        "top_k_ids":     planner_out["top_k_ids"],
        "top_k_probs":   planner_out["top_k_probs"],
        "token_embed":   token_embed,
        "intuition_mlp": intuition_mlp,
        "planner_mlp":   planner_out["planner_mlp"],
    }


# ---------------------------------------------------------------------------
# run_token  — full pipeline entry point
# ---------------------------------------------------------------------------

def run_token(
    token_id,
    raw_state,
    prev_token_id,
    vocab_size,
    encoder_weights,
    token_table,
    unmerge_fn,
    duration_fn,
    send_controls_fn,
    read_state_fn,
    start_frame,
    deadline_frame,
    rf_current,
    max_search_budget = 20,
    token_embed       = None,
    intuition_mlp     = None,
    reward_mlp        = None,
    rf_predictor      = None,
    planner_mlp       = None,
    meta_mlp          = None,
    token_embed_dim   = 32,
    hidden_dim        = HIDDEN_DIM,
    top_k             = 3,
    **reward_kwargs,
):
    """
    Full pipeline for one token: encode → get candidates → drive + search + train.

    Steps:
    1. Encodes the current GTA state → z_t.
    2. Runs intuition head + action planner → top-k root candidates for the search.
    3. Drives token_id to completion while the metacontroller searches for the next one.
    4. Trains metacontroller and reward head from the realized outcomes.

    Input:
    - token_id:         int, the token currently being played in GTA
    - raw_state:        raw GTA state dict at the start of this token
    - prev_token_id:    int, the token played before this one
                        (seeds the intuition head for candidate generation)
    - vocab_size:       token vocabulary size
    - encoder_weights:  dict from create_encoder_weights()
    - token_table:      token lookup table for the executor
    - unmerge_fn:       callable(int) -> list[int]  (unpack a merged token)
    - duration_fn:      callable(int) -> int        (token_id -> duration in frames)
    - send_controls_fn: callable, sends one control frame to GTA
    - read_state_fn:    callable, reads the latest GTA state dict
    - start_frame:      frame number when token_id starts executing
    - deadline_frame:   frame by which the next token must be chosen
    - rf_current:       reward features at the current real GTA state [1, RF_DIM]
    - max_search_budget: max nodes the search tree may expand per token
    - token_embed, intuition_mlp, reward_mlp, rf_predictor,
      planner_mlp, meta_mlp: shared weights (None = create fresh)
    - token_embed_dim:  token embedding dim inside the intuition head
    - hidden_dim, top_k: network dimensions
    - reward_kwargs:    forwarded to reward computation

    Output dict (same as drive_token):
    - next_token_id:  int, the token to play next
    - rollout:        per-frame states and rewards from this token's execution
    - interrupted:    bool, True if the metacontroller cut the token short
    - meta_mlp, reward_mlp, rf_predictor, planner_mlp: updated weights
    - meta_loss, reward_loss, rf_loss, token_return: training scalars
    """
    # -----------------------------------------------------------------------
    # 1. encode the current state
    # -----------------------------------------------------------------------

    z_t = encode_state(raw_state, encoder_weights)

    # -----------------------------------------------------------------------
    # 2. intuition head + action planner → root candidates for the search tree
    # -----------------------------------------------------------------------

    candidates = get_candidates(
        z_t             = z_t,
        prev_token_id   = prev_token_id,
        vocab_size      = vocab_size,
        token_embed     = token_embed,
        intuition_mlp   = intuition_mlp,
        planner_mlp     = planner_mlp,
        token_embed_dim = token_embed_dim,
        hidden_dim      = hidden_dim,
        top_k           = top_k,
    )

    # z_running: predicted embedding when we committed this token
    z_running = candidates["z_next_pred"]

    # -----------------------------------------------------------------------
    # 3. drive the token + metacontroller search + training
    # -----------------------------------------------------------------------

    encode_fn = lambda s: encode_state(s, encoder_weights)

    return drive_token(
        token_id                = token_id,
        token_table             = token_table,
        planner_candidate_ids   = candidates["top_k_ids"][0],
        planner_candidate_priors= candidates["top_k_probs"][0],
        fallback_token_id       = candidates["top_k_ids"][0, 0].item(),
        z_t                     = z_t,
        z_running               = z_running,
        rf_current              = rf_current,
        vocab_size              = vocab_size,
        unmerge_fn              = unmerge_fn,
        duration_fn             = duration_fn,
        encode_fn               = encode_fn,
        send_controls_fn        = send_controls_fn,
        read_state_fn           = read_state_fn,
        start_frame             = start_frame,
        deadline_frame          = deadline_frame,
        max_search_budget       = max_search_budget,
        token_embed             = candidates["token_embed"],
        intuition_mlp           = candidates["intuition_mlp"],
        reward_mlp              = reward_mlp,
        rf_predictor            = rf_predictor,
        planner_mlp             = candidates["planner_mlp"],
        meta_mlp                = meta_mlp,
        embed_dim               = token_embed_dim,
        hidden_dim              = hidden_dim,
        top_k                   = top_k,
        **reward_kwargs,
    )


# ---------------------------------------------------------------------------
# smoke test — run with: python main_model/main_model.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # synthetic zero state — no GTA connection needed
    raw_state = {"near_entities": [], "near_vehs": [], "near_peds": [], "near_objects": []}

    weights = create_encoder_weights()
    z_t = encode_state(raw_state, weights)

    vocab_size = 874
    cands = get_candidates(z_t, prev_token_id=0, vocab_size=vocab_size)

    print("z_t shape         :", tuple(z_t.shape))
    print("z_next_pred shape :", tuple(cands["z_next_pred"].shape))
    print("top_k_ids shape   :", tuple(cands["top_k_ids"].shape))
    print("top_k_probs shape :", tuple(cands["top_k_probs"].shape))
    print("\nTop-3 candidates:")
    for rank in range(cands["top_k_ids"].shape[1]):
        tid  = cands["top_k_ids"][0, rank].item()
        prob = cands["top_k_probs"][0, rank].item()
        print(f"  rank {rank + 1}: token {tid}  prob={prob:.4f}")
