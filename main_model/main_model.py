"""
Notebook-style first pass for the embedding + attention model.

This file is intentionally written in a flat, top-to-bottom style so the flow
is easy to follow. There are no helper functions and no custom classes here.

Plan from `emebddings_and _attention.txt`:

ego [46] ---------> ego MLP ----------> ego_emb [64]
scene [16] -------> scene MLP --------> scene_emb [64]
route [14] -------> route MLP --------> route_emb [64]

entities [32,24] -> shared entity MLP -> entity_embs [32,64]
mask [32]

ego_emb + scene_emb + route_emb
            -> query MLP -> q [64]

entity_embs -> key MLP   -> K [32,64]
entity_embs -> value MLP -> V [32,64]

attention(q, K, V, mask)
            -> entity_context [64]

[ego_emb | scene_emb | route_emb | entity_context] -> [256]
fusion MLP -> z_t [128]

z_t -> planner MLP -> logits [vocab_size] -> softmax -> top-k token candidates
"""

import json
import sys
from pathlib import Path

import torch
from torch import nn

ACTION_PLANNER_DIR = Path(__file__).resolve().parent.parent / "action_planner"
if str(ACTION_PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(ACTION_PLANNER_DIR))

from action_planner import action_planner
from multi_head_attention import create_multi_head_attention_weights, multi_head_attention


# -----------------------------------------------------------------------------
# Step 1: choose the exact fields we want in each input block
# -----------------------------------------------------------------------------

# Ego block:
# Current player + vehicle state.
ego_fields = [
    "px",
    "py",
    "pz",
    "vx",
    "vy",
    "vz",
    "rx",
    "ry",
    "rz",
    "hp",
    "hp_max",
    "armor",
    "dead",
    "v_speed",
    "v_throttle",
    "v_brake",
    "v_steer",
    "u_throttle",
    "u_brake",
    "u_steer",
    "u_handbrake",
    "agent_throttle",
    "agent_brake",
    "agent_steer",
    "agent_handbrake",
    "v_rpm",
    "v_gear",
    "v_max_gear",
    "v_steer_angle",
    "v_clutch",
    "v_wheel_speed",
    "v_accel",
    "v_heading",
    "v_engine_hp",
    "v_body_hp",
    "v_engine_on",
    "v_turbo",
    "v_class",
    "v_ax",
    "v_ay",
    "v_az",
    "v_right_x",
    "v_right_y",
    "v_dim_w",
    "v_dim_l",
    "v_dim_h",
]

# Scene block:
# Global environment and nearby counts.
scene_fields = [
    "clock_h",
    "clock_m",
    "clock_s",
    "weather",
    "rain",
    "wind_speed",
    "wind_x",
    "wind_y",
    "wanted",
    "near_total",
    "near_entities_kept",
    "near_vehicle_count",
    "near_ped_count",
    "near_object_count",
    "agent_active",
    "agent_age_ms",
]

# Route block:
# Road and waypoint information.
route_fields = [
    "road_heading",
    "road_dist",
    "lane_offset",
    "lane_heading_delta",
    "road_lanes",
    "road_node2_heading",
    "road_node2_dist",
    "road_curve",
    "wp_x",
    "wp_y",
    "wp_dist",
    "v_heading",
    "v_fwd_x",
    "v_fwd_y",
]

# Entity block:
# Shared fields that work for vehicles, pedestrians, and objects.
entity_fields = [
    "type_id",
    "bucket_id",
    "dist",
    "dist3d",
    "rel_fwd",
    "rel_lat",
    "rel_z",
    "speed",
    "rel_v_fwd",
    "rel_v_lat",
    "heading",
    "hdiff",
    "dim_w",
    "dim_l",
    "dim_h",
    "is_static",
    "has_collision",
    "is_visible",
    "is_on_roadside",
    "ttc",
    "vx",
    "vy",
    "vz",
    "attached",
]

# Output:
# len(ego_fields)    = 46
# len(scene_fields)  = 16
# len(route_fields)  = 14
# len(entity_fields) = 24


# -----------------------------------------------------------------------------
# Step 2: load one saved GTA frame so we can build the tensors plainly
# -----------------------------------------------------------------------------

sample_path = Path(__file__).resolve().parent.parent / "gta_stream" / "stats" / "frame_000220.json"

with open(sample_path, "r", encoding="utf-8") as handle:
    raw_state = json.load(handle)

# Output from the current sample frame:
# len(raw_state)                 = 59
# len(raw_state["near_entities"]) = 32
# len(raw_state["near_vehs"])     = 41
# len(raw_state["near_peds"])     = 84
# len(raw_state["near_objects"])  = 420


# -----------------------------------------------------------------------------
# Step 3: build the ego tensor by reading the fields in order
# -----------------------------------------------------------------------------

ego_values = []
for field_name in ego_fields:
    value = raw_state.get(field_name, 0.0)
    if value is True:
        value = 1.0
    elif value is False or value is None:
        value = 0.0
    ego_values.append(float(value))

ego = torch.tensor(ego_values, dtype=torch.float32)

# Output:
# len(ego_values) = 46
# ego.shape       = (46,)
# Example values from this frame:
# px = -154.2162, py = -1486.3750, pz = 32.7531, v_speed = 2.6357, u_steer = 0.0


# -----------------------------------------------------------------------------
# Step 4: build the scene tensor
# -----------------------------------------------------------------------------

# Some scene values are easier to compute from list lengths.
scene_counts = {
    "near_vehicle_count": len(raw_state.get("near_vehs", [])),
    "near_ped_count": len(raw_state.get("near_peds", [])),
    "near_object_count": len(raw_state.get("near_objects", [])),
}

scene_values = []
for field_name in scene_fields:
    if field_name in scene_counts:
        value = scene_counts[field_name]
    else:
        value = raw_state.get(field_name, 0.0)

    if value is True:
        value = 1.0
    elif value is False or value is None:
        value = 0.0
    scene_values.append(float(value))

scene = torch.tensor(scene_values, dtype=torch.float32)

# Output:
# near_vehicle_count = 41
# near_ped_count     = 84
# near_object_count  = 420
# len(scene_values)  = 16
# scene.shape        = (16,)


# -----------------------------------------------------------------------------
# Step 5: build the route tensor
# -----------------------------------------------------------------------------

route_values = []
for field_name in route_fields:
    value = raw_state.get(field_name, 0.0)
    if value is True:
        value = 1.0
    elif value is False or value is None:
        value = 0.0
    route_values.append(float(value))

route = torch.tensor(route_values, dtype=torch.float32)

# Output:
# len(route_values) = 14
# route.shape       = (14,)


# -----------------------------------------------------------------------------
# Step 6: build the entity table and the entity mask
# -----------------------------------------------------------------------------

# The plan expects entities[32,24] and mask[32].
max_entities = 32

entity_rows = []
mask_values = []

for entity in raw_state.get("near_entities", [])[:max_entities]:
    row = []
    for field_name in entity_fields:
        value = entity.get(field_name, 0.0)
        if value is True:
            value = 1.0
        elif value is False or value is None:
            value = 0.0
        row.append(float(value))
    entity_rows.append(row)
    mask_values.append(1.0)

# Pad up to 32 rows so the tensor shape is always fixed.
while len(entity_rows) < max_entities:
    entity_rows.append([0.0] * len(entity_fields))
    mask_values.append(0.0)

entities = torch.tensor(entity_rows, dtype=torch.float32)
mask = torch.tensor(mask_values, dtype=torch.float32)

# Output:
# len(entity_rows)  = 32
# len(mask_values)  = 32
# entities.shape    = (32, 24)
# mask.shape        = (32,)
# mask.sum()        = 32.0 for this sample because the top-32 list is full


# -----------------------------------------------------------------------------
# Step 7: add a batch dimension
# -----------------------------------------------------------------------------

# PyTorch layers usually expect batched inputs.
ego = ego.unsqueeze(0)
scene = scene.unsqueeze(0)
route = route.unsqueeze(0)
entities = entities.unsqueeze(0)
mask = mask.unsqueeze(0)

# Output:
# ego.shape      = (1, 46)
# scene.shape    = (1, 16)
# route.shape    = (1, 14)
# entities.shape = (1, 32, 24)
# mask.shape     = (1, 32)


# -----------------------------------------------------------------------------
# Step 8: define the small MLP blocks
# -----------------------------------------------------------------------------

# We keep every embedding at width 64, just like the plan.
embed_dim = 64
hidden_dim = 128
fused_dim = 128

# This is now a hyperparameter you can change while experimenting.
# The only rule is: embed_dim must divide evenly by num_heads.
num_heads = 4

ego_mlp = nn.Sequential(
    nn.Linear(len(ego_fields), hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, embed_dim),
)

scene_mlp = nn.Sequential(
    nn.Linear(len(scene_fields), hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, embed_dim),
)

route_mlp = nn.Sequential(
    nn.Linear(len(route_fields), hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, embed_dim),
)

entity_mlp = nn.Sequential(
    nn.Linear(len(entity_fields), hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, embed_dim),
)

fusion_mlp = nn.Sequential(
    nn.Linear(embed_dim * 4, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, fused_dim),
)

# The reusable helper creates the trainable attention weights for us.
#
# query width = 64 + 64 + 64 = 192
# embed width = 64
qw, kw, vw, ow = create_multi_head_attention_weights(embed_dim * 3, embed_dim)

# Output:
# ego_mlp    maps 46  -> 64
# scene_mlp  maps 16  -> 64
# route_mlp  maps 14  -> 64
# entity_mlp maps 24  -> 64
# num_heads  = 4
# head_dim   = 64 / 4 = 16
# qw         maps 192 -> 64
# kw         maps 64  -> 64
# vw         maps 64  -> 64
# ow         maps 64  -> 64
# fusion_mlp maps 256 -> 128


# -----------------------------------------------------------------------------
# Step 9: run the three scalar blocks through their encoders
# -----------------------------------------------------------------------------

ego_emb = ego_mlp(ego)
scene_emb = scene_mlp(scene)
route_emb = route_mlp(route)

# Output:
# ego_emb.shape   = (1, 64)
# scene_emb.shape = (1, 64)
# route_emb.shape = (1, 64)


# -----------------------------------------------------------------------------
# Step 10: run each nearby entity through the shared entity encoder
# -----------------------------------------------------------------------------

entity_embs = entity_mlp(entities)

# Output:
# entity_embs.shape = (1, 32, 64)


# -----------------------------------------------------------------------------
# Step 11: create the attention query from ego + scene + route
# -----------------------------------------------------------------------------

query_input = torch.cat([ego_emb, scene_emb, route_emb], dim=-1)

# Output:
# query_input.shape = (1, 192)


# -----------------------------------------------------------------------------
# Step 12: run multi-head attention
# -----------------------------------------------------------------------------

# The helper function performs:
# 1. Q, K, V projections
# 2. splitting into heads
# 3. scaled dot-product attention
# 4. masking
# 5. weighted sum over V
# 6. merging heads back together
attention_outputs = multi_head_attention(
    query_input=query_input,
    entity_embs=entity_embs,
    mask=mask,
    qw=qw,
    kw=kw,
    vw=vw,
    ow=ow,
    num_heads=num_heads,
)

q = attention_outputs["q"]
k = attention_outputs["k"]
v = attention_outputs["v"]
q_heads = attention_outputs["q_heads"]
k_heads = attention_outputs["k_heads"]
v_heads = attention_outputs["v_heads"]
scores = attention_outputs["scores"]
attention = attention_outputs["attention"]
merged_context = attention_outputs["merged_context"]
entity_context = attention_outputs["entity_context"]
head_dim = attention_outputs["head_dim"]

# Output:
# q.shape              = (1, 64)
# k.shape              = (1, 32, 64)
# v.shape              = (1, 32, 64)
# q_heads.shape        = (1, 4, 1, 16)
# k_heads.shape        = (1, 4, 32, 16)
# v_heads.shape        = (1, 4, 32, 16)
# scores.shape         = (1, 4, 1, 32)
# attention.shape      = (1, 4, 1, 32)
# merged_context.shape = (1, 64)
# entity_context.shape = (1, 64)
# head_dim             = 16


# -----------------------------------------------------------------------------
# Step 13: fuse everything together into the final state embedding z_t
# -----------------------------------------------------------------------------

fusion_input = torch.cat([ego_emb, scene_emb, route_emb, entity_context], dim=-1)
z_t = fusion_mlp(fusion_input)

# Output:
# fusion_input.shape = (1, 256)
# z_t.shape          = (1, 128)


# -----------------------------------------------------------------------------
# Step 14: send the fused embedding into the token-distribution action planner
# -----------------------------------------------------------------------------

# vocab_size is passed in from outside — the model does not load the tokenizer.
# For this notebook demo, hardcode the value from the latest build.
vocab_size = 874

# prev_token_id defaults to IDLE_TOKEN_ID (0) on the first frame.
# In a live loop the caller passes the previously chosen token here.
planner_output = action_planner(
    z_t,
    vocab_size=vocab_size,
    prev_token_id=None,
    hidden_dim=hidden_dim,
    top_k=3,
)

z_next_pred = planner_output["z_next_pred"]
delta_z = planner_output["delta_z"]
logits = planner_output["logits"]
token_probs = planner_output["token_probs"]
top_k_ids = planner_output["top_k_ids"]
top_k_probs = planner_output["top_k_probs"]
token_embed = planner_output["token_embed"]
intuition_mlp = planner_output["intuition_mlp"]
planner_mlp = planner_output["planner_mlp"]

# Output:
# z_next_pred.shape = (1, 128)
# delta_z.shape     = (1, 128)
# logits.shape      = (1, 874)
# token_probs.shape = (1, 874)
# top_k_ids.shape   = (1, 3)
# top_k_probs.shape = (1, 3)


# -----------------------------------------------------------------------------
# Step 15: print shapes so the flow is easy to inspect
# -----------------------------------------------------------------------------

print("Input sizes")
print("  ego      ->", tuple(ego.shape))
print("  scene    ->", tuple(scene.shape))
print("  route    ->", tuple(route.shape))
print("  entities ->", tuple(entities.shape))
print("  mask     ->", tuple(mask.shape))

print("\nIntermediate sizes")
print("  ego_emb        ->", tuple(ego_emb.shape))
print("  scene_emb      ->", tuple(scene_emb.shape))
print("  route_emb      ->", tuple(route_emb.shape))
print("  entity_embs    ->", tuple(entity_embs.shape))
print("  q              ->", tuple(q.shape))
print("  k              ->", tuple(k.shape))
print("  v              ->", tuple(v.shape))
print("  q_heads        ->", tuple(q_heads.shape))
print("  k_heads        ->", tuple(k_heads.shape))
print("  v_heads        ->", tuple(v_heads.shape))
print("  scores         ->", tuple(scores.shape))
print("  attention      ->", tuple(attention.shape))
print("  merged_context ->", tuple(merged_context.shape))
print("  entity_context ->", tuple(entity_context.shape))

print("\nFinal size")
print("  z_t            ->", tuple(z_t.shape))

print("\nIntuition head sizes")
print("  z_next_pred    ->", tuple(z_next_pred.shape))
print("  delta_z        ->", tuple(delta_z.shape))

print("\nAction planner sizes")
print("  vocab_size     ->", vocab_size)
print("  logits         ->", tuple(logits.shape))
print("  token_probs    ->", tuple(token_probs.shape))
print("  top_k_ids      ->", tuple(top_k_ids.shape))
print("  top_k_probs    ->", tuple(top_k_probs.shape))

print("\nTop-3 token candidates")
for rank in range(top_k_ids.shape[1]):
    tid = top_k_ids[0, rank].item()
    prob = top_k_probs[0, rank].item()
    print(f"  rank {rank + 1}: token {tid}  prob={prob:.4f}")
