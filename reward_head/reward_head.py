"""
Reward head — neural network node evaluator for the search tree.

Takes the parent embedding, the predicted child embedding, and the explicit
reward-relevant signals for both states, then predicts r_edge for that
transition.

The signals that directly drive reward are kept explicit so the network
does not have to discover them buried inside a 128-dim embedding:

    rf = [wp_dist, hp, v_engine_hp, v_body_hp, road_dist, dead]   (RF_DIM = 6)

For the parent node these come from the real GTA state.
For the child node they are predicted by a small MLP (predict_reward_features).

Both live here so the reward head and its feature predictor travel together.
"""

import torch
from torch import nn


RF_DIM = 6   # number of explicit reward-relevant features per state


# =========================================================================
# Feature extractor  (real GTA state -> reward feature tensor)
# =========================================================================

def extract_reward_features(gta_state, device="cpu"):
    """
    Pull the reward-relevant signals out of a raw GTA state dict.

    Output: shape [1, RF_DIM]
        [wp_dist, hp, v_engine_hp, v_body_hp, road_dist, dead]
    """
    return torch.tensor(
        [[
            float(gta_state.get("wp_dist",     0.0) or 0.0),
            float(gta_state.get("hp",          0.0) or 0.0),
            float(gta_state.get("v_engine_hp", 0.0) or 0.0),
            float(gta_state.get("v_body_hp",   0.0) or 0.0),
            float(gta_state.get("road_dist",   0.0) or 0.0),
            float(bool(gta_state.get("dead",   False))),
        ]],
        dtype=torch.float32,
        device=device,
    )


# =========================================================================
# Reward feature predictor  (predict child rf from parent embedding + delta)
# =========================================================================

def predict_reward_features(
    z_parent,
    delta_z,
    rf_parent,
    rf_predictor=None,
    fused_dim=128,
    hidden_dim=64,
):
    """
    Predict what the reward-relevant features will look like at the child state.

    The child state is only predicted (via the intuition head), so we cannot
    read its raw GTA values.  This MLP estimates them from the parent features
    and the embedding change.

    Input:
    - z_parent:     current node embedding,      shape [batch, fused_dim]
    - delta_z:      z_child - z_parent,          shape [batch, fused_dim]
    - rf_parent:    real reward features at parent, shape [batch, RF_DIM]
    - rf_predictor: shared MLP weights (None = create fresh)

    Output:
    - rf_child:     predicted reward features,   shape [batch, RF_DIM]
    - rf_predictor: updated MLP
    """

    # -------------------------------------------------------------------------
    # Step 1: assemble input
    # [z_parent | delta_z | rf_parent]  ->  [fused_dim*2 + RF_DIM]
    # -------------------------------------------------------------------------

    features  = torch.cat([z_parent, delta_z, rf_parent], dim=-1)
    input_dim = fused_dim * 2 + RF_DIM

    # -------------------------------------------------------------------------
    # Step 2: create predictor MLP if not passed in
    # -------------------------------------------------------------------------

    if rf_predictor is None:
        rf_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, RF_DIM),
        )

    # -------------------------------------------------------------------------
    # Step 3: predict child reward features
    # -------------------------------------------------------------------------

    rf_child = rf_predictor(features)   # [batch, RF_DIM]

    return rf_child, rf_predictor


# =========================================================================
# Reward head  (scores a parent -> child transition)
# =========================================================================

def reward_head(
    z_parent,
    z_child,
    rf_parent,
    rf_child,
    duration,
    time_left,
    reward_mlp=None,
    fused_dim=128,
    hidden_dim=128,
):
    """
    Predict r_edge for one (parent -> child) transition.

    Input:
    - z_parent:   embedding at parent node,          shape [batch, fused_dim]
    - z_child:    predicted embedding at child node, shape [batch, fused_dim]
    - rf_parent:  reward features at parent,         shape [batch, RF_DIM]
                  [wp_dist, hp, v_engine_hp, v_body_hp, road_dist, dead]
    - rf_child:   predicted reward features at child, shape [batch, RF_DIM]
    - duration:   token duration in frames,          shape [batch, 1]
    - time_left:  frames remaining until deadline,   shape [batch, 1]
    - reward_mlp: shared MLP weights (None = create fresh)

    Output:
    - r_edge:     predicted reward,   shape [batch, 1]
    - reward_mlp: updated MLP
    """

    # -------------------------------------------------------------------------
    # Step 1: embedding transition signal
    # -------------------------------------------------------------------------

    delta_z  = z_child  - z_parent    # [batch, fused_dim]

    # -------------------------------------------------------------------------
    # Step 2: explicit reward signal delta
    # delta_rf makes progress (wp_dist change), damage, and off-road explicit
    # -------------------------------------------------------------------------

    delta_rf = rf_child - rf_parent   # [batch, RF_DIM]

    # -------------------------------------------------------------------------
    # Step 3: assemble full feature vector
    #
    # [z_parent | delta_z | rf_parent | delta_rf | duration | time_left]
    #  128         128       6           6          1          1          = 270
    # -------------------------------------------------------------------------

    features  = torch.cat(
        [z_parent, delta_z, rf_parent, delta_rf, duration, time_left],
        dim=-1,
    )
    input_dim = fused_dim * 2 + RF_DIM * 2 + 2

    # -------------------------------------------------------------------------
    # Step 4: create reward MLP if not passed in
    # -------------------------------------------------------------------------

    if reward_mlp is None:
        reward_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    # -------------------------------------------------------------------------
    # Step 5: predict r_edge
    # -------------------------------------------------------------------------

    r_edge = reward_mlp(features)     # [batch, 1]

    return r_edge, reward_mlp
