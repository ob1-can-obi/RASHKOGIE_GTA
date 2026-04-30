"""
Reward head.

This is not a trainable network right now. The reward method is a direct formula
from reward_method.txt, calculated from the frame before an action and the frame
after that action.
"""

import torch


def reward_head(
    previous_state,
    current_state,
    device="cpu",
    distance_weight=0.1,
    progress_weight=1.0,
    step_penalty=0.01,
    collision_weight=5.0,
    offroad_weight=1.0,
    goal_reward=20.0,
    time_weight=0.1,
    goal_radius_m=5.0,
    offroad_start_m=8.0,
    offroad_scale_m=10.0,
    damage_threshold=1.0,
    time_limit_s=0.0,
    episode_start_ts=None,
):
    """
    Calculate the reward after one action.

    Input:
    - previous_state: GTA frame before the action
    - current_state: GTA frame after the action

    Output:
    - reward_tensor: shape [1, 1]
    - components: plain numbers explaining where the reward came from
    """

    # -------------------------------------------------------------------------
    # Step 1: read distance-to-goal from the waypoint distance
    # -------------------------------------------------------------------------

    previous_distance = float(previous_state.get("wp_dist", 0.0) or 0.0)
    current_distance = float(current_state.get("wp_dist", 0.0) or 0.0)

    has_goal = previous_distance > 0.0 or current_distance > 0.0

    if has_goal:
        distance_term = -distance_weight * current_distance
        progress = previous_distance - current_distance
        progress_term = progress_weight * progress
    else:
        distance_term = 0.0
        progress = 0.0
        progress_term = 0.0

    # -------------------------------------------------------------------------
    # Step 2: detect collision severity from damage between frames
    # -------------------------------------------------------------------------

    previous_hp = float(previous_state.get("hp", 0.0) or 0.0)
    current_hp = float(current_state.get("hp", 0.0) or 0.0)
    previous_engine_hp = float(previous_state.get("v_engine_hp", 0.0) or 0.0)
    current_engine_hp = float(current_state.get("v_engine_hp", 0.0) or 0.0)
    previous_body_hp = float(previous_state.get("v_body_hp", 0.0) or 0.0)
    current_body_hp = float(current_state.get("v_body_hp", 0.0) or 0.0)

    player_damage = max(0.0, previous_hp - current_hp)
    engine_damage = max(0.0, previous_engine_hp - current_engine_hp)
    body_damage = max(0.0, previous_body_hp - current_body_hp)

    damage_points = player_damage + engine_damage + body_damage
    collision = 0.0
    if damage_points > damage_threshold:
        collision = min(1.0, damage_points / 100.0)
    if bool(current_state.get("dead", False)):
        collision = 1.0

    collision_term = -collision_weight * collision

    # -------------------------------------------------------------------------
    # Step 3: estimate off-road severity from distance to nearest road center
    # -------------------------------------------------------------------------

    road_distance = float(current_state.get("road_dist", 0.0) or 0.0)
    offroad = 0.0
    if road_distance > offroad_start_m:
        offroad = (road_distance - offroad_start_m) / max(0.001, offroad_scale_m)
        offroad = max(0.0, min(1.0, offroad))

    offroad_term = -offroad_weight * offroad

    # -------------------------------------------------------------------------
    # Step 4: give goal and remaining-time bonus only when goal is reached
    # -------------------------------------------------------------------------

    goal_reached = 0.0
    if has_goal and current_distance <= goal_radius_m:
        goal_reached = 1.0

    time_remaining = 0.0
    if goal_reached > 0.0 and time_limit_s > 0.0 and episode_start_ts is not None:
        current_ts = float(current_state.get("ts", 0.0) or 0.0)
        elapsed_s = max(0.0, (current_ts - float(episode_start_ts)) / 1000.0)
        time_remaining = max(0.0, float(time_limit_s) - elapsed_s)

    goal_term = goal_reward * goal_reached
    time_term = time_weight * time_remaining * goal_reached

    # -------------------------------------------------------------------------
    # Step 5: add every term into one scalar reward
    # -------------------------------------------------------------------------

    step_term = -step_penalty

    reward_value = (
        distance_term
        + progress_term
        + step_term
        + collision_term
        + offroad_term
        + goal_term
        + time_term
    )

    reward_tensor = torch.tensor(
        [[reward_value]],
        dtype=torch.float32,
        device=device,
    )

    components = {
        "reward": reward_value,
        "distance": current_distance,
        "previous_distance": previous_distance,
        "progress": progress,
        "distance_term": distance_term,
        "progress_term": progress_term,
        "step_term": step_term,
        "collision": collision,
        "collision_term": collision_term,
        "offroad": offroad,
        "road_dist": road_distance,
        "offroad_term": offroad_term,
        "goal_reached": goal_reached,
        "goal_term": goal_term,
        "time_remaining": time_remaining,
        "time_term": time_term,
    }

    return reward_tensor, components

