"""
Reward — formula-based reward from raw GTA state.

Pure math. No neural network, no torch.
Input:  two GTA state dicts (frame before and frame after an action).
Output: reward float + breakdown dict.

Formula (from reward_method.txt):

    r_t = -w_d * d_t
          + w_p * (d_{t-1} - d_t)
          - w_step
          - w_C * C_t
          - w_O * O_t
          + R_goal * 1_goal
          + w_T * T_remaining * 1_goal
"""


def compute_reward(
    previous_state,
    current_state,
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
    Compute reward for one frame transition.

    Input:
    - previous_state: GTA state dict before the action
    - current_state:  GTA state dict after the action

    Output:
    - reward: float
    - components: dict with breakdown of every term
    """

    # -------------------------------------------------------------------------
    # Distance and progress toward goal
    # -------------------------------------------------------------------------

    prev_dist = float(previous_state.get("wp_dist", 0.0) or 0.0)
    curr_dist = float(current_state.get("wp_dist", 0.0) or 0.0)

    has_goal = prev_dist > 0.0 or curr_dist > 0.0

    if has_goal:
        distance_term = -distance_weight * curr_dist
        progress = prev_dist - curr_dist
        progress_term = progress_weight * progress
    else:
        distance_term = 0.0
        progress = 0.0
        progress_term = 0.0

    # -------------------------------------------------------------------------
    # Collision — detected via damage between frames
    # -------------------------------------------------------------------------

    prev_hp        = float(previous_state.get("hp", 0.0) or 0.0)
    curr_hp        = float(current_state.get("hp", 0.0) or 0.0)
    prev_engine_hp = float(previous_state.get("v_engine_hp", 0.0) or 0.0)
    curr_engine_hp = float(current_state.get("v_engine_hp", 0.0) or 0.0)
    prev_body_hp   = float(previous_state.get("v_body_hp", 0.0) or 0.0)
    curr_body_hp   = float(current_state.get("v_body_hp", 0.0) or 0.0)

    damage = (
        max(0.0, prev_hp        - curr_hp)
        + max(0.0, prev_engine_hp - curr_engine_hp)
        + max(0.0, prev_body_hp   - curr_body_hp)
    )

    collision = 0.0
    if damage > damage_threshold:
        collision = min(1.0, damage / 100.0)
    if bool(current_state.get("dead", False)):
        collision = 1.0

    collision_term = -collision_weight * collision

    # -------------------------------------------------------------------------
    # Off-road — scaled by how far from nearest road centre
    # -------------------------------------------------------------------------

    road_dist = float(current_state.get("road_dist", 0.0) or 0.0)
    offroad = 0.0
    if road_dist > offroad_start_m:
        offroad = (road_dist - offroad_start_m) / max(0.001, offroad_scale_m)
        offroad = min(1.0, offroad)

    offroad_term = -offroad_weight * offroad

    # -------------------------------------------------------------------------
    # Goal reached + early-finish bonus
    # -------------------------------------------------------------------------

    goal_reached = has_goal and curr_dist <= goal_radius_m

    time_remaining = 0.0
    if goal_reached and time_limit_s > 0.0 and episode_start_ts is not None:
        curr_ts  = float(current_state.get("ts", 0.0) or 0.0)
        elapsed  = max(0.0, (curr_ts - float(episode_start_ts)) / 1000.0)
        time_remaining = max(0.0, float(time_limit_s) - elapsed)

    goal_term = goal_reward * float(goal_reached)
    time_term = time_weight * time_remaining * float(goal_reached)

    # -------------------------------------------------------------------------
    # Sum
    # -------------------------------------------------------------------------

    step_term = -step_penalty

    reward = (
        distance_term
        + progress_term
        + step_term
        + collision_term
        + offroad_term
        + goal_term
        + time_term
    )

    components = {
        "reward":            reward,
        "distance":          curr_dist,
        "previous_distance": prev_dist,
        "progress":          progress,
        "distance_term":     distance_term,
        "progress_term":     progress_term,
        "step_term":         step_term,
        "collision":         collision,
        "collision_term":    collision_term,
        "offroad":           offroad,
        "road_dist":         road_dist,
        "offroad_term":      offroad_term,
        "goal_reached":      goal_reached,
        "goal_term":         goal_term,
        "time_remaining":    time_remaining,
        "time_term":         time_term,
    }

    return reward, components
