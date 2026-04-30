# Reward Head

This folder calculates the reward after each action.

The reward is not a neural network right now. It is the direct formula from
`reward_method.txt`:

```text
r_t = -w_d * d_t
      + w_p * (d_{t-1} - d_t)
      - w_step
      - w_C * C_t
      - w_O * O_t
      + R_goal * 1_goal
      + w_T * T_remaining * 1_goal
```

## File

- `reward_head.py`
  - Has one function: `reward_head(...)`.
  - Takes the GTA frame before the action and the GTA frame after the action.
  - Returns `reward_tensor` with shape `[1, 1]`.
  - Also returns plain component values so the reward can be debugged.

## Signals Used

```text
d_t:
  current_state["wp_dist"]

d_{t-1}:
  previous_state["wp_dist"]

progress:
  previous wp_dist - current wp_dist

C_t:
  damage between frames from hp, v_engine_hp, and v_body_hp
  dead=True forces collision to 1

O_t:
  road_dist above the off-road threshold

1_goal:
  current wp_dist <= goal_radius_m

T_remaining:
  only used when the goal is reached and a time limit is provided
```

## Training Loop

`action_planner_training/online_train.py` calculates this reward every frame
transition that it trains on:

```text
train_frame + actual action -> next_frame -> reward_head(train_frame, next_frame)
```

Reward logs are written to:

```text
reward_head/training_data/session_YYYYMMDD_HHMMSS.jsonl
reward_head/logs/session_YYYYMMDD_HHMMSS.log
```

## Game Runner

`gta_stream/run_action_planner.py` also logs reward while the model is driving.
It scores the next GTA frame against the frame where the previous action was
chosen.

The model-driving runner also writes:

```text
gta_stream/logs/run_action_planner_YYYYMMDD_HHMMSS.log
reward_head/training_data/run_action_planner_YYYYMMDD_HHMMSS.jsonl
```

## Stats

The reward head is formula-based, so it has no loss and no checkpoint.

Stats are rebuilt after each training session:

```text
reward_head/training_data/stats/reward_head_stats.csv
reward_head/training_data/stats/summary.txt
reward_head/training_data/graphs/reward_curve.svg
reward_head/training_data/graphs/reward_components.svg
```

You can rebuild them manually with:

```bash
python reward_head/stats.py
```
