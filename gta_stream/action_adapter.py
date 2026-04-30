"""
Converts action planner outputs into GTA control timelines.

This stays in `gta_stream` because it is game/protocol code, not model code.
"""

import sys
from pathlib import Path

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "tokenizer"
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))

from tokenizer import get_token_chunk


def token_to_timeline(token_id, tokenizer_state, frame_dt_s=0.05):
    """
    Convert one token ID into a GTA command timeline using the tokenizer lookup.

    Each frame in the token's raw_control_chunk becomes one command in the timeline.
    The chunk stores right_steer/left_steer which get merged back into signed steer.
    """

    entry = get_token_chunk(tokenizer_state, token_id)
    chunk = entry["raw_control_chunk"]

    timeline = []
    for frame in chunk:
        right_steer = float(frame.get("right_steer", 0.0) or 0.0)
        left_steer = float(frame.get("left_steer", 0.0) or 0.0)
        steer = right_steer - left_steer

        timeline.append(
            {
                "duration_s": frame_dt_s,
                "steer": max(-1.0, min(1.0, steer)),
                "throttle": max(0.0, min(1.0, float(frame.get("forward_throttle", 0.0) or 0.0))),
                "brake": max(0.0, min(1.0, float(frame.get("brake", 0.0) or 0.0))),
                "handbrake": False,
                "enabled": True,
            }
        )

    if not timeline:
        timeline.append(
            {
                "duration_s": frame_dt_s,
                "steer": 0.0,
                "throttle": 0.0,
                "brake": 0.0,
                "handbrake": False,
                "enabled": True,
            }
        )

    return timeline


def planner_tensor_to_timeline(
    action_tensor,
    active_threshold=0.50,
    frame_dt_s=0.05,
):
    """
    Convert one planner tensor into plain GTA command dictionaries.

    Expected action_tensor layout:
    [0:5]   strength    = steer, throttle, brake, handbrake, enabled
    [5:10]  active_prob = steer, throttle, brake, handbrake, enabled
    [10:15] duration_s  = steer, throttle, brake, handbrake, enabled
    """

    if hasattr(action_tensor, "detach"):
        values = action_tensor.detach().cpu()
        if len(values.shape) == 2:
            values = values[0]
        values = values.tolist()
    else:
        values = action_tensor
        if values and isinstance(values[0], list):
            values = values[0]

    strength = values[0:5]
    active_prob = values[5:10]
    duration_s = values[10:15]

    active = []
    for value in active_prob:
        active.append(float(value) >= active_threshold)

    action_names = ["steer", "throttle", "brake", "handbrake", "enabled"]
    controls = []
    for index, name in enumerate(action_names):
        controls.append(
            {
                "name": name,
                "strength": float(strength[index]),
                "active": bool(active[index]),
                "duration_s": max(0.0, float(duration_s[index])),
            }
        )

    enabled_control = controls[4]
    active_controls = [control for control in controls if control["active"]]
    timeline = []

    if not active_controls:
        return [
            {
                "duration_s": frame_dt_s,
                "steer": 0.0,
                "throttle": 0.0,
                "brake": 0.0,
                "handbrake": False,
                "enabled": bool(enabled_control["active"]),
            }
        ]

    max_duration = 0.0
    for control in active_controls:
        if control["duration_s"] > max_duration:
            max_duration = control["duration_s"]

    step_count = max(1, int((max_duration / frame_dt_s) + 0.999999))

    for step_index in range(step_count):
        elapsed_s = step_index * frame_dt_s
        command = {
            "duration_s": frame_dt_s,
            "steer": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "handbrake": False,
            "enabled": bool(
                enabled_control["active"] and elapsed_s < enabled_control["duration_s"]
            ),
        }

        for control in controls:
            if not control["active"]:
                continue
            if elapsed_s >= control["duration_s"]:
                continue

            if control["name"] == "steer":
                command["steer"] = max(-1.0, min(1.0, control["strength"]))
            elif control["name"] == "throttle":
                command["throttle"] = max(0.0, min(1.0, control["strength"]))
            elif control["name"] == "brake":
                command["brake"] = max(0.0, min(1.0, control["strength"]))
            elif control["name"] == "handbrake":
                command["handbrake"] = True
            elif control["name"] == "enabled":
                command["enabled"] = True

        timeline.append(command)

    return timeline

