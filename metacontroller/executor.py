"""
Executor — plays a committed token in GTA one frame at a time.

The executor is the token runner / GTA bridge.  It looks up the raw control
chunk from the token table and exposes a per-frame interface so the frame
loop can interleave execution with search.

It does NOT do any learning.  Learning happens in trainer.py.
"""

import sys
from pathlib import Path

REWARD_HEAD_DIR = Path(__file__).resolve().parent.parent / "reward_head"
if str(REWARD_HEAD_DIR) not in sys.path:
    sys.path.insert(0, str(REWARD_HEAD_DIR))

from reward_head import reward_head


def lookup_chunk(token_id, token_table):
    """
    Look up the raw control chunk and duration for a token.

    Input:
    - token_id: int, the committed token
    - token_table: dict or lookup object mapping token_id -> chunk info
      each entry has "controls" (raw control values) and "duration" (frames)

    Output dict:
    - controls: the raw control values for this token
    - duration: int, how many frames this chunk lasts
    """
    entry = token_table[token_id]
    return {
        "controls": entry["controls"],
        "duration": entry["duration"],
    }


def env_reward_fn(previous_state, current_state, device="cpu", **reward_kwargs):
    """
    Compute a single frame's environment reward from actual GTA state.

    This is the hand-designed formula from reward_head.py applied to real
    frames.  It is NOT a learned predictor — it is ground truth.

    Input:
    - previous_state: GTA frame dict before the frame
    - current_state: GTA frame dict after the frame

    Output:
    - reward_value: float, the scalar reward for this frame transition
    - components: dict, breakdown of reward terms
    """
    reward_tensor, components = reward_head(
        previous_state,
        current_state,
        device=device,
        **reward_kwargs,
    )
    return components["reward"], components


# =========================================================================
# Execution state (carries rollout data between frames)
# =========================================================================

class ExecutionState:
    """Mutable state for a token being played, passed between frames."""

    def __init__(self, token_id, controls, duration):
        self.token_id = token_id
        self.controls = controls
        self.duration = duration

        self.frame_idx = 0
        self.states = []       # GTA states (length = duration + 1)
        self.rewards = []      # per-frame rewards (length = duration)
        self.components = []   # per-frame reward breakdowns
        self.done = False


# =========================================================================
# Per-frame interface
# =========================================================================

def execution_init(token_id, token_table, read_state_fn):
    """
    Start executing a token.  Call once when switching to a new token.

    Reads the initial GTA state and prepares the execution state.

    Input:
    - token_id: int, the token to play
    - token_table: token lookup (see lookup_chunk)
    - read_state_fn: callable() -> GTA state dict

    Output:
    - state: ExecutionState
    """
    chunk = lookup_chunk(token_id, token_table)

    state = ExecutionState(
        token_id=token_id,
        controls=chunk["controls"],
        duration=chunk["duration"],
    )

    # read the state before we start playing
    state.states.append(read_state_fn())

    return state


def execution_frame(state, send_controls_fn, read_state_fn, **reward_kwargs):
    """
    Play one frame of the current token.  Call once per frame.

    Sends controls to GTA, reads the new state, computes the environment
    reward for this frame transition.

    Input:
    - state: ExecutionState from execution_init (or previous execution_frame)
    - send_controls_fn: callable(controls) -> None
    - read_state_fn: callable() -> GTA state dict
    - reward_kwargs: forwarded to env_reward_fn

    Output dict:
    - reward: float, environment reward for this frame
    - components: dict, reward breakdown
    - done: bool, True if the token has finished all its frames
    - frame_idx: int, which frame just played (0-indexed)
    """

    if state.done:
        return {
            "reward": 0.0,
            "components": {},
            "done": True,
            "frame_idx": state.frame_idx,
        }

    # send controls to GTA
    send_controls_fn(state.controls)

    # read what happened
    new_state = read_state_fn()

    # compute environment reward for this frame
    reward_value, comps = env_reward_fn(
        state.states[-1],
        new_state,
        **reward_kwargs,
    )

    # record
    state.states.append(new_state)
    state.rewards.append(reward_value)
    state.components.append(comps)
    state.frame_idx += 1

    # check if token is finished
    if state.frame_idx >= state.duration:
        state.done = True

    return {
        "reward": reward_value,
        "components": comps,
        "done": state.done,
        "frame_idx": state.frame_idx - 1,
    }


def get_rollout(state):
    """
    Extract the rollout summary from a finished execution.

    Input:
    - state: ExecutionState (should be done)

    Output dict:
    - token_id: int
    - duration: int, frames played
    - states: list of GTA state dicts
    - rewards: list of per-frame reward floats
    - components: list of per-frame reward breakdowns
    - state_before: first GTA state
    - state_after: last GTA state
    """
    return {
        "token_id": state.token_id,
        "duration": state.frame_idx,
        "states": state.states,
        "rewards": state.rewards,
        "components": state.components,
        "state_before": state.states[0],
        "state_after": state.states[-1] if len(state.states) > 1 else state.states[0],
    }
