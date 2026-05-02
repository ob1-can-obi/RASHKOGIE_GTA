"""
Executor — plays a committed token in GTA one frame at a time.

Responsibilities:
1. Unmerge the committed (possibly BPE-merged) token into its base token sequence.
2. Build a flat per-frame control schedule from those base tokens.
3. Send each frame's controls to GTA via send_controls_fn.
4. Read back the resulting GTA state via read_state_fn.
5. Hand (prev_state, curr_state) to the reward module and record the result.

No learning happens here. Learning is trainer.py's job.
"""

from reward import compute_reward


# =========================================================================
# Token helpers
# =========================================================================

def build_control_schedule(token_id, token_table, unmerge_fn):
    """
    Unmerge a (possibly merged) token and flatten into a per-frame list.

    Each base token in the token_table has:
      "controls"  — the control values for that chunk (sent every frame)
      "duration"  — how many frames the chunk lasts

    Input:
    - token_id:   int, the committed token (may be a merged BPE token)
    - token_table: dict mapping token_id -> {"controls": ..., "duration": int}
    - unmerge_fn: callable(token_id: int) -> list[int] of base token ids

    Output:
    - schedule: list of control dicts, one entry per frame
    """
    base_ids = unmerge_fn(token_id)
    schedule = []
    for base_id in base_ids:
        entry    = token_table[base_id]
        controls = entry["controls"]
        duration = int(entry["duration"])
        for _ in range(duration):
            schedule.append(controls)
    return schedule


# =========================================================================
# Execution state
# =========================================================================

class ExecutionState:
    """Mutable per-token state passed between frames."""

    def __init__(self, token_id, schedule):
        self.token_id = token_id
        self.schedule  = schedule          # flat list of per-frame controls
        self.duration  = len(schedule)

        self.frame_idx  = 0
        self.states     = []               # GTA states  (length = duration + 1)
        self.rewards    = []               # per-frame reward floats
        self.components = []               # per-frame reward breakdowns
        self.done       = False


# =========================================================================
# Per-frame interface
# =========================================================================

def execution_init(token_id, token_table, unmerge_fn, read_state_fn):
    """
    Prepare to execute a token. Call once when switching to a new token.

    Unmerges the token, builds the control schedule, and reads the initial
    GTA state before the first frame plays.

    Input:
    - token_id:    int
    - token_table: token lookup (see build_control_schedule)
    - unmerge_fn:  callable(int) -> list[int]
    - read_state_fn: callable() -> GTA state dict

    Output:
    - state: ExecutionState
    """
    schedule = build_control_schedule(token_id, token_table, unmerge_fn)
    state    = ExecutionState(token_id, schedule)
    state.states.append(read_state_fn())
    return state


def execution_frame(state, send_controls_fn, read_state_fn, **reward_kwargs):
    """
    Play one frame of the current token.

    Sends controls to GTA, reads back the new state, computes the reward
    for this frame transition, and records everything.

    Input:
    - state:            ExecutionState
    - send_controls_fn: callable(controls) -> None
    - read_state_fn:    callable() -> GTA state dict
    - reward_kwargs:    forwarded to compute_reward (weights, thresholds, etc.)

    Output dict:
    - reward:    float
    - components: dict, reward breakdown
    - done:      bool, True if all frames of the token have played
    - frame_idx: int, which frame just played (0-indexed)
    """

    if state.done:
        return {
            "reward":     0.0,
            "components": {},
            "done":       True,
            "frame_idx":  state.frame_idx,
        }

    # send this frame's controls to GTA
    send_controls_fn(state.schedule[state.frame_idx])

    # read what happened
    new_state = read_state_fn()

    # compute reward from the two raw states
    reward_value, comps = compute_reward(
        state.states[-1],
        new_state,
        **reward_kwargs,
    )

    # record
    state.states.append(new_state)
    state.rewards.append(reward_value)
    state.components.append(comps)
    state.frame_idx += 1

    if state.frame_idx >= state.duration:
        state.done = True

    return {
        "reward":     reward_value,
        "components": comps,
        "done":       state.done,
        "frame_idx":  state.frame_idx - 1,
    }


def get_rollout(state):
    """
    Package the rollout after a token finishes.

    Output dict:
    - token_id:    int
    - duration:    int, frames actually played
    - states:      list of GTA state dicts
    - rewards:     list of per-frame reward floats
    - components:  list of per-frame reward breakdowns
    - state_before: first GTA state
    - state_after:  last GTA state
    """
    return {
        "token_id":    state.token_id,
        "duration":    state.frame_idx,
        "states":      state.states,
        "rewards":     state.rewards,
        "components":  state.components,
        "state_before": state.states[0],
        "state_after":  state.states[-1] if len(state.states) > 1 else state.states[0],
    }
