"""
Time context — computes external timing signals for the metacontroller.

The metacontroller needs urgency, elapsed ratio, and search budget but
should not compute them itself.  This helper turns raw frame counts and
deadlines into the tensors the metacontroller expects.
"""

import torch


def time_context(
    current_frame,
    deadline_frame,
    token_start_frame,
    token_duration_frames,
    nodes_expanded,
    max_search_budget,
    device="cpu",
):
    """
    Compute timing features from raw external signals.

    Input:
    - current_frame: int, the global frame counter right now
    - deadline_frame: int, frame by which an action must be committed
    - token_start_frame: int, frame when the running token started
    - token_duration_frames: int, how many frames the running token lasts
    - nodes_expanded: int, how many tree nodes have been expanded so far
    - max_search_budget: int, maximum nodes the tree is allowed to expand

    Output dict:
    - frames_left: int, frames until the deadline
    - elapsed_ratio: shape [1, 1], fraction of running token elapsed
    - urgency: shape [1, 1], 0 = no rush, 1 = critical
    - budget_remaining: int, how many more nodes can be expanded
    - budget_ratio: shape [1, 1], fraction of search budget spent
    """

    # -------------------------------------------------------------------------
    # Step 1: frames remaining until the deadline
    # -------------------------------------------------------------------------

    frames_left = max(0, deadline_frame - current_frame)

    # -------------------------------------------------------------------------
    # Step 2: how far into the running token we are
    # -------------------------------------------------------------------------

    frames_into_token = current_frame - token_start_frame
    if token_duration_frames > 0:
        elapsed = min(1.0, max(0.0, frames_into_token / token_duration_frames))
    else:
        elapsed = 1.0

    elapsed_ratio = torch.tensor([[elapsed]], dtype=torch.float32, device=device)

    # frames the metacontroller has left before the current token ends
    # and it MUST have a decision ready
    token_frames_left = max(0, token_duration_frames - frames_into_token)

    token_frames_left_t = torch.tensor(
        [[float(token_frames_left)]], dtype=torch.float32, device=device
    )

    # -------------------------------------------------------------------------
    # Step 3: urgency from deadline proximity
    # -------------------------------------------------------------------------

    total_window = max(1, deadline_frame - token_start_frame)
    urgency_value = 1.0 - (frames_left / total_window)
    urgency_value = min(1.0, max(0.0, urgency_value))

    urgency = torch.tensor([[urgency_value]], dtype=torch.float32, device=device)

    # -------------------------------------------------------------------------
    # Step 4: search budget
    # -------------------------------------------------------------------------

    budget_remaining = max(0, max_search_budget - nodes_expanded)

    if max_search_budget > 0:
        budget_spent = min(1.0, nodes_expanded / max_search_budget)
    else:
        budget_spent = 1.0

    budget_ratio = torch.tensor([[budget_spent]], dtype=torch.float32, device=device)

    return {
        "frames_left":      frames_left,
        "elapsed_ratio":    elapsed_ratio,
        "urgency":          urgency,
        "budget_remaining": budget_remaining,
        "budget_ratio":     budget_ratio,
        "token_frames_left": token_frames_left_t,  # raw frames left in current token
    }
