"""
Frame loop — the main driver that interleaves execution and search.

While the current token drives the car, the metacontroller thinks about
what to do next.  Both happen one step per frame, in lockstep.

Simple loop:
    1. Start current token
    2. While current token is running:
        - send next control frame to GTA
        - read latest game state
        - metacontroller searches one step for the best next token
    3. When current token ends:
        - if metacontroller found a next token, use it
        - otherwise, fall back to planner's top-1
    4. Repeat

If the metacontroller says INTERRUPT mid-token, we stop early and switch.
GTA never waits.
"""

import sys
from pathlib import Path

import torch

ACTION_PLANNER_DIR = Path(__file__).resolve().parent.parent / "action_planner"
if str(ACTION_PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(ACTION_PLANNER_DIR))

MAIN_MODEL_DIR = Path(__file__).resolve().parent.parent / "main_model"
if str(MAIN_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_MODEL_DIR))

from executor import execution_init, execution_frame, get_rollout
from search_tree import search_init, search_step
from time_context import time_context
from trainer import train_step, train_reward_head


def run_frame(
    exec_state,
    search_state,
    z_t,
    current_frame,
    deadline_frame,
    token_start_frame,
    token_duration_frames,
    max_search_budget,
    send_controls_fn,
    read_state_fn,
    **reward_kwargs,
):
    """
    Run one frame: play one control frame + do one search step.

    Input:
    - exec_state: ExecutionState from executor
    - search_state: SearchState from search_tree
    - z_t: latest real fused embedding from the main model, shape [1, fused_dim]
    - current_frame, deadline_frame, etc.: raw timing signals
    - max_search_budget: max nodes the search may expand
    - send_controls_fn, read_state_fn: GTA bridge functions
    - reward_kwargs: forwarded to env_reward_fn

    Output dict:
    - exec_result: result from execution_frame
    - search_result: result from search_step
    - token_done: bool, current token finished
    - search_done: bool, search concluded (COMMIT/INTERRUPT or budget out)
    - interrupt: bool, metacontroller says stop current token NOW
    """

    # -----------------------------------------------------------------
    # 1. play one frame of the current token
    # -----------------------------------------------------------------

    exec_result = execution_frame(
        exec_state, send_controls_fn, read_state_fn, **reward_kwargs,
    )

    # -----------------------------------------------------------------
    # 2. get fresh time context
    # -----------------------------------------------------------------

    time_ctx = time_context(
        current_frame=current_frame,
        deadline_frame=deadline_frame,
        token_start_frame=token_start_frame,
        token_duration_frames=token_duration_frames,
        nodes_expanded=search_state.nodes_expanded,
        max_search_budget=max_search_budget,
    )

    # -----------------------------------------------------------------
    # 3. search one step for the next token
    # -----------------------------------------------------------------

    search_result = search_step(search_state, z_t, time_ctx)

    # -----------------------------------------------------------------
    # 4. check for interrupt
    # -----------------------------------------------------------------

    from metacontroller import INTERRUPT
    interrupt = (search_result["decision"] == INTERRUPT)

    return {
        "exec_result": exec_result,
        "search_result": search_result,
        "token_done": exec_state.done,
        "search_done": search_state.done,
        "interrupt": interrupt,
    }


def drive_token(
    token_id,
    token_table,
    planner_candidate_ids,
    planner_candidate_priors,
    fallback_token_id,
    z_t,
    z_running,
    rf_current,
    vocab_size,
    unmerge_fn,
    duration_fn,
    encode_fn,
    send_controls_fn,
    read_state_fn,
    start_frame,
    deadline_frame,
    max_search_budget,
    token_embed=None,
    intuition_mlp=None,
    reward_mlp=None,
    rf_predictor=None,
    planner_mlp=None,
    meta_mlp=None,
    embed_dim=32,
    hidden_dim=128,
    top_k=3,
    **reward_kwargs,
):
    """
    Drive one token to completion while searching for the next one.

    This is the main per-token driver.  It plays the current token frame
    by frame, runs one search step per frame, and returns the rollout +
    the next token to play.

    Input:
    - token_id: int, the current token to play
    - token_table: token lookup for the executor
    - planner_candidate_ids: top-k token ids for the NEXT token search
    - planner_candidate_priors: top-k probs
    - fallback_token_id: planner's top-1, used if search doesn't finish
    - z_t: current fused embedding, shape [1, fused_dim]
    - z_running: state prediction when this token was committed, shape [1, fused_dim]
    - rf_current: reward features at the current real GTA state, shape [1, RF_DIM]
    - vocab_size: token vocabulary size
    - unmerge_fn: callable(int) -> list[int]
    - duration_fn: callable(int) -> int  (token_id -> duration in frames)
    - encode_fn: callable(state) -> z_t, encodes a GTA state into embedding
    - send_controls_fn, read_state_fn: GTA bridge
    - start_frame: frame number when this token starts
    - deadline_frame: frame by which next token must be ready
    - max_search_budget: max nodes to expand
    - token_embed, intuition_mlp, reward_mlp, rf_predictor,
      planner_mlp, meta_mlp: shared weights
    - embed_dim, hidden_dim: network dimensions
    - top_k: candidates per node in the search tree
    - reward_kwargs: forwarded to compute_reward (weights, thresholds, etc.)

    Output dict:
    - rollout: dict from get_rollout (per-frame rewards, states)
    - next_token_id: int, the token to play next
    - search_state: SearchState (for trainer to access meta_trajectory, root)
    - interrupted: bool, True if token was cut short by INTERRUPT
    """

    # -----------------------------------------------------------------
    # init executor (start playing current token)
    # -----------------------------------------------------------------

    exec_state = execution_init(token_id, token_table, unmerge_fn, read_state_fn)
    token_duration = exec_state.duration

    # -----------------------------------------------------------------
    # init search (start looking for next token)
    # -----------------------------------------------------------------

    running_token_id = torch.tensor([token_id], dtype=torch.long)

    search_state = search_init(
        z_t=z_t,
        rf_current=rf_current,
        planner_candidate_ids=planner_candidate_ids,
        planner_candidate_priors=planner_candidate_priors,
        running_token_id=running_token_id,
        z_running=z_running,
        vocab_size=vocab_size,
        unmerge_fn=unmerge_fn,
        duration_fn=duration_fn,
        token_embed=token_embed,
        intuition_mlp=intuition_mlp,
        reward_mlp=reward_mlp,
        rf_predictor=rf_predictor,
        planner_mlp=planner_mlp,
        meta_mlp=meta_mlp,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        top_k=top_k,
    )

    # -----------------------------------------------------------------
    # frame loop: play + think, one step each per frame
    # -----------------------------------------------------------------

    interrupted = False
    current_frame = start_frame

    while not exec_state.done:

        # get latest real state embedding
        z_t = encode_fn(exec_state.states[-1])

        result = run_frame(
            exec_state=exec_state,
            search_state=search_state,
            z_t=z_t,
            current_frame=current_frame,
            deadline_frame=deadline_frame,
            token_start_frame=start_frame,
            token_duration_frames=token_duration,
            max_search_budget=max_search_budget,
            send_controls_fn=send_controls_fn,
            read_state_fn=read_state_fn,
            **reward_kwargs,
        )

        current_frame += 1

        # if metacontroller says INTERRUPT, stop current token early
        if result["interrupt"]:
            interrupted = True
            break

    # -----------------------------------------------------------------
    # decide next token (D-03: fallback to planner top-1 if no commit)
    # -----------------------------------------------------------------

    is_fallback = False
    if search_state.chosen_token_id is not None:
        next_token_id = search_state.chosen_token_id
    else:
        next_token_id = fallback_token_id
        is_fallback = True  # D-04: flag for tracking no-decision rate

    rollout = get_rollout(exec_state)

    # -----------------------------------------------------------------
    # train metacontroller from realized outcomes
    # -----------------------------------------------------------------

    meta_result = train_step(
        rollout            = rollout,
        root               = search_state.root,
        committed_token_id = token_id,
        meta_trajectory    = search_state.meta_trajectory,
        meta_mlp           = search_state.meta_mlp,
        is_fallback        = is_fallback,
        nodes_expanded     = search_state.nodes_expanded,
        token_duration_frames = token_duration,
    )

    # -----------------------------------------------------------------
    # train reward head + rf_predictor against realized token return
    # -----------------------------------------------------------------

    reward_result = train_reward_head(
        rollout        = rollout,
        token_return   = meta_result["token_return"],
        encode_fn      = encode_fn,
        reward_mlp     = search_state.reward_mlp,
        rf_predictor   = search_state.rf_predictor,
        hidden_dim     = hidden_dim,
    )

    return {
        "rollout":       rollout,
        "next_token_id": next_token_id,
        "search_state":  search_state,
        "interrupted":   interrupted,
        "meta_mlp":      search_state.meta_mlp,
        "reward_mlp":    reward_result["reward_mlp"],
        "rf_predictor":  reward_result["rf_predictor"],
        "planner_mlp":   search_state.planner_mlp,
        "meta_loss":     meta_result["total_loss"],
        "reward_loss":   reward_result["reward_loss"],
        "rf_loss":       reward_result["rf_loss"],
        "token_return":  meta_result["token_return"],
        "is_fallback":   is_fallback,
    }
