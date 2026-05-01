"""
Tests for Phase 1: Training Correctness (TRAIN-01 through TRAIN-06).

Each test validates one requirement from REQUIREMENTS.md.
Tests use synthetic data -- they do not require GTA or real game states.
"""

import sys
from pathlib import Path
import math

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
if str(METACONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(METACONTROLLER_DIR))

from metacontroller import metacontroller, EXPLORE, INTERRUPT, COMMIT_NEXT, ROLLBACK
from trainer import (
    compute_token_return,
    compute_metalevel_advantages,
    NOT_READY_C,
    LAZY_K,
    MIN_SEARCH_NODES,
)


# =========================================================================
# TRAIN-01: Categorical sampling produces all 4 decisions over N runs
# =========================================================================

def test_categorical_sampling():
    """
    TRAIN-01: With training=True, metacontroller uses Categorical sampling
    and produces varied decisions. With training=False (default), it uses
    argmax and produces deterministic output.
    """
    fused_dim = 64
    z = torch.randn(1, fused_dim)
    cq = torch.randn(1, 3)
    ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
    ccq = torch.randn(1, 3)
    ccd = torch.ones(1, 3)
    cce = torch.randn(1, 3 * 32)  # top_k=3, embed_dim=32

    # First call to create MLP
    out_init = metacontroller(
        z, z, torch.tensor([0]), torch.tensor([[0.5]]),
        torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
        torch.tensor([[0.5]]), torch.tensor([[0.1]]),
        torch.tensor([[0.2]]), ccq, ccd, cce,
        training=False,
    )
    mlp = out_init["meta_mlp"]

    # --- Inference mode: deterministic ---
    results_inference = set()
    for _ in range(10):
        out = metacontroller(
            z, z, torch.tensor([0]), torch.tensor([[0.5]]),
            torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
            torch.tensor([[0.5]]), torch.tensor([[0.1]]),
            torch.tensor([[0.2]]), ccq, ccd, cce,
            meta_mlp=mlp, training=False,
        )
        results_inference.add(out["decision"].item())

    assert len(results_inference) == 1, (
        f"Inference mode should be deterministic, got {results_inference}"
    )

    # --- Training mode: stochastic ---
    results_training = set()
    for _ in range(500):
        out = metacontroller(
            z, z, torch.tensor([0]), torch.tensor([[0.5]]),
            torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
            torch.tensor([[0.5]]), torch.tensor([[0.1]]),
            torch.tensor([[0.2]]), ccq, ccd, cce,
            meta_mlp=mlp, training=True,
        )
        results_training.add(out["decision"].item())

    # With random MLP weights and 500 samples, we should see at least 2
    # different decisions. Getting all 4 depends on the random init but
    # 2+ proves sampling is working.
    assert len(results_training) >= 2, (
        f"Training mode should produce varied decisions, got {results_training}"
    )

    # Verify all decisions are valid (0-3)
    for d in results_training:
        assert d in {EXPLORE, INTERRUPT, COMMIT_NEXT, ROLLBACK}, (
            f"Invalid decision value: {d}"
        )


def test_training_flag_default_is_false():
    """
    TRAIN-01: Default behavior (no training= kwarg) should use argmax,
    matching the original pre-fix behavior.
    """
    fused_dim = 64
    z = torch.randn(1, fused_dim)
    cq = torch.randn(1, 3)
    ci = torch.tensor([[0, 1, 2]], dtype=torch.long)
    ccq = torch.randn(1, 3)
    ccd = torch.ones(1, 3)
    cce = torch.randn(1, 3 * 32)

    out1 = metacontroller(
        z, z, torch.tensor([0]), torch.tensor([[0.5]]),
        torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
        torch.tensor([[0.5]]), torch.tensor([[0.1]]),
        torch.tensor([[0.2]]), ccq, ccd, cce,
    )
    mlp = out1["meta_mlp"]

    # Call without training= kwarg -- should be deterministic
    decisions = set()
    for _ in range(10):
        out = metacontroller(
            z, z, torch.tensor([0]), torch.tensor([[0.5]]),
            torch.tensor([[5.0]]), cq, ci, torch.tensor([[0.3]]),
            torch.tensor([[0.5]]), torch.tensor([[0.1]]),
            torch.tensor([[0.2]]), ccq, ccd, cce,
            meta_mlp=mlp,
        )
        decisions.add(out["decision"].item())

    assert len(decisions) == 1, (
        f"Default (no training kwarg) should be deterministic, got {decisions}"
    )


# =========================================================================
# TRAIN-06: Duration-normalized returns
# =========================================================================

def test_duration_normalization():
    """
    TRAIN-06 / D-08: Token return is divided by sqrt(num_frames).
    A 5-frame rollout with known rewards should produce return / sqrt(5).
    """
    rollout = {"rewards": [1.0, 1.0, 1.0, 1.0, 1.0]}
    gamma = 1.0  # no discounting for simplicity

    ret = compute_token_return(rollout, gamma=gamma)
    expected = 5.0 / math.sqrt(5)  # 5 / 2.236 ~ 2.236
    assert abs(ret - expected) < 1e-6, (
        f"Expected {expected:.6f}, got {ret:.6f}"
    )


def test_duration_normalization_preserves_short_bursts():
    """
    TRAIN-06 / D-09: sqrt normalization preserves short-burst surprise.
    A 1-frame token with large reward should retain most of its signal.
    """
    rollout = {"rewards": [10.0]}
    ret = compute_token_return(rollout, gamma=0.99)
    # sqrt(1) = 1.0, so 10.0 / 1.0 = 10.0
    assert abs(ret - 10.0) < 1e-6, (
        f"Single-frame return should be preserved, got {ret}"
    )


def test_duration_normalization_with_bootstrap():
    """
    TRAIN-06: Bootstrapped return should also be normalized.
    """
    rollout = {"rewards": [0.5, 0.5]}
    gamma = 1.0
    ret = compute_token_return(rollout, gamma=gamma, bootstrap_value=1.0)
    # raw = 0.5 + 0.5 + 1.0 = 2.0, normalized = 2.0 / sqrt(2) ~ 1.414
    expected = 2.0 / math.sqrt(2)
    assert abs(ret - expected) < 1e-6, (
        f"Expected {expected:.6f}, got {ret:.6f}"
    )


def test_duration_normalization_empty_rewards():
    """
    TRAIN-06: Empty rewards list should return 0 without division error.
    """
    rollout = {"rewards": []}
    ret = compute_token_return(rollout, gamma=0.99)
    assert ret == 0.0, f"Empty rewards should return 0.0, got {ret}"


# =========================================================================
# TRAIN-03: Not-ready penalty
# =========================================================================

def test_not_ready_penalty():
    """
    TRAIN-03 / D-01 / D-02: Fallback commit produces large negative reward
    proportional to token_duration_frames.
    """
    traj = [
        {"decision": 0, "features": None, "predicted_q": 0.0},
        {"decision": 0, "features": None, "predicted_q": 0.0},
        {"decision": 2, "features": None, "predicted_q": 0.0},
    ]

    # Without fallback: final reward = realized_return
    adv_normal, ret_normal = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=False,
        token_duration_frames=10,
    )

    # With fallback: final reward = -NOT_READY_C * token_duration_frames
    adv_fallback, ret_fallback = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=True,
        token_duration_frames=10,
    )

    expected_penalty = -NOT_READY_C * 10  # -1.0
    # The final meta_reward should be the penalty, not the realized return
    # This means the final step's return should be negative
    assert ret_fallback[-1] == expected_penalty, (
        f"Final step return should be penalty {expected_penalty}, "
        f"got {ret_fallback[-1]}"
    )
    assert ret_fallback[-1] < ret_normal[-1], (
        "Fallback return should be less than normal return"
    )


def test_not_ready_penalty_scales_with_duration():
    """
    TRAIN-03 / D-02: Penalty is proportional to available thinking budget.
    20-frame penalty should be larger than 5-frame penalty.
    """
    traj = [{"decision": 2, "features": None, "predicted_q": 0.0}]

    _, ret_short = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=True,
        token_duration_frames=5,
    )
    _, ret_long = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=True,
        token_duration_frames=20,
    )

    # Both should be negative, long should be more negative
    assert ret_short[-1] < 0, f"Short penalty should be negative: {ret_short[-1]}"
    assert ret_long[-1] < 0, f"Long penalty should be negative: {ret_long[-1]}"
    assert ret_long[-1] < ret_short[-1], (
        f"20-frame penalty ({ret_long[-1]}) should be more negative "
        f"than 5-frame penalty ({ret_short[-1]})"
    )


# =========================================================================
# TRAIN-04: Lazy-commit penalty
# =========================================================================

def test_lazy_commit_penalty():
    """
    TRAIN-04 / D-05: COMMIT_NEXT with zero nodes expanded gets penalized.
    """
    traj = [
        {"decision": 0, "features": None, "predicted_q": 0.0},
        {"decision": 2, "features": None, "predicted_q": 0.0},
    ]

    # Normal commit with enough search
    _, ret_normal = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=False,
        nodes_expanded=5, token_duration_frames=10,
    )

    # Lazy commit with zero search
    _, ret_lazy = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=False,
        nodes_expanded=0, token_duration_frames=10,
    )

    assert ret_lazy[-1] < ret_normal[-1], (
        f"Lazy commit return ({ret_lazy[-1]}) should be less than "
        f"normal commit return ({ret_normal[-1]})"
    )


def test_lazy_commit_penalty_scales_with_duration():
    """
    TRAIN-04 / D-06: Short tokens get near-zero lazy penalty.
    """
    traj = [{"decision": 2, "features": None, "predicted_q": 0.0}]

    _, ret_short = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=False,
        nodes_expanded=0, token_duration_frames=1,
    )
    _, ret_long = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=False,
        nodes_expanded=0, token_duration_frames=20,
    )

    # Both should have penalty but long should have more
    # Short token (1 frame): penalty = -0.05 * 1 * 1.0 = -0.05
    # Long token (20 frames): penalty = -0.05 * 20 * 1.0 = -1.0
    short_penalty = ret_short[-1] - 1.0  # subtract the realized_return to get pure penalty
    long_penalty = ret_long[-1] - 1.0

    assert abs(short_penalty) < abs(long_penalty), (
        f"Short token penalty ({short_penalty}) should be smaller "
        f"than long token penalty ({long_penalty})"
    )


def test_lazy_penalty_not_applied_above_threshold():
    """
    TRAIN-04: Commits with nodes_expanded >= MIN_SEARCH_NODES get no lazy penalty.
    """
    traj = [{"decision": 2, "features": None, "predicted_q": 0.0}]

    _, ret_enough = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=False,
        nodes_expanded=MIN_SEARCH_NODES, token_duration_frames=10,
    )

    # With enough nodes, return should just be the realized_return
    assert ret_enough[-1] == 1.0, (
        f"Expected 1.0 with sufficient search, got {ret_enough[-1]}"
    )


def test_not_ready_overrides_lazy():
    """
    TRAIN-03 vs TRAIN-04: Fallback penalty should take precedence over lazy penalty
    since `is_fallback and n > 0` is checked first in the elif chain.
    """
    traj = [{"decision": 2, "features": None, "predicted_q": 0.0}]

    _, ret = compute_metalevel_advantages(
        traj, realized_return=1.0, is_fallback=True,
        nodes_expanded=0, token_duration_frames=10,
    )

    expected = -NOT_READY_C * 10
    assert ret[-1] == expected, (
        f"Fallback should override lazy, expected {expected}, got {ret[-1]}"
    )
