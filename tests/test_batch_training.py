"""
Tests for Phase 2: Batch Training and Checkpointing (BATCH-01 through BATCH-04).

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
TESTS_DIR = PROJECT_ROOT / "tests"
if str(METACONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(METACONTROLLER_DIR))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from trainer import (
    TrainingState,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BUFFER_CAPACITY,
    DEFAULT_LR,
    DEFAULT_EPS,
    DEFAULT_MAX_GRAD_NORM,
)
from conftest import _make_trajectory_dict


# =========================================================================
# BATCH-01: Trajectory replay buffer accumulates and evicts
# =========================================================================

def test_buffer_accumulation():
    """
    BATCH-01: Buffer accumulates trajectory dicts and reports not-ready
    until batch_size is reached.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=4, buffer_capacity=100)

    results = []
    for i in range(3):
        result = ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
        results.append(result)

    assert len(ts.buffer) == 3, (
        f"Buffer should have 3 entries, got {len(ts.buffer)}"
    )
    assert all(r is False for r in results), (
        f"All 3 add_trajectory calls should return False, got {results}"
    )


def test_buffer_eviction():
    """
    BATCH-01: Buffer evicts oldest entries when capacity is exceeded.
    deque(maxlen=N) automatically discards the oldest on append.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=4, buffer_capacity=5)

    # Add first trajectory with a unique marker
    first_traj = _make_trajectory_dict(input_dim=input_dim)
    first_traj["marker"] = "FIRST"
    ts.add_trajectory(first_traj)

    # Add 6 more (total 7, capacity 5)
    for i in range(6):
        ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))

    assert len(ts.buffer) == 5, (
        f"Buffer should cap at 5 entries, got {len(ts.buffer)}"
    )
    # Oldest entries should have been evicted
    assert ts.buffer[0].get("marker") != "FIRST", (
        "First trajectory should have been evicted but marker 'FIRST' found at position 0"
    )


# =========================================================================
# BATCH-02: Batch update triggers every N trajectories
# =========================================================================

def test_batch_trigger_every_n():
    """
    BATCH-02: add_trajectory returns True on every batch_size-th call,
    signaling that a batch update should occur.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=4, buffer_capacity=100)

    # First 4 trajectories
    results_1 = []
    for i in range(4):
        results_1.append(ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim)))

    assert results_1[-1] is True, (
        f"4th trajectory should trigger update, got {results_1[-1]}"
    )

    # Reset counter by calling get_batch
    ts.get_batch()

    # Next 4 trajectories
    results_2 = []
    for i in range(4):
        results_2.append(ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim)))

    assert results_2[-1] is True, (
        f"8th trajectory (4th after reset) should trigger update, got {results_2[-1]}"
    )


def test_no_premature_update():
    """
    BATCH-02: add_trajectory returns False for trajectories 1 through
    batch_size-1. Only the batch_size-th trajectory returns True.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=4, buffer_capacity=100)

    for i in range(3):
        result = ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
        assert result is False, (
            f"Trajectory {i+1} of 4 should not trigger update, got {result}"
        )

    result = ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
    assert result is True, (
        f"Trajectory 4 of 4 should trigger update, got {result}"
    )


def test_trajectories_since_update_resets():
    """
    BATCH-02: After get_batch(), the trajectories_since_update counter
    resets to 0 so the next batch trigger happens after another batch_size
    trajectories.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=4, buffer_capacity=100)

    # Fill one batch
    for i in range(4):
        ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))

    assert ts.trajectories_since_update == 4, (
        f"Before get_batch: expected 4, got {ts.trajectories_since_update}"
    )

    ts.get_batch()

    assert ts.trajectories_since_update == 0, (
        f"After get_batch: expected 0, got {ts.trajectories_since_update}"
    )

    # Add one more
    ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
    assert ts.trajectories_since_update == 1, (
        f"After adding 1 more: expected 1, got {ts.trajectories_since_update}"
    )


# =========================================================================
# BATCH-03: Adam optimizer replaces manual SGD
# =========================================================================

def test_adam_updates_weights():
    """
    BATCH-03: After a batch update via update_metapolicy_batch, at least
    one parameter in meta_mlp must have changed (proving Adam stepped).
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2, buffer_capacity=100)

    # Record initial weights
    initial_params = [p.clone().detach() for p in meta.parameters()]

    # Add batch_size trajectories
    for i in range(2):
        ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))

    batch = ts.get_batch()
    result = ts.update_metapolicy_batch(meta, batch)

    # At least one parameter should have changed
    any_changed = False
    for p_init, p_new in zip(initial_params, meta.parameters()):
        if not torch.allclose(p_init, p_new.data):
            any_changed = True
            break

    assert any_changed, (
        "Adam should have updated at least one parameter, but all remained the same"
    )
    assert isinstance(result["loss"], float), (
        f"Loss should be float, got {type(result['loss'])}"
    )


def test_adam_state_persists():
    """
    BATCH-03: Adam optimizer state (momentum/variance estimates) persists
    across batch flushes. After two rounds of updates, the optimizer's
    internal step count should be 2 and exp_avg tensors should be non-zero.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2, buffer_capacity=100)

    # --- Round 1 ---
    for i in range(2):
        ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
    batch1 = ts.get_batch()
    ts.update_metapolicy_batch(meta, batch1)

    # Check optimizer state after first update
    opt_state = ts.optimizer_meta.state
    assert len(opt_state) > 0, (
        "Optimizer state should be populated after first update"
    )

    # Get state for first parameter
    first_param = list(meta.parameters())[0]
    param_state = opt_state[first_param]
    assert param_state["step"].item() == 1, (
        f"Step count should be 1 after first update, got {param_state['step'].item()}"
    )

    # exp_avg (first moment / momentum) should be non-zero after a gradient step
    exp_avg = param_state["exp_avg"]
    assert not torch.allclose(exp_avg, torch.zeros_like(exp_avg)), (
        "exp_avg (momentum) should be non-zero after first update"
    )

    # --- Round 2 ---
    for i in range(2):
        ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
    batch2 = ts.get_batch()
    ts.update_metapolicy_batch(meta, batch2)

    param_state_2 = opt_state[first_param]
    assert param_state_2["step"].item() == 2, (
        f"Step count should be 2 after second update, got {param_state_2['step'].item()}"
    )


# =========================================================================
# BATCH-04: Gradient clipping with clip event reporting
# =========================================================================

def test_gradient_clipping():
    """
    BATCH-04: When gradients are large (extreme advantages), gradient norm
    is clipped to max_grad_norm and the return dict reports clipped=True.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2, buffer_capacity=100, max_grad_norm=0.5)

    # Create trajectories with extreme advantages to force large gradients
    for i in range(2):
        traj = _make_trajectory_dict(input_dim=input_dim)
        # Manipulate the realized_return to create extreme advantages
        traj["rollout"]["rewards"] = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        ts.add_trajectory(traj)

    batch = ts.get_batch()
    result = ts.update_metapolicy_batch(meta, batch)

    assert "grad_norm" in result, (
        "Return dict should contain 'grad_norm' key"
    )
    assert "clipped" in result, (
        "Return dict should contain 'clipped' key"
    )
    assert result["clipped"] is True, (
        f"Extreme advantages should trigger clipping, got clipped={result['clipped']}, "
        f"grad_norm={result['grad_norm']}, max_grad_norm={ts.max_grad_norm}"
    )


def test_clip_event_reported():
    """
    BATCH-04: With a very large max_grad_norm, normal trajectories should
    NOT trigger clipping. The return dict should report clipped=False.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    # Very large max_grad_norm -- should not clip with normal data
    ts = TrainingState(meta, rw, rf, batch_size=2, buffer_capacity=100, max_grad_norm=100.0)

    for i in range(2):
        ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))

    batch = ts.get_batch()
    result = ts.update_metapolicy_batch(meta, batch)

    assert result["clipped"] is False, (
        f"Normal gradients with max_grad_norm=100.0 should not clip, "
        f"got clipped={result['clipped']}, grad_norm={result['grad_norm']}"
    )


def test_gradient_clipping_limits_norm():
    """
    BATCH-04: After clipping, the actual gradient norm applied should be
    bounded by max_grad_norm. We verify this by checking that the reported
    grad_norm is the pre-clip value (for logging), and that the weights
    did not change drastically despite extreme gradients.
    """
    input_dim = 10
    max_norm = 0.5

    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2, buffer_capacity=100, max_grad_norm=max_norm)

    # Record initial weights
    initial_params = [p.clone().detach() for p in meta.parameters()]

    # Create extreme trajectories
    for i in range(2):
        traj = _make_trajectory_dict(input_dim=input_dim)
        traj["rollout"]["rewards"] = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        ts.add_trajectory(traj)

    batch = ts.get_batch()
    result = ts.update_metapolicy_batch(meta, batch)

    # The pre-clip gradient norm should exceed max_grad_norm
    assert result["grad_norm"] > max_norm, (
        f"Pre-clip grad_norm should exceed {max_norm}, got {result['grad_norm']}"
    )

    # After clipping and stepping, verify the weight change is bounded.
    # With a small max_norm, the weight change per step should be small.
    max_weight_change = 0.0
    for p_init, p_new in zip(initial_params, meta.parameters()):
        change = (p_new.data - p_init).abs().max().item()
        max_weight_change = max(max_weight_change, change)

    # With lr=3e-4 and max_grad_norm=0.5, the max weight change should be
    # bounded. The exact bound depends on Adam's internal state, but it
    # should be much smaller than without clipping.
    assert max_weight_change < 1.0, (
        f"Weight change should be bounded by gradient clipping, "
        f"got max change = {max_weight_change}"
    )


# =========================================================================
# Constants verification
# =========================================================================

def test_default_constants():
    """
    Verify Phase 2 constants are exported with correct values.
    """
    assert DEFAULT_BATCH_SIZE == 8, (
        f"DEFAULT_BATCH_SIZE should be 8, got {DEFAULT_BATCH_SIZE}"
    )
    assert DEFAULT_BUFFER_CAPACITY == 10000, (
        f"DEFAULT_BUFFER_CAPACITY should be 10000, got {DEFAULT_BUFFER_CAPACITY}"
    )
    assert DEFAULT_LR == 3e-4, (
        f"DEFAULT_LR should be 3e-4, got {DEFAULT_LR}"
    )
    assert DEFAULT_EPS == 1e-5, (
        f"DEFAULT_EPS should be 1e-5, got {DEFAULT_EPS}"
    )
    assert DEFAULT_MAX_GRAD_NORM == 0.5, (
        f"DEFAULT_MAX_GRAD_NORM should be 0.5, got {DEFAULT_MAX_GRAD_NORM}"
    )
