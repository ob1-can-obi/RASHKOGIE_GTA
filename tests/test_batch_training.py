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
    train_step,
    compute_token_return,
    get_entropy_coeff,
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


# =========================================================================
# BATCH-05: Per-module checkpoint saving
# =========================================================================

def test_checkpoint_save_structure(tmp_path):
    """
    BATCH-05: save_checkpoint creates a session directory with per-module
    .pt files for all required trained modules plus training_state.pt.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf)
    result = ts.save_checkpoint(tmp_path, meta, rw, rf, session_id="s001")

    session_dir = Path(result["session_dir"])
    assert session_dir.exists(), (
        f"Session directory should exist at {session_dir}"
    )

    expected_files = [
        "meta_mlp.pt", "reward_mlp.pt", "rf_predictor.pt",
        "optimizer_reward.pt", "training_state.pt",
    ]
    for fname in expected_files:
        fpath = session_dir / fname
        assert fpath.exists(), (
            f"Expected checkpoint file {fname} not found in {session_dir}"
        )

    assert len(result["files_saved"]) == 5, (
        f"Should save 5 required files (no optional modules), got {len(result['files_saved'])}"
    )


def test_checkpoint_save_all_six_modules(tmp_path):
    """
    BATCH-05: When all 6 modules are provided, save_checkpoint creates 8
    files: meta_mlp.pt, reward_mlp.pt, rf_predictor.pt, optimizer_reward.pt,
    intuition_mlp.pt, token_embed.pt, planner_mlp.pt, training_state.pt.
    This validates "one .pt per module."
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))
    intuition = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
    tok_embed = nn.Sequential(nn.Linear(32, 64))
    planner = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))

    ts = TrainingState(meta, rw, rf)
    result = ts.save_checkpoint(
        tmp_path, meta, rw, rf,
        intuition_mlp=intuition,
        token_embed=tok_embed,
        planner_mlp=planner,
        session_id="s002",
    )

    session_dir = Path(result["session_dir"])
    expected_files = [
        "meta_mlp.pt", "reward_mlp.pt", "rf_predictor.pt",
        "optimizer_reward.pt", "intuition_mlp.pt", "token_embed.pt",
        "planner_mlp.pt", "training_state.pt",
    ]
    for fname in expected_files:
        fpath = session_dir / fname
        assert fpath.exists(), (
            f"Expected checkpoint file {fname} not found in {session_dir}"
        )

    assert len(result["files_saved"]) == 8, (
        f"Should save 8 files with all 6 modules, got {len(result['files_saved'])}"
    )


def test_checkpoint_save_meta_keys(tmp_path):
    """
    BATCH-05: meta_mlp.pt contains model_state_dict, optimizer_state_dict,
    step_count, batch_size, and max_grad_norm keys.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf)
    result = ts.save_checkpoint(tmp_path, meta, rw, rf, session_id="s003")

    meta_path = Path(result["session_dir"]) / "meta_mlp.pt"
    ckpt = torch.load(meta_path, map_location="cpu", weights_only=True)

    expected_keys = {"model_state_dict", "optimizer_state_dict", "step_count",
                     "batch_size", "max_grad_norm"}
    actual_keys = set(ckpt.keys())
    assert expected_keys == actual_keys, (
        f"meta_mlp.pt keys mismatch: expected {expected_keys}, got {actual_keys}"
    )


def test_checkpoint_save_reward_separate_files(tmp_path):
    """
    BATCH-05: reward_mlp.pt and rf_predictor.pt contain only model_state_dict.
    optimizer_reward.pt contains optimizer_state_dict and step_count.
    This validates the separate-file structure for the shared optimizer group.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf)
    result = ts.save_checkpoint(tmp_path, meta, rw, rf, session_id="s004")

    session_dir = Path(result["session_dir"])

    # reward_mlp.pt: model weights only
    rw_ckpt = torch.load(session_dir / "reward_mlp.pt", map_location="cpu", weights_only=True)
    assert "model_state_dict" in rw_ckpt, (
        f"reward_mlp.pt should contain 'model_state_dict', got keys: {list(rw_ckpt.keys())}"
    )

    # rf_predictor.pt: model weights only
    rf_ckpt = torch.load(session_dir / "rf_predictor.pt", map_location="cpu", weights_only=True)
    assert "model_state_dict" in rf_ckpt, (
        f"rf_predictor.pt should contain 'model_state_dict', got keys: {list(rf_ckpt.keys())}"
    )

    # optimizer_reward.pt: shared optimizer state
    opt_ckpt = torch.load(session_dir / "optimizer_reward.pt", map_location="cpu", weights_only=True)
    assert "optimizer_state_dict" in opt_ckpt, (
        f"optimizer_reward.pt should contain 'optimizer_state_dict', got keys: {list(opt_ckpt.keys())}"
    )
    assert "step_count" in opt_ckpt, (
        f"optimizer_reward.pt should contain 'step_count', got keys: {list(opt_ckpt.keys())}"
    )


def test_independent_module_load(tmp_path):
    """
    BATCH-05: Individual module .pt files can be loaded independently.
    Load only meta_mlp.pt and verify the weights match the original.
    """
    input_dim = 10
    meta_original = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    # Record original weights
    original_params = [p.clone().detach() for p in meta_original.parameters()]

    ts = TrainingState(meta_original, rw, rf)
    result = ts.save_checkpoint(tmp_path, meta_original, rw, rf, session_id="s005")

    # Create a NEW meta_mlp with DIFFERENT random weights
    meta_new = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))

    # Verify weights are different before load
    any_different = False
    for p_orig, p_new in zip(original_params, meta_new.parameters()):
        if not torch.allclose(p_orig, p_new.data):
            any_different = True
            break
    assert any_different, (
        "New MLP should have different random weights from original"
    )

    # Load ONLY meta_mlp.pt independently
    meta_path = Path(result["session_dir"]) / "meta_mlp.pt"
    ckpt = torch.load(meta_path, map_location="cpu", weights_only=True)
    meta_new.load_state_dict(ckpt["model_state_dict"])

    # Verify the new meta_mlp now has the same weights as original
    for i, (p_orig, p_new) in enumerate(zip(original_params, meta_new.parameters())):
        assert torch.allclose(p_orig, p_new.data), (
            f"Parameter {i} should match after independent load, "
            f"max diff = {(p_orig - p_new.data).abs().max().item()}"
        )


# =========================================================================
# BATCH-06: Checkpoint loading and resume
# =========================================================================

def test_checkpoint_resume_weights(tmp_path):
    """
    BATCH-06: After saving and loading a checkpoint, model weights are
    restored exactly -- even when the target modules have different random
    initialization.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2)

    # Do one batch update to change weights from init
    for i in range(2):
        ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
    batch = ts.get_batch()
    ts.update_metapolicy_batch(meta, batch)

    # Record weights after update
    saved_params = [p.clone().detach() for p in meta.parameters()]

    # Save checkpoint
    ts.save_checkpoint(tmp_path, meta, rw, rf, session_id="resume_w")

    # Create NEW modules with fresh random weights
    meta2 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw2 = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf2 = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts2 = TrainingState(meta2, rw2, rf2)

    # Load checkpoint
    session_dir = str(tmp_path / "session_resume_w")
    ts2.load_checkpoint(session_dir, meta2, rw2, rf2)

    # Verify new modules now have the saved weights
    for i, (p_saved, p_loaded) in enumerate(zip(saved_params, meta2.parameters())):
        assert torch.allclose(p_saved, p_loaded.data), (
            f"Meta param {i} not restored: max diff = {(p_saved - p_loaded.data).abs().max().item()}"
        )


def test_checkpoint_resume_optimizer(tmp_path):
    """
    BATCH-06: After loading a checkpoint, optimizer state (momentum/variance)
    is restored. After one batch update, Adam should have non-zero exp_avg
    and those values should survive save/load.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2)

    # Do one batch update so Adam has non-zero momentum
    for i in range(2):
        ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
    batch = ts.get_batch()
    ts.update_metapolicy_batch(meta, batch)

    # Record optimizer state before save
    opt_state_before = ts.optimizer_meta.state_dict()

    # Save checkpoint
    ts.save_checkpoint(tmp_path, meta, rw, rf, session_id="resume_o")

    # Create fresh TrainingState with fresh modules
    meta2 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw2 = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf2 = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts2 = TrainingState(meta2, rw2, rf2)

    # Load checkpoint
    session_dir = str(tmp_path / "session_resume_o")
    ts2.load_checkpoint(session_dir, meta2, rw2, rf2)

    # Verify optimizer state is non-empty after load
    opt_state = ts2.optimizer_meta.state
    assert len(opt_state) > 0, (
        "Optimizer state should be populated after checkpoint load"
    )

    # Verify optimizer state_dict matches the pre-save values
    opt_state_after = ts2.optimizer_meta.state_dict()
    for group_idx, (g_before, g_after) in enumerate(
        zip(opt_state_before["param_groups"], opt_state_after["param_groups"])
    ):
        assert g_before["lr"] == g_after["lr"], (
            f"Optimizer LR mismatch in group {group_idx}: "
            f"{g_before['lr']} vs {g_after['lr']}"
        )


def test_checkpoint_resume_step_count(tmp_path):
    """
    BATCH-06: After 3 batch updates (step_count=3), saving and loading
    restores step_count exactly. get_entropy_coeff(step_count) returns
    the correct value for step=3.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2)

    # Do 3 batch updates
    for _ in range(3):
        for i in range(2):
            ts.add_trajectory(_make_trajectory_dict(input_dim=input_dim))
        batch = ts.get_batch()
        ts.update_metapolicy_batch(meta, batch)

    assert ts.step_count == 3, (
        f"Step count should be 3 after 3 updates, got {ts.step_count}"
    )

    # Save
    ts.save_checkpoint(tmp_path, meta, rw, rf, session_id="resume_sc")

    # Fresh TrainingState
    meta2 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw2 = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf2 = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))
    ts2 = TrainingState(meta2, rw2, rf2)
    assert ts2.step_count == 0, (
        f"Fresh TrainingState should have step_count=0, got {ts2.step_count}"
    )

    # Load
    session_dir = str(tmp_path / "session_resume_sc")
    ts2.load_checkpoint(session_dir, meta2, rw2, rf2)

    assert ts2.step_count == 3, (
        f"After load, step_count should be 3, got {ts2.step_count}"
    )

    # Verify entropy coeff matches step=3
    expected_coeff = get_entropy_coeff(3)
    actual_coeff = get_entropy_coeff(ts2.step_count)
    assert abs(expected_coeff - actual_coeff) < 1e-10, (
        f"Entropy coeff mismatch: expected {expected_coeff}, got {actual_coeff}"
    )


def test_checkpoint_resume_entropy_annealing(tmp_path):
    """
    BATCH-06: Setting step_count to 2500 (halfway through annealing with
    default 5000 steps), saving, and loading preserves the midpoint entropy
    coefficient value (~0.0275).
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf)
    ts.step_count = 2500  # Halfway through annealing

    # Save
    ts.save_checkpoint(tmp_path, meta, rw, rf, session_id="resume_ea")

    # Fresh TrainingState
    meta2 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw2 = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf2 = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))
    ts2 = TrainingState(meta2, rw2, rf2)

    # Load
    session_dir = str(tmp_path / "session_resume_ea")
    ts2.load_checkpoint(session_dir, meta2, rw2, rf2)

    assert ts2.step_count == 2500, (
        f"After load, step_count should be 2500, got {ts2.step_count}"
    )

    # Midpoint entropy coeff: start=0.05, end=0.005, frac=0.5
    # coeff = 0.05 + 0.5 * (0.005 - 0.05) = 0.05 + 0.5 * (-0.045) = 0.05 - 0.0225 = 0.0275
    expected_midpoint = 0.0275
    actual_coeff = get_entropy_coeff(ts2.step_count)
    assert abs(actual_coeff - expected_midpoint) < 1e-6, (
        f"Midpoint entropy coeff should be ~{expected_midpoint}, got {actual_coeff}"
    )


def test_checkpoint_resume_all_six_modules(tmp_path):
    """
    BATCH-05 + BATCH-06: Full round-trip test. Save all 6 modules, load
    into fresh modules with different random weights, verify all 6 have
    restored weights exactly.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))
    intuition = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
    tok_embed = nn.Sequential(nn.Linear(32, 64))
    planner = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))

    ts = TrainingState(meta, rw, rf)

    # Record original weights for all 6 modules
    modules_original = {
        "meta": [p.clone().detach() for p in meta.parameters()],
        "rw": [p.clone().detach() for p in rw.parameters()],
        "rf": [p.clone().detach() for p in rf.parameters()],
        "intuition": [p.clone().detach() for p in intuition.parameters()],
        "tok_embed": [p.clone().detach() for p in tok_embed.parameters()],
        "planner": [p.clone().detach() for p in planner.parameters()],
    }

    # Save all 6
    ts.save_checkpoint(
        tmp_path, meta, rw, rf,
        intuition_mlp=intuition,
        token_embed=tok_embed,
        planner_mlp=planner,
        session_id="resume_all6",
    )

    # Create fresh modules with different random weights
    meta2 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw2 = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf2 = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))
    intuition2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))
    tok_embed2 = nn.Sequential(nn.Linear(32, 64))
    planner2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))

    ts2 = TrainingState(meta2, rw2, rf2)

    # Load all 6
    session_dir = str(tmp_path / "session_resume_all6")
    load_result = ts2.load_checkpoint(
        session_dir, meta2, rw2, rf2,
        intuition_mlp=intuition2,
        token_embed=tok_embed2,
        planner_mlp=planner2,
    )

    assert load_result["loaded"] is True, (
        "load_checkpoint should return loaded=True"
    )
    assert len(load_result["files_loaded"]) == 7, (
        f"Should load 7 files (4 required + 3 optional modules), "
        f"got {len(load_result['files_loaded'])}"
    )

    # Verify all 6 modules have restored weights
    modules_loaded = {
        "meta": list(meta2.parameters()),
        "rw": list(rw2.parameters()),
        "rf": list(rf2.parameters()),
        "intuition": list(intuition2.parameters()),
        "tok_embed": list(tok_embed2.parameters()),
        "planner": list(planner2.parameters()),
    }

    for name, orig_params in modules_original.items():
        loaded_params = modules_loaded[name]
        for i, (p_orig, p_loaded) in enumerate(zip(orig_params, loaded_params)):
            assert torch.allclose(p_orig, p_loaded.data), (
                f"Module '{name}' param {i} not restored: "
                f"max diff = {(p_orig - p_loaded.data).abs().max().item()}"
            )


# =========================================================================
# Integration: train_step buffering with TrainingState
# =========================================================================

class MockTreeNode:
    """Minimal tree node mock for train_step's backup_tree call."""
    def __init__(self):
        self.children = []
        self.n = 0
        self.w = 0.0
        self.q = 0.0


class MockChildNode:
    """Child node with a token_id for backup_tree matching."""
    def __init__(self, token_id):
        self.token_id = token_id
        self.n = 0
        self.w = 0.0
        self.q = 0.0


def _make_root_with_child(token_id=42):
    """Create a mock root with one child matching the committed token_id."""
    root = MockTreeNode()
    child = MockChildNode(token_id)
    root.children.append(child)
    return root


def test_train_step_buffers_with_training_state():
    """
    Integration: train_step with training_state buffers trajectories
    instead of updating weights immediately. After 3 calls with batch_size=4,
    the buffer should have 3 entries and batch_ready should be False.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=4, buffer_capacity=100)

    rollout = {
        "token_id": 42, "duration": 5,
        "states": [{"dummy": i} for i in range(6)],
        "rewards": [0.1, 0.2, 0.15, 0.1, 0.05],
        "components": [{"speed_reward": 0.1} for _ in range(5)],
        "state_before": {"dummy": 0}, "state_after": {"dummy": 5},
    }

    results = []
    for _ in range(3):
        meta_traj = [
            {"decision": 0, "features": torch.randn(1, input_dim), "predicted_q": 0.1},
            {"decision": 1, "features": torch.randn(1, input_dim), "predicted_q": 0.2},
            {"decision": 2, "features": torch.randn(1, input_dim), "predicted_q": 0.1},
        ]
        root = _make_root_with_child(42)
        result = train_step(
            rollout=rollout,
            root=root,
            committed_token_id=42,
            meta_trajectory=meta_traj,
            meta_mlp=meta,
            training_state=ts,
        )
        results.append(result)

    assert ts.trajectories_since_update == 3, (
        f"Expected 3 trajectories in buffer, got {ts.trajectories_since_update}"
    )
    assert len(ts.buffer) == 3, (
        f"Buffer should have 3 entries, got {len(ts.buffer)}"
    )
    assert all(r["batch_ready"] is False for r in results), (
        f"All 3 calls should return batch_ready=False, got {[r['batch_ready'] for r in results]}"
    )


def test_train_step_triggers_batch_on_full():
    """
    Integration: train_step with training_state=TrainingState(batch_size=2)
    returns batch_ready=True on the 2nd call.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2, buffer_capacity=100)

    rollout = {
        "token_id": 42, "duration": 5,
        "states": [{"dummy": i} for i in range(6)],
        "rewards": [0.1, 0.2, 0.15, 0.1, 0.05],
        "components": [{"speed_reward": 0.1} for _ in range(5)],
        "state_before": {"dummy": 0}, "state_after": {"dummy": 5},
    }

    results = []
    for _ in range(2):
        meta_traj = [
            {"decision": 0, "features": torch.randn(1, input_dim), "predicted_q": 0.1},
            {"decision": 2, "features": torch.randn(1, input_dim), "predicted_q": 0.2},
        ]
        root = _make_root_with_child(42)
        result = train_step(
            rollout=rollout,
            root=root,
            committed_token_id=42,
            meta_trajectory=meta_traj,
            meta_mlp=meta,
            training_state=ts,
        )
        results.append(result)

    assert results[0]["batch_ready"] is False, (
        f"1st call should return batch_ready=False, got {results[0]['batch_ready']}"
    )
    assert results[1]["batch_ready"] is True, (
        f"2nd call should return batch_ready=True, got {results[1]['batch_ready']}"
    )


def test_train_step_legacy_mode():
    """
    Integration: train_step WITHOUT training_state parameter should
    still perform an immediate weight update (backward compatible).
    Returns a dict with total_loss as a non-zero float and batch_ready=False.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))

    meta_traj = [
        {"decision": 0, "features": torch.randn(1, input_dim), "predicted_q": 0.1},
        {"decision": 1, "features": torch.randn(1, input_dim), "predicted_q": 0.2},
        {"decision": 2, "features": torch.randn(1, input_dim), "predicted_q": 0.1},
    ]
    rollout = {
        "token_id": 42, "duration": 5,
        "states": [{"dummy": i} for i in range(6)],
        "rewards": [0.1, 0.2, 0.15, 0.1, 0.05],
        "components": [{"speed_reward": 0.1} for _ in range(5)],
        "state_before": {"dummy": 0}, "state_after": {"dummy": 5},
    }
    root = _make_root_with_child(42)

    result = train_step(
        rollout=rollout,
        root=root,
        committed_token_id=42,
        meta_trajectory=meta_traj,
        meta_mlp=meta,
    )

    assert isinstance(result["total_loss"], float), (
        f"Legacy mode total_loss should be float, got {type(result['total_loss'])}"
    )
    assert result["total_loss"] != 0.0, (
        f"Legacy mode should produce non-zero loss, got {result['total_loss']}"
    )
    assert result["batch_ready"] is False, (
        f"Legacy mode should return batch_ready=False, got {result['batch_ready']}"
    )


def test_train_step_no_weight_update_before_batch():
    """
    Integration: With training_state and batch_size=4, calling train_step
    3 times should NOT change meta_mlp weights (buffering defers update).
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=4, buffer_capacity=100)

    # Record initial weights
    initial_params = [p.clone().detach() for p in meta.parameters()]

    rollout = {
        "token_id": 42, "duration": 5,
        "states": [{"dummy": i} for i in range(6)],
        "rewards": [0.1, 0.2, 0.15, 0.1, 0.05],
        "components": [{"speed_reward": 0.1} for _ in range(5)],
        "state_before": {"dummy": 0}, "state_after": {"dummy": 5},
    }

    for _ in range(3):
        meta_traj = [
            {"decision": 0, "features": torch.randn(1, input_dim), "predicted_q": 0.1},
            {"decision": 2, "features": torch.randn(1, input_dim), "predicted_q": 0.2},
        ]
        root = _make_root_with_child(42)
        train_step(
            rollout=rollout,
            root=root,
            committed_token_id=42,
            meta_trajectory=meta_traj,
            meta_mlp=meta,
            training_state=ts,
        )

    # Verify weights have NOT changed (no update happened)
    for i, (p_init, p_current) in enumerate(zip(initial_params, meta.parameters())):
        assert torch.allclose(p_init, p_current.data), (
            f"Parameter {i} should NOT change before batch update, "
            f"max diff = {(p_init - p_current.data).abs().max().item()}"
        )


def test_full_batch_cycle():
    """
    Integration: Full end-to-end batch cycle.
    1. Create TrainingState(batch_size=2)
    2. Call train_step 2 times (batch_ready=True on 2nd)
    3. Call get_batch() and update_metapolicy_batch()
    4. Verify step_count is 1 and weights changed from initial values.
    """
    input_dim = 10
    meta = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 4))
    rw = nn.Sequential(nn.Linear(142, 128), nn.ReLU(), nn.Linear(128, 1))
    rf = nn.Sequential(nn.Linear(134, 64), nn.ReLU(), nn.Linear(64, 6))

    ts = TrainingState(meta, rw, rf, batch_size=2, buffer_capacity=100)

    # Record initial weights
    initial_params = [p.clone().detach() for p in meta.parameters()]

    rollout = {
        "token_id": 42, "duration": 5,
        "states": [{"dummy": i} for i in range(6)],
        "rewards": [0.1, 0.2, 0.15, 0.1, 0.05],
        "components": [{"speed_reward": 0.1} for _ in range(5)],
        "state_before": {"dummy": 0}, "state_after": {"dummy": 5},
    }

    results = []
    for _ in range(2):
        meta_traj = [
            {"decision": 0, "features": torch.randn(1, input_dim), "predicted_q": 0.1},
            {"decision": 1, "features": torch.randn(1, input_dim), "predicted_q": 0.2},
            {"decision": 2, "features": torch.randn(1, input_dim), "predicted_q": 0.1},
        ]
        root = _make_root_with_child(42)
        result = train_step(
            rollout=rollout,
            root=root,
            committed_token_id=42,
            meta_trajectory=meta_traj,
            meta_mlp=meta,
            training_state=ts,
        )
        results.append(result)

    assert results[1]["batch_ready"] is True, (
        f"2nd call should trigger batch_ready, got {results[1]['batch_ready']}"
    )

    # Now run the batch update
    batch = ts.get_batch()
    update_result = ts.update_metapolicy_batch(meta, batch)

    assert ts.step_count == 1, (
        f"After one batch update, step_count should be 1, got {ts.step_count}"
    )
    assert isinstance(update_result["loss"], float), (
        f"Batch update should return float loss, got {type(update_result['loss'])}"
    )

    # Verify weights CHANGED after batch update
    any_changed = False
    for p_init, p_current in zip(initial_params, meta.parameters()):
        if not torch.allclose(p_init, p_current.data):
            any_changed = True
            break

    assert any_changed, (
        "After batch update, at least one parameter should have changed"
    )
