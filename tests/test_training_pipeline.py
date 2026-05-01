"""
Tests for Phase 4: Module Training Pipelines (PIPE-01, PIPE-05, PIPE-06 shared infrastructure).

Covers:
  - PIPE-01: training_data directories exist for all trainable modules
  - PIPE-05: freeze_module disables gradients on nn.Module and encoder weight dicts
  - PIPE-06: ConvergenceDetector dual criteria (threshold + patience), config loading,
             update_training_status auto-write mechanism
"""

import json
import sys
import tempfile
import shutil
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METACONTROLLER_DIR = PROJECT_ROOT / "metacontroller"
MAIN_MODEL_DIR = PROJECT_ROOT / "main_model"
INTUITION_HEAD_DIR = PROJECT_ROOT / "intuition_head"
REWARD_HEAD_DIR = PROJECT_ROOT / "reward_head"
ACTION_PLANNER_DIR = PROJECT_ROOT / "action_planner"

for d in (
    PROJECT_ROOT,
    METACONTROLLER_DIR,
    MAIN_MODEL_DIR,
    INTUITION_HEAD_DIR,
    REWARD_HEAD_DIR,
    ACTION_PLANNER_DIR,
):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from training_utils import (
    ConvergenceDetector,
    freeze_module,
    load_training_config,
    update_training_status,
)
from main_model import create_encoder_weights, encode_state
from intuition_head import intuition_head
from action_planner import action_planner


# =========================================================================
# PIPE-01: Training data directories exist
# =========================================================================

def test_training_data_directories_exist():
    """PIPE-01: Per-module training_data/ directories exist for all trainable modules."""
    assert (PROJECT_ROOT / "main_model" / "training_data").is_dir(), (
        "main_model/training_data/ directory should exist"
    )
    assert (PROJECT_ROOT / "reward_head" / "training_data").is_dir(), (
        "reward_head/training_data/ directory should exist"
    )
    assert (PROJECT_ROOT / "action_planner" / "training_data").is_dir(), (
        "action_planner/training_data/ directory should exist"
    )


# =========================================================================
# PIPE-06: Convergence detection (dual criteria: threshold + patience)
# =========================================================================

def test_convergence_detector_min_mode():
    """PIPE-06: ConvergenceDetector in min mode converges when MSE <= threshold AND patience met."""
    cd = ConvergenceDetector(threshold=0.1, patience=3, mode="min")

    # Feed decreasing values -- not converged yet (still improving)
    assert not cd.update(0.5)   # improved (0.5 < inf), threshold not met
    assert not cd.update(0.3)   # improved (0.3 < 0.5), threshold not met
    assert not cd.update(0.15)  # improved (0.15 < 0.3), threshold not met

    # First time below threshold -- patience is 0, need 3
    assert not cd.update(0.09)  # improved (0.09 < 0.15), threshold met, patience=0 (just improved)

    # No improvement -- patience accumulates
    assert not cd.update(0.09)  # no improve, patience=1, threshold met
    assert not cd.update(0.09)  # no improve, patience=2, threshold met
    assert cd.update(0.09)      # no improve, patience=3, threshold met -> CONVERGED


def test_convergence_detector_max_mode():
    """PIPE-06: ConvergenceDetector in max mode converges when accuracy >= threshold AND patience met."""
    cd = ConvergenceDetector(threshold=0.6, patience=2, mode="max")

    # Increasing accuracy
    assert not cd.update(0.3)   # improved, threshold not met
    assert not cd.update(0.5)   # improved, threshold not met
    assert not cd.update(0.65)  # improved, threshold met, patience=0

    # Plateau -- patience accumulates
    assert not cd.update(0.65)  # no improve, patience=1
    assert cd.update(0.65)      # no improve, patience=2 -> CONVERGED


def test_convergence_detector_improvement_resets_patience():
    """PIPE-06: An improvement after partial patience resets the patience counter to 0."""
    cd = ConvergenceDetector(threshold=0.1, patience=3, mode="min")

    # Get below threshold with an improvement
    assert not cd.update(0.09)  # improved (below threshold), patience=0

    # Accumulate some patience
    assert not cd.update(0.09)  # no improve, patience=1
    assert not cd.update(0.09)  # no improve, patience=2

    # An improvement resets patience!
    assert not cd.update(0.08)  # improved! (0.08 < 0.09), patience resets to 0

    # Need full patience again
    assert not cd.update(0.08)  # no improve, patience=1
    assert not cd.update(0.08)  # no improve, patience=2
    assert cd.update(0.08)      # no improve, patience=3 -> CONVERGED


def test_convergence_detector_threshold_not_met():
    """PIPE-06: Never converges if metric never meets threshold, even with infinite patience."""
    cd = ConvergenceDetector(threshold=0.05, patience=3, mode="min")

    # Feed 0.1 ten times -- never meets threshold of 0.05
    for _ in range(10):
        assert not cd.update(0.1), (
            "Should never converge when threshold (0.05) is not met (metric=0.1)"
        )


# =========================================================================
# PIPE-05: Module freeze mechanism
# =========================================================================

def test_freeze_module_nn_module():
    """PIPE-05: freeze_module disables requires_grad on all nn.Module parameters."""
    module = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    # Before freeze: all params trainable
    for p in module.parameters():
        assert p.requires_grad, "Parameters should be trainable before freeze"

    freeze_module(module)

    # After freeze: all params frozen
    for p in module.parameters():
        assert not p.requires_grad, "Parameters should not require grad after freeze"

    # Forward pass still works
    x = torch.randn(1, 10)
    out = module(x)
    assert out.shape == (1, 1), f"Expected shape (1, 1), got {out.shape}"


def test_freeze_module_encoder_weights_dict():
    """PIPE-05: freeze_module disables requires_grad on encoder weight dict values."""
    encoder_weights = create_encoder_weights()

    freeze_module(encoder_weights)

    # All nn.Module values in the dict should be frozen
    for key, value in encoder_weights.items():
        if hasattr(value, "parameters"):
            for p in value.parameters():
                assert not p.requires_grad, (
                    f"Encoder weights['{key}'] should have requires_grad=False after freeze"
                )

    # Non-tensor values (embed_dim, num_heads) should still be accessible
    assert isinstance(encoder_weights["embed_dim"], int)
    assert isinstance(encoder_weights["num_heads"], int)

    # Forward pass still works with frozen weights
    raw_state = {
        "near_entities": [],
        "near_vehs": [],
        "near_peds": [],
        "near_objects": [],
        "wp_dist": 10.0,
        "hp": 100.0,
    }
    z_t = encode_state(raw_state, encoder_weights)
    assert z_t.shape == (1, 128), f"Expected shape (1, 128), got {z_t.shape}"


def test_freeze_blocks_gradient_flow():
    """PIPE-05: After freezing, backward pass does not accumulate gradients."""
    module = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    freeze_module(module)

    # Forward pass
    x = torch.randn(1, 10, requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()

    # All module .grad should be None (no gradient accumulated)
    for name, p in module.named_parameters():
        assert p.grad is None, (
            f"Parameter '{name}' should have grad=None after freeze, got {p.grad}"
        )


# =========================================================================
# PIPE-06: Training config loading
# =========================================================================

def test_training_config_loads():
    """PIPE-06: load_training_config returns valid config with expected keys and values."""
    config = load_training_config()

    # Top-level keys
    assert "encoder_intuition" in config
    assert "reward_head" in config
    assert "action_planner" in config
    assert "metacontroller" in config

    # Encoder+intuition convergence settings
    ei = config["encoder_intuition"]["convergence"]
    assert ei["threshold"] == 0.05
    assert ei["patience"] == 10
    assert ei["mode"] == "min"

    # Action planner convergence settings
    ap = config["action_planner"]["convergence"]
    assert ap["threshold"] == 0.6
    assert ap["patience"] == 15
    assert ap["mode"] == "max"


# =========================================================================
# PIPE-07 support (WARNING 2 fix): update_training_status
# =========================================================================

def test_update_training_status():
    """PIPE-07: update_training_status auto-writes stage status changes to training_status.json."""
    # Create a temp copy to avoid modifying the real file
    tmp_dir = tempfile.mkdtemp()
    tmp_status = Path(tmp_dir) / "training_status.json"
    shutil.copy(PROJECT_ROOT / "training_status.json", tmp_status)

    try:
        # Mark encoder_intuition as "training"
        update_training_status("encoder_intuition", "training", status_path=tmp_status)

        with open(tmp_status, encoding="utf-8") as f:
            data = json.load(f)

        stage = data["stages"]["encoder_intuition"]
        assert stage["status"] == "training", (
            f"Expected status 'training', got '{stage['status']}'"
        )
        assert stage["started_at"] is not None, (
            "started_at should be set when status is 'training'"
        )

        # Mark encoder_intuition as "converged"
        update_training_status(
            "encoder_intuition",
            "converged",
            metric=0.04,
            steps=5000,
            checkpoint="main_model/checkpoints/session_test/",
            status_path=tmp_status,
        )

        with open(tmp_status, encoding="utf-8") as f:
            data = json.load(f)

        stage = data["stages"]["encoder_intuition"]
        assert stage["status"] == "converged"
        assert stage["final_metric"] == 0.04
        assert stage["total_steps"] == 5000
        assert stage["checkpoint"] == "main_model/checkpoints/session_test/"
        assert stage["converged_at"] is not None

    finally:
        shutil.rmtree(tmp_dir)
