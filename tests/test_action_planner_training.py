"""
Tests for PIPE-04: Action planner imitation learning.

Each test validates one aspect of the action planner imitation learning pipeline.
Tests use synthetic data -- they do not require GTA or real game states.
"""

import sys
import os
import json
import tempfile
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_MODEL_DIR = PROJECT_ROOT / "main_model"
INTUITION_HEAD_DIR = PROJECT_ROOT / "intuition_head"
ACTION_PLANNER_DIR = PROJECT_ROOT / "action_planner"

for d in (MAIN_MODEL_DIR, INTUITION_HEAD_DIR, ACTION_PLANNER_DIR, str(PROJECT_ROOT)):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from main_model import create_encoder_weights, encode_state
from intuition_head import intuition_head
from action_planner import action_planner

# Import train module via importlib (action_planner/ is not a Python package)
_spec = importlib.util.spec_from_file_location(
    "ap_train", str(ACTION_PLANNER_DIR / "train.py")
)
_ap_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ap_train)

train_action_planner_imitation = _ap_train.train_action_planner_imitation
preprocess_data = _ap_train.preprocess_data


# =========================================================================
# Helper: create synthetic demonstration JSONL data
# =========================================================================

def _create_synthetic_data(data_dir, num_records=10, vocab_size=100):
    """
    Write synthetic JSONL demonstration data for testing.

    Each record contains a minimal GTA state dict and a random token_id
    in range [0, vocab_size).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(num_records):
        record = {
            "state": {
                "near_entities": [],
                "near_vehs": [],
                "near_peds": [],
                "near_objects": [],
                "wp_dist": 10.0 - i * 0.5,
                "hp": 100.0,
            },
            "token_id": i % vocab_size,
            "session_ts": "20260501_143022",
            "frame_idx": i,
        }
        records.append(record)

    jsonl_path = data_dir / "session_20260501_143022.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("# session session_20260501_143022  fps=20.0\n")
        for record in records:
            f.write(json.dumps(record) + "\n")

    return jsonl_path


def _create_easy_config(config_dir):
    """
    Write a training_config.json with easy convergence settings for fast tests.

    Uses threshold=0.0 and patience=1 so any accuracy triggers convergence.
    """
    config = {
        "encoder_intuition": {
            "lr": 3e-4,
            "batch_size": 8,
            "max_grad_norm": 0.5,
            "eval_every_n_steps": 100,
            "convergence": {"metric": "mse", "threshold": 0.05, "patience": 10, "mode": "min"},
        },
        "reward_head": {
            "lr": 3e-4,
            "batch_size": 8,
            "max_grad_norm": 0.5,
            "eval_every_n_steps": 100,
            "convergence": {"metric": "mse", "threshold": 0.1, "patience": 10, "mode": "min"},
        },
        "action_planner": {
            "lr": 3e-4,
            "batch_size": 4,
            "max_grad_norm": 0.5,
            "eval_every_n_steps": 1,
            "convergence": {"metric": "top1_accuracy", "threshold": 0.0, "patience": 1, "mode": "max"},
        },
        "metacontroller": {
            "note": "Uses existing TrainingState from Phase 2."
        },
    }
    config_path = Path(config_dir) / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return str(config_path)


# =========================================================================
# PIPE-04: test_action_planner_imitation -- full training loop
# =========================================================================

def test_action_planner_imitation():
    """PIPE-04: Action planner imitation learning trains on synthetic data and converges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "training_data")
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        config_path = _create_easy_config(tmpdir)

        # Create synthetic demo data with vocab_size=100 for faster test
        _create_synthetic_data(data_dir, num_records=10, vocab_size=100)

        # Also write training_status.json so update_training_status works
        status_path = os.path.join(tmpdir, "training_status.json")
        with open(status_path, "w") as f:
            json.dump({"stages": {}}, f)

        # Monkey-patch status path for this test
        import training_utils
        original_root = training_utils._PROJECT_ROOT
        training_utils._PROJECT_ROOT = Path(tmpdir)

        try:
            result = train_action_planner_imitation(
                data_dir=data_dir,
                config_path=config_path,
                checkpoint_dir=ckpt_dir,
                max_epochs=2,
                vocab_size=100,
            )
        finally:
            training_utils._PROJECT_ROOT = original_root

        assert result is not None, "Result should not be None"
        assert result["total_steps"] > 0, f"Expected total_steps > 0, got {result['total_steps']}"
        assert "final_accuracy" in result, "Result must contain 'final_accuracy' key"


# =========================================================================
# PIPE-04: test_action_planner_cross_entropy_loss -- loss correctness
# =========================================================================

def test_action_planner_cross_entropy_loss():
    """PIPE-04: Cross-entropy loss produces gradients that flow to planner_mlp."""
    vocab_size = 10
    z_t = torch.randn(1, 128)
    z_next = torch.randn(1, 128)

    # Create planner_mlp via action_planner forward pass
    result = action_planner(z_t, z_next, vocab_size=vocab_size)
    planner_mlp = result["planner_mlp"]
    logits = result["logits"]  # [1, vocab_size]

    # Compute cross-entropy loss against a target token
    target_token_id = 3
    target = torch.tensor([target_token_id], dtype=torch.long)
    loss = F.cross_entropy(logits, target)

    # Loss must be a scalar tensor with gradient
    assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
    assert loss.requires_grad is True, "Loss should require gradient (flows to planner)"

    # Verify gradients actually flow to planner_mlp parameters
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in planner_mlp.parameters()
    )
    assert has_grad, "Gradients must flow to planner_mlp parameters"


# =========================================================================
# PIPE-04: test_action_planner_filters_null_token_id -- Pitfall 4
# =========================================================================

def test_action_planner_filters_null_token_id():
    """PIPE-04 (Pitfall 4): preprocess_data filters records where token_id is None."""
    records = [
        {"state": {}, "token_id": 5, "session_ts": "test", "frame_idx": 0},
        {"state": {}, "token_id": None, "session_ts": "test", "frame_idx": 1},
        {"state": {}, "token_id": 12, "session_ts": "test", "frame_idx": 2},
        {"state": {}, "token_id": None, "session_ts": "test", "frame_idx": 3},
        {"state": {}, "token_id": 0, "session_ts": "test", "frame_idx": 4},
    ]

    filtered = preprocess_data(records)

    assert len(filtered) == 3, f"Expected 3 valid records, got {len(filtered)}"

    # All remaining records must have valid (non-None) token_id
    for r in filtered:
        assert r["token_id"] is not None, f"Found record with token_id=None after filtering"

    # Verify the correct records survived
    token_ids = [r["token_id"] for r in filtered]
    assert token_ids == [5, 12, 0], f"Expected [5, 12, 0], got {token_ids}"
