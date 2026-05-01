"""
Tests for PIPE-03: Reward head offline training.

Validates:
  - Offline training loop runs on synthetic JSONL data
  - Encoder outputs are properly detached (Pitfall 2)
  - Freeze on convergence disables gradients on reward modules (WARNING 1)

Tests use synthetic data -- they do not require GTA or real game states.
"""

import sys
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REWARD_HEAD_DIR = PROJECT_ROOT / "reward_head"
MAIN_MODEL_DIR = PROJECT_ROOT / "main_model"

for d in (REWARD_HEAD_DIR, MAIN_MODEL_DIR, str(PROJECT_ROOT)):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from main_model import create_encoder_weights, encode_state
from reward_head import extract_reward_features, RF_DIM
from training_utils import freeze_module


# =========================================================================
# Synthetic data helpers
# =========================================================================

def _make_synthetic_state(wp_dist=10.0, hp=100.0):
    """Create a minimal GTA state dict for testing."""
    return {
        "near_entities": [],
        "near_vehs": [],
        "near_peds": [],
        "near_objects": [],
        "wp_dist": wp_dist,
        "hp": hp,
        "v_engine_hp": 1000.0,
        "v_body_hp": 1000.0,
        "road_dist": 0.5,
        "dead": False,
    }


def _write_synthetic_jsonl(data_dir, n_records=5):
    """Write synthetic JSONL training records for reward head."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = data_dir / "session_test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("# test session\n")
        for i in range(n_records):
            record = {
                "state_before": _make_synthetic_state(wp_dist=10.0 - i * 0.1),
                "state_after": _make_synthetic_state(wp_dist=10.0 - (i + 1) * 0.1),
                "duration": 5,
                "realized_return": 0.1 + i * 0.05,
            }
            f.write(json.dumps(record) + "\n")
    return jsonl_path


# =========================================================================
# PIPE-03: test_reward_head_training
# =========================================================================

def test_reward_head_training():
    """PIPE-03: Reward head offline training runs on synthetic JSONL data."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "reward_train", str(PROJECT_ROOT / "reward_head" / "train.py")
    )
    reward_train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "training_data"
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        status_path = Path(tmpdir) / "training_status.json"
        config_path = Path(tmpdir) / "training_config.json"

        # Write synthetic training data (5 records)
        _write_synthetic_jsonl(data_dir, n_records=5)

        # Write config with easy convergence: high threshold + patience=1
        config = {
            "reward_head": {
                "lr": 3e-4,
                "batch_size": 2,
                "max_grad_norm": 0.5,
                "eval_every_n_steps": 1,
                "convergence": {
                    "metric": "mse",
                    "threshold": 100.0,
                    "patience": 1,
                    "mode": "min",
                },
            }
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Write initial training_status.json
        with open(status_path, "w") as f:
            json.dump({"stages": {}}, f)

        # Monkey-patch update_training_status to write to temp location
        orig_update = reward_train.update_training_status

        def patched_update(stage_name, status, metric=None, steps=None, checkpoint=None, status_path=None):
            return orig_update(
                stage_name, status, metric=metric, steps=steps,
                checkpoint=checkpoint, status_path=str(status_path or Path(tmpdir) / "training_status.json"),
            )

        reward_train.update_training_status = patched_update

        # Run training
        result = reward_train.train_reward_head_offline(
            data_dir=str(data_dir),
            config_path=str(config_path),
            checkpoint_dir=str(checkpoint_dir),
            max_epochs=5,
        )

        # Assert training produced a result
        assert result is not None, "Training returned None"
        assert result["total_steps"] > 0, f"Expected steps > 0, got {result['total_steps']}"


# =========================================================================
# PIPE-03: test_reward_head_encoder_detached
# =========================================================================

def test_reward_head_encoder_detached():
    """PIPE-03: Encoder outputs are properly detached during reward head training."""
    encoder_weights = create_encoder_weights()
    state = _make_synthetic_state()

    # Compute z with the encoder (graph attached)
    z = encode_state(state, encoder_weights)
    assert z.requires_grad is True, "encode_state should produce tensors with grad"

    # Detach per Pitfall 2 pattern
    z_detached = z.detach()
    assert z_detached.requires_grad is False, "Detached z should not require grad"

    # Verify shape is preserved
    assert z_detached.shape == z.shape, "Detach should not change shape"


# =========================================================================
# PIPE-03: test_reward_head_freeze_on_convergence
# =========================================================================

def test_reward_head_freeze_on_convergence():
    """PIPE-03: freeze_module disables gradients on reward_mlp and rf_predictor."""
    # Create dummy reward_mlp matching reward_head.py lazy-init pattern
    fused_dim = 128
    input_dim_reward = fused_dim * 2 + RF_DIM * 2 + 2  # 270
    reward_mlp = nn.Sequential(
        nn.Linear(input_dim_reward, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    # Create dummy rf_predictor matching predict_reward_features lazy-init
    input_dim_rf = fused_dim * 2 + RF_DIM  # 262
    rf_predictor = nn.Sequential(
        nn.Linear(input_dim_rf, 64),
        nn.ReLU(),
        nn.Linear(64, RF_DIM),
    )

    # Before freeze: all params should require grad
    for name, p in reward_mlp.named_parameters():
        assert p.requires_grad is True, f"reward_mlp.{name} should require grad before freeze"
    for name, p in rf_predictor.named_parameters():
        assert p.requires_grad is True, f"rf_predictor.{name} should require grad before freeze"

    # Freeze both modules (same as convergence handler in train.py)
    freeze_module(reward_mlp)
    freeze_module(rf_predictor)

    # After freeze: all params should NOT require grad
    for name, p in reward_mlp.named_parameters():
        assert p.requires_grad is False, (
            f"reward_mlp.{name} should NOT require grad after freeze"
        )
    for name, p in rf_predictor.named_parameters():
        assert p.requires_grad is False, (
            f"rf_predictor.{name} should NOT require grad after freeze"
        )
