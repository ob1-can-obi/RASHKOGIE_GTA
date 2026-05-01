"""
Tests for PIPE-02: Encoder + intuition head joint training.

Validates:
  - train_encoder_intuition runs on synthetic data and produces a result
  - encode_state outputs have live gradients (Pitfall 1 avoided)
  - training_status.json is auto-updated during training (WARNING 2 fix)

Separate file from test_training_pipeline.py to avoid parallel write
conflicts with Plans 03 and 04.
"""

import json
import sys
import tempfile
import shutil
import importlib.util
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_MODEL_DIR = PROJECT_ROOT / "main_model"
INTUITION_HEAD_DIR = PROJECT_ROOT / "intuition_head"

for d in (PROJECT_ROOT, MAIN_MODEL_DIR, INTUITION_HEAD_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from main_model import create_encoder_weights, encode_state

# main_model/ is not a Python package (no __init__.py), so use importlib
# to import train.py directly from the directory path.
_train_spec = importlib.util.spec_from_file_location(
    "main_model_train", str(MAIN_MODEL_DIR / "train.py")
)
_train_mod = importlib.util.module_from_spec(_train_spec)
_train_spec.loader.exec_module(_train_mod)
load_data = _train_mod.load_data
train_encoder_intuition = _train_mod.train_encoder_intuition


# =========================================================================
# PIPE-02: Joint encoder+intuition training loop
# =========================================================================


def test_encoder_intuition_training():
    """PIPE-02: Joint training runs on synthetic JSONL data and produces a result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "training_data"
        data_dir.mkdir()
        ckpt_dir = tmpdir / "checkpoints"
        ckpt_dir.mkdir()
        config_path = tmpdir / "training_config.json"

        # Write synthetic JSONL training data (5 records)
        jsonl_path = data_dir / "session_20260501_000000.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(5):
                record = {
                    "state_t": {
                        "near_entities": [],
                        "near_vehs": [],
                        "near_peds": [],
                        "near_objects": [],
                        "wp_dist": 10.0 + i,
                        "hp": 100.0,
                        "v_engine_hp": 1000.0,
                        "v_body_hp": 1000.0,
                        "road_dist": 0.5,
                        "dead": False,
                    },
                    "state_t1": {
                        "near_entities": [],
                        "near_vehs": [],
                        "near_peds": [],
                        "near_objects": [],
                        "wp_dist": 9.5 + i,
                        "hp": 100.0,
                        "v_engine_hp": 1000.0,
                        "v_body_hp": 1000.0,
                        "road_dist": 0.5,
                        "dead": False,
                    },
                    "token_id": 0,
                    "session_ts": "20260501_000000",
                    "frame_idx": i,
                }
                f.write(json.dumps(record) + "\n")

        # Write training config with LOW thresholds for fast test convergence
        # threshold=100.0 means any MSE < 100 passes the threshold check
        # patience=1 means just 1 non-improving eval triggers convergence
        config = {
            "encoder_intuition": {
                "lr": 3e-4,
                "batch_size": 5,
                "max_grad_norm": 0.5,
                "eval_every_n_steps": 1,
                "convergence": {
                    "metric": "mse",
                    "threshold": 100.0,
                    "patience": 1,
                    "mode": "min",
                },
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Run training
        result = train_encoder_intuition(
            data_dir=str(data_dir),
            config_path=str(config_path),
            checkpoint_dir=str(ckpt_dir),
            max_epochs=2,
            vocab_size=874,
        )

        # Validate result
        assert result is not None, "Training should return a result dict"
        assert result["total_steps"] > 0, "Should complete at least one training step"

        # Check checkpoint files were created
        sessions = list(ckpt_dir.glob("session_*"))
        assert len(sessions) > 0, "Should create at least one checkpoint session"

        session_dir = sessions[0]
        assert (session_dir / "encoder_weights.pt").exists(), "Missing encoder_weights.pt"
        assert (session_dir / "intuition_mlp.pt").exists(), "Missing intuition_mlp.pt"
        assert (session_dir / "token_embed.pt").exists(), "Missing token_embed.pt"
        assert (session_dir / "optimizer.pt").exists(), "Missing optimizer.pt"


# =========================================================================
# PIPE-02: Encoder outputs retain gradients (Pitfall 1 correctness)
# =========================================================================


def test_encoder_intuition_no_detach():
    """PIPE-02: encode_state output has requires_grad=True with trainable weights."""
    encoder_weights = create_encoder_weights()

    # Synthetic state with minimal fields
    state = {
        "near_entities": [],
        "near_vehs": [],
        "near_peds": [],
        "near_objects": [],
        "wp_dist": 10.0,
        "hp": 100.0,
        "v_engine_hp": 1000.0,
        "v_body_hp": 1000.0,
        "road_dist": 0.5,
        "dead": False,
    }

    z_t = encode_state(state, encoder_weights)

    # z_t must have live gradients (connected to encoder weight graph)
    assert z_t.requires_grad is True, (
        "encode_state output should have requires_grad=True when encoder weights "
        "are trainable. If False, gradients cannot flow to encoder (Pitfall 1)."
    )

    # Verify gradient actually flows through to encoder params
    loss = z_t.sum()
    loss.backward()

    # At least some encoder parameters should have gradients
    has_grad = False
    for key, val in encoder_weights.items():
        if hasattr(val, "parameters"):
            for p in val.parameters():
                if p.grad is not None:
                    has_grad = True
                    break
        if has_grad:
            break

    assert has_grad, (
        "No encoder parameters received gradients. The encoder is not being "
        "trained through the autograd graph."
    )


# =========================================================================
# PIPE-02: Auto-status update (WARNING 2 fix)
# =========================================================================


def test_encoder_intuition_auto_status_update():
    """PIPE-02: Training auto-updates training_status.json on start and convergence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "training_data"
        data_dir.mkdir()
        ckpt_dir = tmpdir / "checkpoints"
        ckpt_dir.mkdir()
        config_path = tmpdir / "training_config.json"
        status_path = tmpdir / "training_status.json"

        # Write initial training_status.json
        initial_status = {
            "pipeline_run_id": None,
            "stages": {
                "encoder_intuition": {
                    "status": "pending",
                    "started_at": None,
                    "converged_at": None,
                    "final_metric": None,
                    "total_steps": None,
                    "checkpoint": None,
                },
            },
            "frozen_modules": [],
        }
        with open(status_path, "w") as f:
            json.dump(initial_status, f)

        # Write synthetic JSONL training data
        jsonl_path = data_dir / "session_20260501_000000.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(3):
                record = {
                    "state_t": {
                        "near_entities": [],
                        "near_vehs": [],
                        "near_peds": [],
                        "near_objects": [],
                        "wp_dist": 10.0 + i,
                        "hp": 100.0,
                    },
                    "state_t1": {
                        "near_entities": [],
                        "near_vehs": [],
                        "near_peds": [],
                        "near_objects": [],
                        "wp_dist": 9.5 + i,
                        "hp": 100.0,
                    },
                    "token_id": 0,
                    "session_ts": "20260501_000000",
                    "frame_idx": i,
                }
                f.write(json.dumps(record) + "\n")

        # Write training config with fast convergence
        config = {
            "encoder_intuition": {
                "lr": 3e-4,
                "batch_size": 3,
                "max_grad_norm": 0.5,
                "eval_every_n_steps": 1,
                "convergence": {
                    "metric": "mse",
                    "threshold": 100.0,
                    "patience": 1,
                    "mode": "min",
                },
            },
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Monkey-patch update_training_status to write to our temp status file
        import training_utils

        original_fn = training_utils.update_training_status

        def patched_update(stage_name, status, metric=None, steps=None, checkpoint=None, status_path_arg=None):
            return original_fn(
                stage_name, status, metric=metric, steps=steps,
                checkpoint=checkpoint, status_path=str(status_path),
            )

        # Patch the function in the train module's namespace (loaded via importlib)
        original_train_fn = _train_mod.update_training_status
        _train_mod.update_training_status = patched_update

        try:
            result = train_encoder_intuition(
                data_dir=str(data_dir),
                config_path=str(config_path),
                checkpoint_dir=str(ckpt_dir),
                max_epochs=2,
                vocab_size=874,
            )

            # Read the status file
            with open(status_path) as f:
                data = json.load(f)

            stage = data["stages"]["encoder_intuition"]

            # If converged, should show "converged" status with metric
            if result.get("converged"):
                assert stage["status"] == "converged", (
                    f"Expected status 'converged', got '{stage['status']}'"
                )
                assert stage["converged_at"] is not None, "converged_at should be set"
                assert stage["final_metric"] is not None, "final_metric should be set"
                assert stage["total_steps"] is not None, "total_steps should be set"
            else:
                # At minimum, status should have been updated to "training"
                assert stage["status"] in ("training", "converged"), (
                    f"Expected status 'training' or 'converged', got '{stage['status']}'"
                )
        finally:
            # Restore original function
            _train_mod.update_training_status = original_train_fn


# =========================================================================
# PIPE-02: JSONL data loading
# =========================================================================


def test_load_data_handles_malformed_lines():
    """PIPE-02: load_data skips malformed JSON lines and comments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        jsonl_path = tmpdir / "session_test.jsonl"

        with open(jsonl_path, "w") as f:
            f.write("# This is a comment\n")
            f.write('{"state_t": {}, "state_t1": {}, "token_id": 0}\n')
            f.write("this is not valid json\n")
            f.write("\n")  # empty line
            f.write('{"state_t": {}, "state_t1": {}, "token_id": 1}\n')

        records = load_data(tmpdir)

        assert len(records) == 2, f"Expected 2 valid records, got {len(records)}"
        assert records[0]["token_id"] == 0
        assert records[1]["token_id"] == 1
