"""
Shared training infrastructure for Phase 4: Module Training Pipelines.

Exports:
    ConvergenceDetector  - Dual criteria convergence detection (threshold + patience)
    freeze_module        - Hard freeze for nn.Module and encoder weight dicts
    load_training_config - Load training_config.json from project root
    update_training_status - Atomically update training_status.json per-stage
"""

import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# ConvergenceDetector (D-11: dual criteria -- threshold + patience)
# ---------------------------------------------------------------------------

class ConvergenceDetector:
    """
    Detects convergence using dual criteria:
      1. Metric must meet a quality threshold (below for MSE, above for accuracy)
      2. No improvement for `patience` consecutive evaluations

    Both criteria must be satisfied simultaneously for convergence.

    Args:
        threshold: Quality threshold (MSE <= threshold or accuracy >= threshold)
        patience:  Number of consecutive no-improvement evaluations required
        mode:      "min" for metrics where lower is better (MSE),
                   "max" for metrics where higher is better (accuracy)
    """

    def __init__(self, threshold, patience, mode="min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.threshold = threshold
        self.patience = patience
        self.mode = mode
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.no_improve_count = 0

    def update(self, metric):
        """
        Feed a new metric value. Returns True when converged.

        Convergence requires BOTH:
          - threshold_met: metric <= threshold (min) or metric >= threshold (max)
          - no_improve_count >= patience

        Args:
            metric: Current evaluation metric value

        Returns:
            bool: True if converged, False otherwise
        """
        if self.mode == "min":
            threshold_met = metric <= self.threshold
            improved = metric < self.best_metric
        else:
            threshold_met = metric >= self.threshold
            improved = metric > self.best_metric

        if improved:
            self.best_metric = metric
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        return threshold_met and self.no_improve_count >= self.patience


# ---------------------------------------------------------------------------
# freeze_module (D-12: hard freeze)
# ---------------------------------------------------------------------------

def freeze_module(module):
    """
    Permanently disable gradients for all parameters in a module.

    Handles two patterns used in this project:
      1. nn.Module (intuition_mlp, reward_mlp, etc.) -- calls requires_grad_(False)
      2. dict (encoder_weights) -- iterates values, freezes any that have .parameters()
         (skips non-tensor values like "embed_dim" and "num_heads" which are int/float)

    After freezing, forward passes still work but no gradients accumulate.

    Args:
        module: Either an nn.Module or a dict of encoder weights
    """
    if isinstance(module, dict):
        # Handle stateless weight pattern (encoder_weights dict)
        for key, value in module.items():
            if hasattr(value, "parameters"):
                value.requires_grad_(False)
    elif hasattr(module, "parameters"):
        # Handle nn.Module (intuition_mlp, reward_mlp, MetaMLP, etc.)
        module.requires_grad_(False)


# ---------------------------------------------------------------------------
# load_training_config (D-13: configurable thresholds)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent


def load_training_config(config_path=None):
    """
    Load training_config.json from the project root.

    Args:
        config_path: Optional path override. Defaults to
                     <project_root>/training_config.json

    Returns:
        dict: Parsed training configuration

    Raises:
        FileNotFoundError: If the config file does not exist
        json.JSONDecodeError: If the config file contains invalid JSON
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "training_config.json"
    else:
        config_path = Path(config_path)

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Malformed training_config.json: {e.msg}", e.doc, e.pos
        )

    return config


# ---------------------------------------------------------------------------
# update_training_status (D-10: pipeline state tracking, fixes WARNING 2)
# ---------------------------------------------------------------------------

_INITIAL_STATUS_TEMPLATE = {
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
        "reward_head": {
            "status": "pending",
            "started_at": None,
            "converged_at": None,
            "final_metric": None,
            "total_steps": None,
            "checkpoint": None,
        },
        "action_planner": {
            "status": "pending",
            "started_at": None,
            "converged_at": None,
            "final_metric": None,
            "total_steps": None,
            "checkpoint": None,
        },
        "metacontroller": {
            "status": "pending",
            "depends_on_frozen": ["encoder_intuition", "reward_head"],
        },
    },
    "frozen_modules": [],
}


def update_training_status(
    stage_name, status, metric=None, steps=None, checkpoint=None, status_path=None
):
    """
    Atomically update a stage's status in training_status.json.

    Read-modify-write pattern: reads current state, updates the specified
    stage, writes back. Each train.py calls this on start ("training")
    and on convergence ("converged") so the coordinator always sees
    up-to-date pipeline state.

    Args:
        stage_name:  One of "encoder_intuition", "reward_head",
                     "action_planner", "metacontroller"
        status:      New status string ("pending", "training", "converged")
        metric:      Final metric value (used when status == "converged")
        steps:       Total training steps (used when status == "converged")
        checkpoint:  Checkpoint path string (used when status == "converged")
        status_path: Optional path override. Defaults to
                     <project_root>/training_status.json
    """
    if status_path is None:
        status_path = _PROJECT_ROOT / "training_status.json"
    else:
        status_path = Path(status_path)

    # Read existing or create from template (T-04-16: validate JSON)
    if status_path.exists():
        try:
            with open(status_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Malformed file -- reset to template
            data = json.loads(json.dumps(_INITIAL_STATUS_TEMPLATE))
    else:
        data = json.loads(json.dumps(_INITIAL_STATUS_TEMPLATE))

    # Validate expected structure
    if "stages" not in data:
        data["stages"] = {}

    if stage_name not in data["stages"]:
        data["stages"][stage_name] = {}

    stage = data["stages"][stage_name]

    # Update status
    stage["status"] = status

    if status == "training":
        stage["started_at"] = datetime.now().isoformat()
    elif status == "converged":
        stage["converged_at"] = datetime.now().isoformat()
        if metric is not None:
            stage["final_metric"] = metric
        if steps is not None:
            stage["total_steps"] = steps
        if checkpoint is not None:
            stage["checkpoint"] = checkpoint

    # Write back atomically (T-04-16)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
