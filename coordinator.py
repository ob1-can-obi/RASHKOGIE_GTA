"""
Training pipeline coordinator -- stateless CLI per D-08, D-10.

Reads training_status.json and checkpoint files to determine pipeline state.
Enforces strict training order per D-03:
    encoder_intuition -> reward_head -> freeze -> action_planner -> metacontroller

Commands:
    python coordinator.py status   -- Show pipeline status table
    python coordinator.py next     -- What to run next
    python coordinator.py init     -- Initialize new pipeline run
    python coordinator.py freeze <stage>  -- Freeze a converged stage
    python coordinator.py update <stage> <status>  -- Manual status update
"""

import sys
import json
import argparse
import os
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training_utils import load_training_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_FILE = _ROOT / "training_status.json"

STAGE_ORDER = ["encoder_intuition", "reward_head", "action_planner", "metacontroller"]

STAGE_DISPLAY = {
    "encoder_intuition": "Encoder + Intuition",
    "reward_head": "Reward Head",
    "action_planner": "Action Planner",
    "metacontroller": "Metacontroller",
}

STAGE_SCRIPTS = {
    "encoder_intuition": "python main_model/train.py",
    "reward_head": "python reward_head/train.py",
    "action_planner": "python action_planner/train.py",
    "metacontroller": "(uses existing frame_loop.py -- run GTA agent)",
}

STAGE_DEPENDS = {
    "encoder_intuition": [],
    "reward_head": ["encoder_intuition"],
    "action_planner": ["encoder_intuition", "reward_head"],
    "metacontroller": ["encoder_intuition", "reward_head"],
}

FREEZE_AFTER = {
    "encoder_intuition": True,
    "reward_head": True,
    "action_planner": False,
    "metacontroller": False,
}

# Pitfall 5: stale training detection threshold (seconds)
STALE_THRESHOLD_SECONDS = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Status I/O
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


def load_status(status_path=None):
    """
    Load pipeline status from training_status.json.

    Args:
        status_path: Optional path override. Defaults to STATUS_FILE.

    Returns:
        dict: Parsed pipeline status. Returns initial template if file
              doesn't exist or is malformed (T-04-13 mitigation).
    """
    if status_path is None:
        status_path = STATUS_FILE
    else:
        status_path = Path(status_path)

    if not status_path.exists():
        return json.loads(json.dumps(_INITIAL_STATUS_TEMPLATE))

    try:
        with open(status_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # Malformed file -- return template (T-04-13)
        return json.loads(json.dumps(_INITIAL_STATUS_TEMPLATE))

    # Validate expected keys
    if "stages" not in data:
        data["stages"] = {}
    if "frozen_modules" not in data:
        data["frozen_modules"] = []

    return data


def save_status(status, status_path=None):
    """
    Write pipeline status back to training_status.json.

    Args:
        status: Pipeline status dict
        status_path: Optional path override. Defaults to STATUS_FILE.
    """
    if status_path is None:
        status_path = STATUS_FILE
    else:
        status_path = Path(status_path)

    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Stale training detection (Pitfall 5)
# ---------------------------------------------------------------------------

def check_stale_training(status):
    """
    Detect stale "training" status when no checkpoint modification in >5 minutes.

    Per Pitfall 5: if a training script crashes, training_status.json may still
    say "training" with no active process. We check the checkpoint file's mtime
    to detect this condition.

    Args:
        status: Pipeline status dict

    Returns:
        list[str]: Warning strings for any stale stages detected
    """
    warnings = []
    now = datetime.now().timestamp()

    for stage_name in STAGE_ORDER:
        stage = status.get("stages", {}).get(stage_name, {})
        if stage.get("status") != "training":
            continue

        checkpoint = stage.get("checkpoint")
        if not checkpoint:
            # No checkpoint recorded -- check started_at timestamp instead
            started_at = stage.get("started_at")
            if started_at:
                try:
                    start_ts = datetime.fromisoformat(started_at).timestamp()
                    if (now - start_ts) > STALE_THRESHOLD_SECONDS:
                        warnings.append(
                            f"WARNING: {STAGE_DISPLAY.get(stage_name, stage_name)} "
                            f"has been 'training' since {started_at} with no checkpoint. "
                            f"Process may have crashed."
                        )
                except (ValueError, TypeError):
                    pass
            continue

        # Check checkpoint directory/file mtime
        ckpt_path = _ROOT / checkpoint
        if not ckpt_path.exists():
            warnings.append(
                f"WARNING: {STAGE_DISPLAY.get(stage_name, stage_name)} "
                f"checkpoint path '{checkpoint}' does not exist. "
                f"Process may have crashed."
            )
            continue

        # Find the most recently modified .pt file in checkpoint path
        if ckpt_path.is_dir():
            pt_files = list(ckpt_path.glob("*.pt"))
            if pt_files:
                latest_mtime = max(f.stat().st_mtime for f in pt_files)
            else:
                latest_mtime = ckpt_path.stat().st_mtime
        else:
            latest_mtime = ckpt_path.stat().st_mtime

        elapsed = now - latest_mtime
        if elapsed > STALE_THRESHOLD_SECONDS:
            minutes = int(elapsed // 60)
            warnings.append(
                f"WARNING: {STAGE_DISPLAY.get(stage_name, stage_name)} "
                f"has been 'training' but checkpoint not modified for "
                f"{minutes} minutes. Process may have crashed."
            )

    return warnings


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_status(status):
    """
    Print pipeline status table.

    Shows pipeline run ID, per-stage status/metric/checkpoint, stale warnings,
    and frozen modules list.
    """
    run_id = status.get("pipeline_run_id", "none")
    print(f"Pipeline Run: {run_id or 'not initialized'}")
    print()

    # Table header
    header = f"{'Stage':<24} {'Status':<12} {'Metric':<14} {'Checkpoint':<30}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for stage_name in STAGE_ORDER:
        stage = status.get("stages", {}).get(stage_name, {})
        display = STAGE_DISPLAY.get(stage_name, stage_name)
        st = stage.get("status", "pending")

        # Format metric
        metric_val = stage.get("final_metric")
        if metric_val is not None:
            metric_str = f"{metric_val:.4f}" if isinstance(metric_val, float) else str(metric_val)
        else:
            metric_str = "--"

        # Format checkpoint
        ckpt = stage.get("checkpoint")
        if ckpt:
            # Show just the session directory name for brevity
            ckpt_str = Path(ckpt).name if len(str(ckpt)) > 28 else str(ckpt)
        else:
            ckpt_str = "--"

        print(f"{display:<24} {st:<12} {metric_str:<14} {ckpt_str:<30}")

    print()

    # Stale training warnings (Pitfall 5)
    warnings = check_stale_training(status)
    for w in warnings:
        print(w)
    if warnings:
        print()

    # Frozen modules
    frozen = status.get("frozen_modules", [])
    if frozen:
        print(f"Frozen modules: {', '.join(frozen)}")
    else:
        print("Frozen modules: none")


def cmd_next(status):
    """
    Determine and print what to run next based on pipeline state.

    Iterates STAGE_ORDER, finds the first actionable stage, checks
    dependencies, and prints the appropriate command.
    """
    stages = status.get("stages", {})
    frozen = status.get("frozen_modules", [])

    for stage_name in STAGE_ORDER:
        stage = stages.get(stage_name, {})
        st = stage.get("status", "pending")
        display = STAGE_DISPLAY.get(stage_name, stage_name)

        if st == "training":
            # Check for stale status first
            warnings = check_stale_training(status)
            stale = any(stage_name in w.lower().replace(" + ", "_").replace(" ", "_") for w in warnings)
            if stale:
                for w in warnings:
                    print(w)
                print()
                print(f"If {display} crashed, reset it with:")
                print(f"  python coordinator.py update {stage_name} pending")
            else:
                print(f"Currently in progress: {display}")
                print(f"Resume: {STAGE_SCRIPTS[stage_name]} --resume")
            return

        if st == "converged":
            # Check if this stage needs freezing
            if FREEZE_AFTER.get(stage_name, False):
                if stage_name not in _get_frozen_stage_names(frozen):
                    print(f"Next action: Freeze {display}")
                    print(f"  python coordinator.py freeze {stage_name}")
                    return
            continue

        if st == "pending":
            # Check dependencies
            deps = STAGE_DEPENDS.get(stage_name, [])
            deps_met = True
            unmet_deps = []

            for dep in deps:
                dep_stage = stages.get(dep, {})
                dep_status = dep_stage.get("status", "pending")
                if dep_status != "converged":
                    deps_met = False
                    unmet_deps.append(f"{STAGE_DISPLAY.get(dep, dep)} ({dep_status})")
                    continue
                # Check if dep needs to be frozen first
                if FREEZE_AFTER.get(dep, False):
                    if dep not in _get_frozen_stage_names(frozen):
                        deps_met = False
                        unmet_deps.append(f"{STAGE_DISPLAY.get(dep, dep)} (not frozen)")

            if deps_met:
                print(f"Next action: Start {display}")
                print(f"  {STAGE_SCRIPTS[stage_name]}")
                return
            else:
                # This stage is blocked; skip to see if any later stage is runnable
                # (in this strict pipeline, no later stage will be runnable, but
                # we report the blocker on the FIRST blocked stage)
                print(f"Blocked: {display}")
                print(f"  Waiting on: {', '.join(unmet_deps)}")
                return

        if st == "available":
            # metacontroller special status
            print(f"Ready: {display}")
            print(f"  {STAGE_SCRIPTS[stage_name]}")
            return

    # All stages complete
    print("Pipeline complete! All stages have finished.")
    print()
    frozen = status.get("frozen_modules", [])
    if frozen:
        print(f"Frozen modules: {', '.join(frozen)}")


def _get_frozen_stage_names(frozen_modules):
    """
    Map frozen_modules list (module names) back to stage names.

    frozen_modules contains individual module names like "encoder",
    "intuition_head", "token_embed". This maps them back to the stage
    that owns them.
    """
    stages = set()
    encoder_modules = {"encoder", "intuition_head", "token_embed"}
    reward_modules = {"reward_mlp", "rf_predictor"}

    if encoder_modules.intersection(frozen_modules):
        stages.add("encoder_intuition")
    if reward_modules.intersection(frozen_modules):
        stages.add("reward_head")

    return stages


def cmd_freeze(status, stage_name, status_path=None):
    """
    Freeze a converged stage per D-12 (hard freeze).

    Validates that the stage is converged before adding its modules
    to the frozen_modules list in training_status.json.

    Args:
        status: Pipeline status dict
        stage_name: Stage to freeze (must be in STAGE_ORDER)
        status_path: Optional path override for saving
    """
    if stage_name not in STAGE_ORDER:
        print(f"Error: Unknown stage '{stage_name}'. Valid stages: {', '.join(STAGE_ORDER)}")
        return

    stage = status.get("stages", {}).get(stage_name, {})
    st = stage.get("status", "pending")

    if st != "converged":
        print(f"Error: Cannot freeze {STAGE_DISPLAY.get(stage_name, stage_name)} -- "
              f"status is '{st}', must be 'converged'.")
        return

    # Map stage to module names for frozen_modules list
    module_map = {
        "encoder_intuition": ["encoder", "intuition_head", "token_embed"],
        "reward_head": ["reward_mlp", "rf_predictor"],
    }

    modules_to_freeze = module_map.get(stage_name, [])
    if not modules_to_freeze:
        print(f"Error: Stage '{stage_name}' does not have freezable modules.")
        return

    frozen = status.get("frozen_modules", [])
    added = []
    for m in modules_to_freeze:
        if m not in frozen:
            frozen.append(m)
            added.append(m)

    status["frozen_modules"] = frozen

    save_status(status, status_path)

    display = STAGE_DISPLAY.get(stage_name, stage_name)
    if added:
        print(f"Frozen {display}: {', '.join(added)}")
    else:
        print(f"{display} modules already frozen.")
    print(f"Frozen modules: {', '.join(frozen)}")


def cmd_init(status, status_path=None):
    """
    Initialize a new pipeline run.

    Creates a fresh pipeline with a timestamped run ID, resets all stages
    to "pending", and clears frozen_modules.

    Args:
        status: Pipeline status dict (will be replaced)
        status_path: Optional path override for saving
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    new_status = json.loads(json.dumps(_INITIAL_STATUS_TEMPLATE))
    new_status["pipeline_run_id"] = run_id

    save_status(new_status, status_path)

    print(f"Initialized new pipeline run: {run_id}")
    print("All stages reset to 'pending'. Frozen modules cleared.")


def cmd_update(status, stage_name, new_status, metric=None, steps=None,
               checkpoint=None, status_path=None):
    """
    Manually update a stage's status.

    Args:
        status: Pipeline status dict
        stage_name: Stage to update
        new_status: New status string
        metric: Optional metric value
        steps: Optional step count
        checkpoint: Optional checkpoint path
        status_path: Optional path override for saving
    """
    if stage_name not in STAGE_ORDER:
        print(f"Error: Unknown stage '{stage_name}'. Valid stages: {', '.join(STAGE_ORDER)}")
        return

    stages = status.get("stages", {})
    if stage_name not in stages:
        stages[stage_name] = {}

    stage = stages[stage_name]
    stage["status"] = new_status

    if new_status == "training":
        stage["started_at"] = datetime.now().isoformat()
    elif new_status == "converged":
        stage["converged_at"] = datetime.now().isoformat()
        if metric is not None:
            try:
                stage["final_metric"] = float(metric)
            except (ValueError, TypeError):
                stage["final_metric"] = metric
        if steps is not None:
            stage["total_steps"] = steps
        if checkpoint is not None:
            stage["checkpoint"] = checkpoint

    save_status(status, status_path)

    display = STAGE_DISPLAY.get(stage_name, stage_name)
    print(f"Updated {display}: status={new_status}")
    if metric:
        print(f"  metric={metric}")
    if steps:
        print(f"  steps={steps}")
    if checkpoint:
        print(f"  checkpoint={checkpoint}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training pipeline coordinator (stateless CLI per D-10)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    subparsers.add_parser("status", help="Show pipeline status table")

    # next
    subparsers.add_parser("next", help="What to run next")

    # init
    subparsers.add_parser("init", help="Initialize new pipeline run")

    # freeze
    freeze_parser = subparsers.add_parser("freeze", help="Freeze a converged stage")
    freeze_parser.add_argument(
        "stage", choices=STAGE_ORDER[:2],
        help="Stage to freeze (encoder_intuition or reward_head)"
    )

    # update
    update_parser = subparsers.add_parser("update", help="Update stage status manually")
    update_parser.add_argument("stage", choices=STAGE_ORDER, help="Stage name")
    update_parser.add_argument(
        "new_status",
        choices=["pending", "training", "converged", "available"],
        help="New status"
    )
    update_parser.add_argument("--metric", default=None, help="Metric value")
    update_parser.add_argument("--steps", type=int, default=None, help="Step count")
    update_parser.add_argument("--checkpoint", default=None, help="Checkpoint path")

    args = parser.parse_args()
    status = load_status()

    if args.command == "status":
        cmd_status(status)
    elif args.command == "next":
        cmd_next(status)
    elif args.command == "init":
        cmd_init(status)
    elif args.command == "freeze":
        cmd_freeze(status, args.stage)
    elif args.command == "update":
        cmd_update(
            status, args.stage, args.new_status,
            args.metric, args.steps, args.checkpoint,
        )
    else:
        parser.print_help()
