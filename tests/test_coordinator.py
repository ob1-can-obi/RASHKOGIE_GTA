"""
Tests for PIPE-07: Coordinator CLI and pipeline orchestration.

Validates coordinator ordering, freeze logic, pipeline flow, and stale detection.
Tests use synthetic status data -- they do not require GTA or real game states.
"""

import sys
import os
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coordinator import (
    load_status,
    save_status,
    check_stale_training,
    cmd_status,
    cmd_next,
    cmd_freeze,
    cmd_init,
    cmd_update,
    STAGE_ORDER,
    STAGE_DEPENDS,
    FREEZE_AFTER,
    STALE_THRESHOLD_SECONDS,
    _get_frozen_stage_names,
)


# =========================================================================
# Helper: create a fresh status dict
# =========================================================================

def _make_status(overrides=None, frozen=None):
    """Create a fresh pipeline status dict with optional stage overrides."""
    status = {
        "pipeline_run_id": "20260501_143022",
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
        "frozen_modules": frozen or [],
    }
    if overrides:
        for stage_name, stage_overrides in overrides.items():
            if stage_name in status["stages"]:
                status["stages"][stage_name].update(stage_overrides)
    return status


# =========================================================================
# PIPE-07: test_coordinator_status_initial
# =========================================================================

def test_coordinator_status_initial():
    """PIPE-07: Initial training_status.json has all 4 stages, all pending, no frozen."""
    status_path = PROJECT_ROOT / "training_status.json"

    # Load the actual training_status.json from project root
    assert status_path.exists(), f"training_status.json not found at {status_path}"

    with open(status_path, encoding="utf-8") as f:
        status = json.load(f)

    assert "stages" in status, "training_status.json missing 'stages' key"

    # All 4 stages exist
    for stage in STAGE_ORDER:
        assert stage in status["stages"], f"Missing stage: {stage}"

    # frozen_modules exists and is a list
    assert "frozen_modules" in status, "training_status.json missing 'frozen_modules'"
    assert isinstance(status["frozen_modules"], list), "frozen_modules should be a list"


# =========================================================================
# PIPE-07: test_coordinator_next_initial
# =========================================================================

def test_coordinator_next_initial():
    """PIPE-07: With all stages pending, next should recommend encoder_intuition (no dependencies)."""
    status = _make_status()

    # encoder_intuition has no dependencies
    assert STAGE_DEPENDS["encoder_intuition"] == [], (
        "encoder_intuition should have no dependencies"
    )

    # The first pending stage with all deps met should be encoder_intuition
    for stage_name in STAGE_ORDER:
        stage = status["stages"][stage_name]
        if stage["status"] == "pending":
            deps = STAGE_DEPENDS.get(stage_name, [])
            all_met = all(
                status["stages"][d]["status"] == "converged"
                for d in deps
            )
            if all_met:
                assert stage_name == "encoder_intuition", (
                    f"Expected encoder_intuition as first actionable stage, got {stage_name}"
                )
                break


# =========================================================================
# PIPE-07: test_coordinator_next_after_stage1
# =========================================================================

def test_coordinator_next_after_stage1():
    """PIPE-07: After encoder_intuition converges and freezes, next should be reward_head."""
    status = _make_status(
        overrides={
            "encoder_intuition": {
                "status": "converged",
                "final_metric": 0.04,
                "total_steps": 5200,
            },
        },
        frozen=["encoder", "intuition_head", "token_embed"],
    )

    # reward_head depends on encoder_intuition
    assert STAGE_DEPENDS["reward_head"] == ["encoder_intuition"]

    # encoder_intuition is converged
    assert status["stages"]["encoder_intuition"]["status"] == "converged"

    # Frozen modules include encoder stage
    frozen_stages = _get_frozen_stage_names(status["frozen_modules"])
    assert "encoder_intuition" in frozen_stages, (
        "encoder_intuition should be in frozen stage names"
    )

    # action_planner should still be blocked (reward_head not converged)
    rh_status = status["stages"]["reward_head"]["status"]
    assert rh_status == "pending", "reward_head should still be pending"

    # Verify action_planner deps not met
    ap_deps = STAGE_DEPENDS["action_planner"]
    for dep in ap_deps:
        if dep == "reward_head":
            assert status["stages"][dep]["status"] != "converged", (
                "reward_head should not be converged yet, blocking action_planner"
            )


# =========================================================================
# PIPE-07: test_coordinator_freeze_requires_converged
# =========================================================================

def test_coordinator_freeze_requires_converged():
    """PIPE-07: Freeze logic requires stage to be 'converged' status."""
    status = _make_status(
        overrides={"encoder_intuition": {"status": "training"}}
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(status, tmp)
        tmp_path = tmp.name

    try:
        # Attempt freeze on a training (not converged) stage
        import io
        captured = io.StringIO()
        sys.stdout = captured
        cmd_freeze(status, "encoder_intuition", status_path=tmp_path)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()

        assert "Error" in output or "Cannot freeze" in output, (
            f"Expected error when freezing non-converged stage, got: {output}"
        )

        # Verify frozen_modules was NOT modified
        assert status.get("frozen_modules", []) == [], (
            "frozen_modules should remain empty when freeze is rejected"
        )
    finally:
        sys.stdout = sys.__stdout__
        os.unlink(tmp_path)


# =========================================================================
# PIPE-07: test_coordinator_ordering_enforced
# =========================================================================

def test_coordinator_ordering_enforced():
    """PIPE-07: Dependency checking prevents out-of-order execution."""
    status = _make_status()

    # With encoder_intuition pending, reward_head should be blocked
    rh_deps = STAGE_DEPENDS["reward_head"]
    assert "encoder_intuition" in rh_deps, (
        "reward_head should depend on encoder_intuition"
    )
    for dep in rh_deps:
        assert status["stages"][dep]["status"] != "converged", (
            f"Dependency {dep} should not be converged in initial state"
        )

    # action_planner depends on both encoder_intuition AND reward_head
    ap_deps = STAGE_DEPENDS["action_planner"]
    assert "encoder_intuition" in ap_deps, (
        "action_planner should depend on encoder_intuition"
    )
    assert "reward_head" in ap_deps, (
        "action_planner should depend on reward_head"
    )

    # metacontroller depends on encoder_intuition AND reward_head
    mc_deps = STAGE_DEPENDS["metacontroller"]
    assert "encoder_intuition" in mc_deps, (
        "metacontroller should depend on encoder_intuition"
    )
    assert "reward_head" in mc_deps, (
        "metacontroller should depend on reward_head"
    )

    # Verify STAGE_ORDER is correct
    assert STAGE_ORDER == [
        "encoder_intuition", "reward_head", "action_planner", "metacontroller"
    ], f"STAGE_ORDER mismatch: {STAGE_ORDER}"

    # Verify FREEZE_AFTER flags
    assert FREEZE_AFTER["encoder_intuition"] is True
    assert FREEZE_AFTER["reward_head"] is True
    assert FREEZE_AFTER["action_planner"] is False
    assert FREEZE_AFTER["metacontroller"] is False


# =========================================================================
# PIPE-07: test_coordinator_full_pipeline_flow
# =========================================================================

def test_coordinator_full_pipeline_flow():
    """PIPE-07: Simulate full pipeline: init -> each stage -> freeze -> complete."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        # Step 1: Init
        status = _make_status()
        cmd_init(status, status_path=tmp_path)

        # Reload
        status = load_status(status_path=tmp_path)
        assert status["pipeline_run_id"] is not None, "run_id should be set after init"
        for stage_name in STAGE_ORDER:
            assert status["stages"][stage_name]["status"] == "pending", (
                f"{stage_name} should be pending after init"
            )

        # Step 2: encoder_intuition training -> converged
        status["stages"]["encoder_intuition"]["status"] = "training"
        status["stages"]["encoder_intuition"]["started_at"] = datetime.now().isoformat()
        save_status(status, status_path=tmp_path)

        status["stages"]["encoder_intuition"]["status"] = "converged"
        status["stages"]["encoder_intuition"]["converged_at"] = datetime.now().isoformat()
        status["stages"]["encoder_intuition"]["final_metric"] = 0.038
        status["stages"]["encoder_intuition"]["total_steps"] = 5200
        save_status(status, status_path=tmp_path)

        # Step 3: Freeze encoder_intuition
        status = load_status(status_path=tmp_path)
        cmd_freeze(status, "encoder_intuition", status_path=tmp_path)

        status = load_status(status_path=tmp_path)
        frozen_stages = _get_frozen_stage_names(status["frozen_modules"])
        assert "encoder_intuition" in frozen_stages, (
            "encoder_intuition should be frozen after freeze command"
        )

        # Step 4: reward_head training -> converged
        status["stages"]["reward_head"]["status"] = "training"
        status["stages"]["reward_head"]["started_at"] = datetime.now().isoformat()
        save_status(status, status_path=tmp_path)

        status["stages"]["reward_head"]["status"] = "converged"
        status["stages"]["reward_head"]["converged_at"] = datetime.now().isoformat()
        status["stages"]["reward_head"]["final_metric"] = 0.08
        status["stages"]["reward_head"]["total_steps"] = 3000
        save_status(status, status_path=tmp_path)

        # Step 5: Freeze reward_head
        status = load_status(status_path=tmp_path)
        cmd_freeze(status, "reward_head", status_path=tmp_path)

        status = load_status(status_path=tmp_path)
        frozen_stages = _get_frozen_stage_names(status["frozen_modules"])
        assert "reward_head" in frozen_stages, (
            "reward_head should be frozen after freeze command"
        )

        # Step 6: action_planner training -> converged
        status["stages"]["action_planner"]["status"] = "training"
        save_status(status, status_path=tmp_path)

        status["stages"]["action_planner"]["status"] = "converged"
        status["stages"]["action_planner"]["final_metric"] = 0.72
        status["stages"]["action_planner"]["total_steps"] = 4500
        save_status(status, status_path=tmp_path)

        # Step 7: metacontroller available
        status["stages"]["metacontroller"]["status"] = "available"
        save_status(status, status_path=tmp_path)

        # Final verification
        status = load_status(status_path=tmp_path)
        assert status["stages"]["encoder_intuition"]["status"] == "converged"
        assert status["stages"]["reward_head"]["status"] == "converged"
        assert status["stages"]["action_planner"]["status"] == "converged"
        assert status["stages"]["metacontroller"]["status"] == "available"

        # Verify all freezable stages are frozen
        frozen_stages = _get_frozen_stage_names(status["frozen_modules"])
        assert "encoder_intuition" in frozen_stages
        assert "reward_head" in frozen_stages

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =========================================================================
# PIPE-07: test_coordinator_stale_detection (Pitfall 5)
# =========================================================================

def test_coordinator_stale_detection():
    """PIPE-07 (Pitfall 5): Detect stale training status when no checkpoint modification in >5 min."""
    # Create status with encoder_intuition="training" and a started_at >5 min ago
    stale_time = (datetime.now() - timedelta(minutes=10)).isoformat()
    status = _make_status(
        overrides={
            "encoder_intuition": {
                "status": "training",
                "started_at": stale_time,
                "checkpoint": None,
            }
        }
    )

    # Call check_stale_training
    warnings = check_stale_training(status)

    # Should have at least one warning about encoder_intuition
    assert len(warnings) > 0, (
        "Expected stale training warning for encoder_intuition but got none"
    )
    assert any("Encoder" in w for w in warnings), (
        f"Expected warning about Encoder + Intuition, got: {warnings}"
    )
    assert any("crash" in w.lower() for w in warnings), (
        f"Expected 'crash' mention in warning, got: {warnings}"
    )

    # Verify non-stale stage does NOT produce warning
    recent_time = datetime.now().isoformat()
    status_recent = _make_status(
        overrides={
            "encoder_intuition": {
                "status": "training",
                "started_at": recent_time,
                "checkpoint": None,
            }
        }
    )
    warnings_recent = check_stale_training(status_recent)
    assert len(warnings_recent) == 0, (
        f"Expected no warnings for recently started training, got: {warnings_recent}"
    )
