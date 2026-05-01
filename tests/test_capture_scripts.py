"""
Tests for Phase 4 Plan 06: Data Capture Scripts (PIPE-01, PIPE-02, PIPE-03, PIPE-04).

Tests validate that capture_states.py and capture_demos.py produce correctly
formatted JSONL files with the expected schemas.

Tests use temporary directories -- they do not require GTA or real game states.
"""

import sys
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture_states import (
    StateCaptureSession,
    write_synthetic_encoder_data,
    write_synthetic_reward_data,
    _minimal_state,
)
from capture_demos import (
    DemoCaptureSession,
    write_synthetic_demo_data,
)


def test_capture_states_writes_encoder_jsonl():
    """PIPE-01: write_synthetic_encoder_data produces valid JSONL with correct schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_synthetic_encoder_data(tmpdir, num_records=5)

        assert path.exists(), f"Expected file at {path}"

        with open(path, "r") as f:
            lines = f.readlines()

        # First line is a comment header
        assert lines[0].startswith("#"), "First line should be a # comment header"

        # 5 data records after the comment
        data_lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
        assert len(data_lines) == 5, f"Expected 5 records, got {len(data_lines)}"

        # Verify each record has the expected schema
        for line in data_lines:
            record = json.loads(line.strip())
            assert "state_t" in record, "Record must have state_t"
            assert "state_t1" in record, "Record must have state_t1"
            assert "token_id" in record, "Record must have token_id"
            assert "session_ts" in record, "Record must have session_ts"
            assert "frame_idx" in record, "Record must have frame_idx"

            # state_t should be a dict with expected keys
            assert isinstance(record["state_t"], dict), "state_t should be a dict"
            assert "wp_dist" in record["state_t"], "state_t should have wp_dist"


def test_capture_states_writes_reward_jsonl():
    """PIPE-01: write_synthetic_reward_data produces valid JSONL with reward head schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_synthetic_reward_data(tmpdir, num_records=5)

        assert path.exists(), f"Expected file at {path}"

        with open(path, "r") as f:
            lines = f.readlines()

        # First line is a comment header
        assert lines[0].startswith("#"), "First line should be a # comment header"

        # 5 data records after the comment
        data_lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
        assert len(data_lines) == 5, f"Expected 5 records, got {len(data_lines)}"

        # Verify each record has the expected reward head schema
        for line in data_lines:
            record = json.loads(line.strip())
            assert "state_before" in record, "Record must have state_before"
            assert "state_after" in record, "Record must have state_after"
            assert "duration" in record, "Record must have duration"
            assert "realized_return" in record, "Record must have realized_return"
            assert "session_ts" in record, "Record must have session_ts"
            assert "frame_idx" in record, "Record must have frame_idx"

            # realized_return should be a float
            assert isinstance(record["realized_return"], float), (
                f"realized_return should be float, got {type(record['realized_return'])}"
            )


def test_capture_demos_writes_jsonl():
    """PIPE-04: write_synthetic_demo_data produces valid JSONL with action planner schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_synthetic_demo_data(tmpdir, num_records=5)

        assert path.exists(), f"Expected file at {path}"

        with open(path, "r") as f:
            lines = f.readlines()

        # First line is a comment header
        assert lines[0].startswith("#"), "First line should be a # comment header"

        # 5 data records after the comment
        data_lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
        assert len(data_lines) == 5, f"Expected 5 records, got {len(data_lines)}"

        # Verify each record has the expected schema
        for line in data_lines:
            record = json.loads(line.strip())
            assert "state" in record, "Record must have state"
            assert "player_controls" in record, "Record must have player_controls"
            assert "token_id" in record, "Record must have token_id"

            # player_controls should have throttle, brake, steer, handbrake
            pc = record["player_controls"]
            assert "throttle" in pc, "player_controls must have throttle"
            assert "brake" in pc, "player_controls must have brake"
            assert "steer" in pc, "player_controls must have steer"
            assert "handbrake" in pc, "player_controls must have handbrake"


def test_capture_session_encoder_class():
    """PIPE-02: StateCaptureSession.record_frame writes consecutive state pairs to encoder JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = StateCaptureSession(project_root=tmpdir)

        # Record two consecutive frames (simulates gameplay)
        state1 = _minimal_state(wp_dist=10.0)
        state2 = _minimal_state(wp_dist=9.7)

        session.record_frame(state1, token_id=5, token_started=True)
        session.record_frame(state2, token_id=5)
        session.close()

        # Read the encoder JSONL output
        encoder_files = list((Path(tmpdir) / "main_model" / "training_data").glob("*.jsonl"))
        assert len(encoder_files) == 1, f"Expected 1 encoder JSONL file, got {len(encoder_files)}"

        with open(encoder_files[0], "r") as f:
            lines = f.readlines()

        # First line is comment, then 1 record (pair from frame 1->2)
        assert lines[0].startswith("#"), "First line should be # comment"
        data_lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
        assert len(data_lines) == 1, (
            f"Expected 1 encoder record (state pair from frame 0->1), got {len(data_lines)}"
        )

        record = json.loads(data_lines[0].strip())
        assert "state_t" in record, "Record must have state_t"
        assert "state_t1" in record, "Record must have state_t1"
        assert record["token_id"] == 5, f"token_id should be 5, got {record['token_id']}"
        assert record["state_t"]["wp_dist"] == 10.0, "state_t should be first frame state"
        assert record["state_t1"]["wp_dist"] == 9.7, "state_t1 should be second frame state"


def test_capture_demo_session_class():
    """PIPE-04: DemoCaptureSession.record_frame writes demo records with token_id=None per D-06."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session = DemoCaptureSession(project_root=tmpdir)

        # Record a single frame with player controls
        state = _minimal_state()
        controls = {"throttle": 0.8, "brake": 0.0, "steer": -0.1, "handbrake": False}
        session.record_frame(state, controls)
        session.close()

        # Read the demo JSONL output
        demo_files = list((Path(tmpdir) / "action_planner" / "training_data").glob("*.jsonl"))
        assert len(demo_files) == 1, f"Expected 1 demo JSONL file, got {len(demo_files)}"

        with open(demo_files[0], "r") as f:
            lines = f.readlines()

        assert lines[0].startswith("#"), "First line should be # comment"
        data_lines = [l for l in lines if not l.strip().startswith("#") and l.strip()]
        assert len(data_lines) == 1, f"Expected 1 demo record, got {len(data_lines)}"

        record = json.loads(data_lines[0].strip())
        assert "state" in record, "Record must have state"
        assert record["token_id"] is None, (
            f"token_id should be None (raw capture per D-06), got {record['token_id']}"
        )
        assert record["player_controls"]["throttle"] == 0.8, "throttle should be 0.8"
        assert record["player_controls"]["steer"] == -0.1, "steer should be -0.1"


def test_jsonl_comment_header():
    """All three synthetic data writers produce files where line 1 starts with #."""
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder_dir = Path(tmpdir) / "encoder"
        reward_dir = Path(tmpdir) / "reward"
        demo_dir = Path(tmpdir) / "demo"

        encoder_path = write_synthetic_encoder_data(encoder_dir, num_records=3)
        reward_path = write_synthetic_reward_data(reward_dir, num_records=3)
        demo_path = write_synthetic_demo_data(demo_dir, num_records=3)

        for name, path in [("encoder", encoder_path), ("reward", reward_path), ("demo", demo_path)]:
            with open(path, "r") as f:
                first_line = f.readline()
            assert first_line.startswith("#"), (
                f"{name} JSONL first line should start with # but got: {first_line[:50]}"
            )
            assert "session" in first_line, (
                f"{name} comment header should contain 'session' but got: {first_line[:50]}"
            )
