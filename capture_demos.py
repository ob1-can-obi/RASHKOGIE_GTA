"""
Demonstration capture utility for action planner imitation learning data.
Per D-06: dedicated capture mode that saves raw states + player actions.
Per D-04: JSONL format, session-timestamped filenames.
Per Pitfall 4: saves raw controls, tokenization happens in preprocessing.

Usage:
  Called during human driving sessions.
  Produces: action_planner/training_data/session_YYYYMMDD_HHMMSS.jsonl
"""

import json
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent


class DemoCaptureSession:
    """
    Captures player demonstrations for action planner imitation learning.

    Per D-06: dedicated capture mode that saves raw states + player actions.
    Saves raw controls -- tokenization happens in preprocessing (Pitfall 4).
    """

    def __init__(self, project_root=None):
        root = Path(project_root) if project_root else _ROOT
        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.data_dir = root / "action_planner" / "training_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.file = open(
            self.data_dir / f"session_{self.session_ts}.jsonl", "w"
        )
        self.file.write(f"# session session_{self.session_ts}\n")
        self.frame_idx = 0

    def record_frame(self, raw_state, player_controls):
        """
        Called once per frame during human driving.

        Args:
            raw_state: Full GTA state dict from SHVDN
            player_controls: dict with throttle, brake, steer, handbrake
        """
        record = {
            "state": raw_state,
            "player_controls": {
                "throttle": float(player_controls.get("throttle", 0.0)),
                "brake": float(player_controls.get("brake", 0.0)),
                "steer": float(player_controls.get("steer", 0.0)),
                "handbrake": bool(player_controls.get("handbrake", False)),
            },
            "token_id": None,  # filled in by tokenization preprocessing per Pitfall 4
            "session_ts": self.session_ts,
            "frame_idx": self.frame_idx,
        }
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        self.frame_idx += 1

    def close(self):
        """Close the JSONL file."""
        self.file.close()


def write_synthetic_demo_data(output_dir, num_records=10):
    """Write synthetic action planner training data for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"session_{ts}.jsonl"
    with open(path, "w") as f:
        f.write(f"# session session_{ts}\n")
        for i in range(num_records):
            record = {
                "state": _minimal_state(),
                "player_controls": {
                    "throttle": 0.5 + (i % 3) * 0.1,
                    "brake": 0.0,
                    "steer": 0.1 * (i % 5 - 2),
                    "handbrake": False,
                },
                "token_id": i % 5,  # pre-tokenized for test convenience
                "session_ts": ts,
                "frame_idx": i,
            }
            f.write(json.dumps(record) + "\n")
    return path


def _minimal_state():
    """Create a minimal GTA state dict for testing."""
    return {
        "near_entities": [], "near_vehs": [], "near_peds": [], "near_objects": [],
        "wp_dist": 10.0, "hp": 100.0, "v_engine_hp": 1000.0,
        "v_body_hp": 1000.0, "road_dist": 0.5, "dead": False,
    }
