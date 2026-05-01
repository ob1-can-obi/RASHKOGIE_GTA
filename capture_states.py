"""
State capture utility for encoder+intuition and reward head training data.
Per D-05: records raw state sequences during gameplay sessions to disk.
Per D-04: JSONL format, session-timestamped filenames.

Usage:
  Called from within the frame loop or as a standalone capture session.
  Produces two JSONL files per session:
    - main_model/training_data/session_YYYYMMDD_HHMMSS.jsonl  (state pairs)
    - reward_head/training_data/session_YYYYMMDD_HHMMSS.jsonl  (reward pairs)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

_ROOT = Path(__file__).resolve().parent
for _d in ("metacontroller",):
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trainer import compute_token_return
from reward import compute_reward


class StateCaptureSession:
    """
    Captures state sequences during gameplay for BOTH encoder+intuition
    AND reward head training data.

    Per D-05: live capture + offline training -- records raw state sequences
    during gameplay sessions to disk, then trains offline from saved files.

    For encoder+intuition: writes (state_t, state_t+1, token_id) pairs.
    For reward head: writes (state_before, state_after, duration, realized_return)
    records at token boundaries.
    """

    def __init__(self, project_root=None):
        root = Path(project_root) if project_root else _ROOT
        self.session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Encoder+intuition data dir
        self.encoder_dir = root / "main_model" / "training_data"
        self.encoder_dir.mkdir(parents=True, exist_ok=True)

        # Reward head data dir
        self.reward_dir = root / "reward_head" / "training_data"
        self.reward_dir.mkdir(parents=True, exist_ok=True)

        # Open JSONL files
        self.encoder_file = open(
            self.encoder_dir / f"session_{self.session_ts}.jsonl", "w"
        )
        self.reward_file = open(
            self.reward_dir / f"session_{self.session_ts}.jsonl", "w"
        )

        # Write session header comment (matches tokenizer pattern)
        self.encoder_file.write(f"# session session_{self.session_ts}\n")
        self.reward_file.write(f"# session session_{self.session_ts}\n")

        self.frame_idx = 0
        self.prev_state = None
        self.token_rewards = []  # accumulate per-frame rewards for current token
        self.token_start_state = None  # state at start of current token

    def record_frame(self, raw_state, token_id, token_started=False, token_ended=False):
        """
        Called once per gameplay frame.

        Args:
            raw_state: Full GTA state dict from SHVDN
            token_id: Current active token ID
            token_started: True on the first frame of a new token
            token_ended: True when a token completes (commit/expire)
        """
        if token_started:
            # Reset for new token -- save the state at token start for reward record
            self.token_rewards = []
            self.token_start_state = raw_state

        if self.prev_state is not None:
            # Write encoder+intuition record: consecutive state pair
            encoder_record = {
                "state_t": self.prev_state,
                "state_t1": raw_state,
                "token_id": token_id,
                "session_ts": self.session_ts,
                "frame_idx": self.frame_idx,
            }
            self.encoder_file.write(json.dumps(encoder_record) + "\n")
            self.encoder_file.flush()

            # Accumulate per-frame reward for reward head data
            # compute_reward returns (reward, components) tuple
            reward, _components = compute_reward(self.prev_state, raw_state)
            self.token_rewards.append(reward)

        if token_ended and self.token_start_state is not None and len(self.token_rewards) > 0:
            # Token completed -- write reward head record
            # Build a rollout dict matching compute_token_return format
            rollout = {
                "rewards": self.token_rewards,
                "duration": len(self.token_rewards),
                "state_before": self.token_start_state,
                "state_after": raw_state,
            }
            realized_return = compute_token_return(rollout)

            reward_record = {
                "state_before": rollout["state_before"],
                "state_after": rollout["state_after"],
                "duration": rollout["duration"],
                "realized_return": float(realized_return),
                "session_ts": self.session_ts,
                "frame_idx": self.frame_idx,
            }
            self.reward_file.write(json.dumps(reward_record) + "\n")
            self.reward_file.flush()

            # Reset token rewards for next token
            self.token_rewards = []
            self.token_start_state = None

        self.prev_state = raw_state
        self.frame_idx += 1

    def close(self):
        """Close both JSONL files."""
        self.encoder_file.close()
        self.reward_file.close()


def write_synthetic_encoder_data(output_dir, num_records=10):
    """Write synthetic encoder+intuition training data for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"session_{ts}.jsonl"
    with open(path, "w") as f:
        f.write(f"# session session_{ts}\n")
        for i in range(num_records):
            record = {
                "state_t": _minimal_state(wp_dist=10.0 + i * 0.5),
                "state_t1": _minimal_state(wp_dist=10.0 + i * 0.5 - 0.3),
                "token_id": i % 10,
                "session_ts": ts,
                "frame_idx": i,
            }
            f.write(json.dumps(record) + "\n")
    return path


def write_synthetic_reward_data(output_dir, num_records=10):
    """Write synthetic reward head training data for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"session_{ts}.jsonl"
    with open(path, "w") as f:
        f.write(f"# session session_{ts}\n")
        for i in range(num_records):
            record = {
                "state_before": _minimal_state(wp_dist=10.0 + i * 0.5),
                "state_after": _minimal_state(wp_dist=10.0 + i * 0.5 - 0.3),
                "duration": 5,
                "realized_return": 0.3 + i * 0.05,
                "session_ts": ts,
                "frame_idx": i,
            }
            f.write(json.dumps(record) + "\n")
    return path


def _minimal_state(wp_dist=10.0):
    """Create a minimal GTA state dict for testing."""
    return {
        "near_entities": [], "near_vehs": [], "near_peds": [], "near_objects": [],
        "wp_dist": wp_dist, "hp": 100.0, "v_engine_hp": 1000.0,
        "v_body_hp": 1000.0, "road_dist": 0.5, "dead": False,
    }
