# Action Planner Training Data

JSONL files containing player demonstrations for imitation learning.
One JSON object per line. First line is a # comment with session metadata.

## Schema

Each record:
- state: Full GTA state dict
- player_controls: {throttle, brake, steer, handbrake}
- token_id: Token label (null until tokenization preprocessing, integer after)
- session_ts: Session timestamp (YYYYMMDD_HHMMSS)
- frame_idx: Frame index within session (integer)

## File naming

session_YYYYMMDD_HHMMSS.jsonl

## Capture

Run: python capture_demos.py (dedicated human driving capture mode per D-06)
Or: write_synthetic_demo_data() for test data

## Preprocessing

Before training, run tokenization to fill token_id from player_controls.
Records with token_id=null are filtered out by the training script.
