# Encoder + Intuition Head Training Data

JSONL files containing consecutive state pairs for next-state prediction training.
One JSON object per line. First line is a # comment with session metadata.

## Schema

Each record:
- state_t: Full GTA state dict at timestep t
- state_t1: Full GTA state dict at timestep t+1
- token_id: Token active during s_t -> s_{t+1} (integer)
- session_ts: Session timestamp (YYYYMMDD_HHMMSS)
- frame_idx: Frame index within session (integer)

## File naming

session_YYYYMMDD_HHMMSS.jsonl

## Capture

Run: python capture_states.py (hooks into frame loop)
Or: write_synthetic_encoder_data() for test data
