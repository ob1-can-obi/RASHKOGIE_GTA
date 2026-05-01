# Reward Head Training Data

JSONL files containing state pairs with realized returns for reward prediction training.
One JSON object per line. First line is a # comment with session metadata.

## Schema

Each record:
- state_before: Full GTA state dict at token start
- state_after: Full GTA state dict at token end
- duration: Token duration in frames (integer)
- realized_return: Discounted return from compute_token_return() (float)
- session_ts: Session timestamp (YYYYMMDD_HHMMSS)
- frame_idx: Frame index within session (integer)

## File naming

session_YYYYMMDD_HHMMSS.jsonl

## Capture

Run: python capture_states.py (produces both encoder and reward data)
Or: write_synthetic_reward_data() for test data

## Subfolders

```
training_data/
  session_YYYYMMDD_HHMMSS.jsonl   one file per capture session
  stats/
    reward_head_stats.csv         aggregated per-step stats
    summary.txt                   overall metrics
  graphs/
    loss_curve.svg                reward_loss and rf_loss over steps
    predicted_vs_actual.svg       scatter: predicted r_edge vs realized return
```

Run `python reward_head/stats.py` to regenerate stats and graphs after training.
