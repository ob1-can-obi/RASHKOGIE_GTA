# Reward Data

This folder stores per-frame reward logs.

Each row is JSONL and contains:

```text
step
ts
next_ts
reward
components
```

The reward is calculated from the frame before the action and the next frame
after that action.

