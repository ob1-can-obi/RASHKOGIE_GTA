# Tokenizer Training Data

This folder stores tokenizer artifacts and capture data.

## Subfolders

```
training_data/
  captures/
    session_YYYYMMDD_HHMMSS.jsonl   one file per play session
                                    (raw control frames captured while driving)
  tokenizer_lookup.json             latest built tokenizer lookup table
  tokenizer_chunks.json             all token chunks (compact integer bin format)
  snapshots/
    tokenizer_YYYYMMDD_HHMMSS.json  timestamped snapshots before rebuilds
```

## Capture Files

Each `.jsonl` file is one play session.  Each line is one frame of raw control input:

```json
{"u_steer": 0.14, "u_throttle": 0.62, "u_brake": 0.0, "u_handbrake": false}
```

Sessions accumulate incrementally.  The tokenizer builder reads all of them at once.

## Tokenizer Lookup

`tokenizer_lookup.json` is the persistent lookup table used by the executor and
search tree at runtime.  Token ids are stable across rebuilds — new tokens are only
appended, never renumbered.

See `tokenizer/README.md` for the full build workflow.
