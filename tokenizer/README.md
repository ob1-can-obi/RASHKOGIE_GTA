# Tokenizer

This folder holds the GTA control tokenizer.

The tokenizer is a lookup table only:

- one token id maps to one raw GTA control chunk
- each lookup entry stores the raw chunk directly
- each lookup entry also stores `frame_count`
- each lookup entry also stores `default_duration_s`
- no learned token embeddings are used
- saved JSON uses compact integer bins for the chunk frames

## Workflow

Two-step process: capture controls while playing, build tokenizer after.

### Step 1: Capture controls (run while playing)

```bash
python tokenizer/tokenizer.py capture
```

This connects to the live game pipe and saves only the control inputs
(~80 bytes/frame) into a timestamped JSONL file under
`tokenizer/training_data/captures/`.

Each play session creates a new file. Run capture as many times as you want
across multiple sessions — data accumulates incrementally:

```
training_data/captures/
  session_20260426_140000.jsonl   # first session
  session_20260426_180000.jsonl   # second session after a break
  session_20260427_100000.jsonl   # next day
```

At the start of each capture it prints how much data you already have.

You can use keyboard or a controller (Xbox, PlayStation, etc.) to play.
The capture reads `u_steer`, `u_throttle`, `u_brake` from the game state,
which are the raw player inputs regardless of input device. A controller
is recommended — analog sticks give smooth continuous values instead of
keyboard's binary 0/1, producing more varied bin combinations.

### Step 2: Build tokenizer (run after playing)

```bash
python tokenizer/tokenizer.py build
```

This reads all `.jsonl` captures (plus any legacy `stats/` frames), runs BPE,
and writes the tokenizer lookup table. It resumes from any existing tokenizer
automatically, keeping old token ids stable.

Custom merge count:

```bash
python tokenizer/tokenizer.py build --max-merges 1024 --min-pair-count 10
```

## Session Boundaries and TOKEN_END

Each `.jsonl` capture file is treated as one driving session. The tokenizer
inserts a special `TOKEN_END` (id = -1) marker between sessions:

```
[...session 1 tokens...] TOKEN_END [...session 2 tokens...] TOKEN_END ...
```

This solves a key training problem: without TOKEN_END, the last idle frames
of one session would bleed into the first frames of the next session. The
model would learn false transitions across session gaps.

TOKEN_END behavior:

- BPE merges never bridge across TOKEN_END — pairs that include it are skipped.
- `encode_control_frames()` appends TOKEN_END by default (`append_end=True`).
- `decode_token_ids()` silently skips TOKEN_END entries.
- The training loop should treat TOKEN_END as a sequence boundary: reset
  hidden state or skip the loss when it appears.

## Stable Token IDs

Token ids are now stable across rebuilds.

The builder automatically resumes from the existing latest tokenizer file when
`tokenizer/training_data/tokenizer_lookup.json` already exists.

That means:

- old token ids stay fixed
- old token chunks stay fixed
- old merge rules stay fixed
- new data can only append new token ids at the end

This is required for using the tokenizer as a persistent action lookup table.

## Base Symbol

One base symbol is one quantized GTA control frame with four axes:

```text
right_steer
left_steer
forward_throttle
brake
```

The capture step reads these from the live game pipe via:

```text
u_steer        -> split into right_steer / left_steer
u_throttle     -> forward_throttle
u_brake        -> brake
```

When loading legacy `gta_stream/stats/` frames, the tokenizer falls back to
`agent_*` then `v_*` fields.

## Quantization

BPE needs repeated discrete symbols. Raw analog controls are continuous,
so the tokenizer quantizes each axis into integer bins first.

Current defaults — 8 bins per axis (0-7):

```text
step = 1/7 (~0.143)

right_steer_bin      = 0..7
left_steer_bin       = 0..7
forward_throttle_bin = 0..7
brake_bin            = 0..7
```

This gives up to 8^4 = 4096 possible base token combinations. In practice
only the combinations that actually appear in your driving data get token ids.

Override with CLI flags:

```bash
python tokenizer/tokenizer.py build --steer-step 0.1 --throttle-step 0.1 --brake-step 0.1
```

On disk, saved chunk files store integer bins instead of float values.
That is smaller and keeps the vocabulary truly discrete.

## Lookup Entry

Each token id stores:

```text
token_id
token_name
token_text
raw_control_chunk
frame_count
default_duration_s
source_pair
source_count
```

`frame_count` is the total number of frames represented by the token.

That means if a token is formed by merging five one-frame symbols, the lookup
entry keeps:

```text
frame_count = 5
default_duration_s = 5 / fps
```

## BPE-Style Build

`tokenizer.py build` does this:

1. Load all JSONL capture files as separate sessions (+ legacy stats frames).
2. Extract one raw control frame from each line.
3. Quantize each frame into a discrete base symbol (8 bins per axis).
4. Build the one-frame base vocabulary.
5. Insert TOKEN_END between sessions.
6. Replay any existing merge rules from a previous tokenizer.
7. Count adjacent token pairs (skipping pairs that touch TOKEN_END).
8. Merge the most frequent pair into a new token.
9. Store the merged raw control chunk, frame count, and duration.
10. Repeat until `max_merges` or no frequent pair remains.

## Files

- `tokenizer.py`
  - `capture` subcommand: captures controls from live game into JSONL.
  - `build` subcommand: builds the BPE tokenizer from captures.
  - Encodes frames into token ids.
  - Decodes token ids back into raw GTA control chunks.
  - Saves and loads the tokenizer JSON.

- `chunk_lookup.py`
  - Loads the saved tokenizer file.
  - Returns the chunk for one token id.

- `training_data/captures/`
  - One JSONL file per play session.
  - Accumulates across sessions for incremental data collection.

- `training_data/tokenizer_chunks.json`
  - Stores every token chunk in one chunk-only file.
  - Uses compact integer bins for the saved chunk frames.

- `training_data/snapshots/`
  - Stores timestamped tokenizer snapshots.
  - Lets you keep today's token set before rebuilding tomorrow with more data.

## Example

Full workflow:

```bash
# Session 1: play for a while
python tokenizer/tokenizer.py capture
# (play GTA, press Ctrl-C when done)

# Session 2: play more another day
python tokenizer/tokenizer.py capture
# (more driving, Ctrl-C)

# Build tokenizer on all accumulated data
python tokenizer/tokenizer.py build

# Inspect one token chunk
python tokenizer/chunk_lookup.py 12
```

`max_merges` is treated as the total lifetime merge budget for the tokenizer,
not a fresh renumbering pass every time you rebuild.

The saved JSON becomes the lookup table for the future action planner,
intuition head, and metacontroller.
