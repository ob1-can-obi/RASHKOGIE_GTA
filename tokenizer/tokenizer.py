"""
Lookup-table tokenizer for GTA control chunks.

This file keeps the tokenizer simple and explicit:

- one token id maps to one raw GTA control chunk
- each lookup entry stores the chunk, total frames, and default duration
- tokenization starts from one-frame control symbols
- longer tokens are learned with a BPE-style merge loop

No learned token embeddings are used here.

Workflow:
1. Capture controls:  python tokenizer.py capture  (run while playing)
2. Build tokenizer:   python tokenizer.py build    (run after playing)
"""

from pathlib import Path
import argparse
import json
import os
import sys
import time


CONTROL_FIELDS = ["right_steer", "left_steer", "forward_throttle", "brake"]

NAV_FIELDS = [
    "speed", "heading", "road_heading", "road_dist",
    "wp_dist", "wp_x", "wp_y", "pos_x", "pos_y",
]

TOKEN_END = -1

# 10 bins per axis: 0..9 over [0.0, 1.0]  ->  step = 1/9
DEFAULT_STEP = 1.0 / 9.0


def quantize_value(value, step):
    """
    Quantize a float to a fixed step so frame symbols stay discrete.
    """

    if step <= 0.0:
        return float(value)

    scaled = float(value) / float(step)
    return round(scaled) * float(step)


def quantize_bin(value, step):
    """
    Quantize a float into one integer bin index.
    """

    if step <= 0.0:
        raise ValueError("quantization step must be positive")

    scaled = float(value) / float(step)
    return int(round(scaled))


def normalize_zero(value, eps=1e-9):
    """
    Remove negative zero after clamping or quantization.
    """

    if abs(float(value)) <= eps:
        return 0.0

    return float(value)


def clamp_control_frame(frame):
    """
    Clamp one raw control frame into the expected GTA control ranges.

    Accepts both new format (right_steer/left_steer/forward_throttle/brake)
    and legacy format (steer/throttle/brake/handbrake/enabled).
    """

    if "steer" in frame and "right_steer" not in frame:
        steer = float(frame.get("steer", 0.0) or 0.0)
        frame = {
            "right_steer": max(0.0, steer),
            "left_steer": max(0.0, -steer),
            "forward_throttle": float(frame.get("throttle", 0.0) or 0.0),
            "brake": float(frame.get("brake", 0.0) or 0.0),
        }

    return {
        "right_steer": max(0.0, min(1.0, float(frame.get("right_steer", 0.0) or 0.0))),
        "left_steer": max(0.0, min(1.0, float(frame.get("left_steer", 0.0) or 0.0))),
        "forward_throttle": max(0.0, min(1.0, float(frame.get("forward_throttle", 0.0) or 0.0))),
        "brake": max(0.0, min(1.0, float(frame.get("brake", 0.0) or 0.0))),
    }


def quantize_control_frame(
    frame,
    steer_step=DEFAULT_STEP,
    throttle_step=DEFAULT_STEP,
    brake_step=DEFAULT_STEP,
):
    """
    Convert one raw control frame into a discrete one-frame control symbol.
    """

    clamped = clamp_control_frame(frame)

    quantized = {
        "right_steer": quantize_value(clamped["right_steer"], steer_step),
        "left_steer": quantize_value(clamped["left_steer"], steer_step),
        "forward_throttle": quantize_value(clamped["forward_throttle"], throttle_step),
        "brake": quantize_value(clamped["brake"], brake_step),
    }

    quantized["right_steer"] = normalize_zero(max(0.0, min(1.0, quantized["right_steer"])))
    quantized["left_steer"] = normalize_zero(max(0.0, min(1.0, quantized["left_steer"])))
    quantized["forward_throttle"] = normalize_zero(max(0.0, min(1.0, quantized["forward_throttle"])))
    quantized["brake"] = normalize_zero(max(0.0, min(1.0, quantized["brake"])))

    return quantized


def control_frame_to_bins(
    frame,
    steer_step=DEFAULT_STEP,
    throttle_step=DEFAULT_STEP,
    brake_step=DEFAULT_STEP,
):
    """
    Convert one control frame into compact integer bins.
    """

    quantized = quantize_control_frame(
        frame,
        steer_step=steer_step,
        throttle_step=throttle_step,
        brake_step=brake_step,
    )

    return {
        "right_steer_bin": quantize_bin(quantized["right_steer"], steer_step),
        "left_steer_bin": quantize_bin(quantized["left_steer"], steer_step),
        "forward_throttle_bin": quantize_bin(quantized["forward_throttle"], throttle_step),
        "brake_bin": quantize_bin(quantized["brake"], brake_step),
    }


def control_frame_from_bins(
    frame_bins,
    steer_step=DEFAULT_STEP,
    throttle_step=DEFAULT_STEP,
    brake_step=DEFAULT_STEP,
):
    """
    Convert compact integer bins back into a quantized control frame.
    """

    return {
        "right_steer": normalize_zero(
            max(0.0, float(frame_bins["right_steer_bin"]) * float(steer_step))
        ),
        "left_steer": normalize_zero(
            max(0.0, float(frame_bins["left_steer_bin"]) * float(steer_step))
        ),
        "forward_throttle": normalize_zero(
            max(0.0, float(frame_bins["forward_throttle_bin"]) * float(throttle_step))
        ),
        "brake": normalize_zero(max(0.0, float(frame_bins["brake_bin"]) * float(brake_step))),
    }


def frame_to_token_text(frame):
    """
    Make one readable symbol name from a one-frame control dictionary.
    """

    return (
        f"rs{frame['right_steer']:.2f}_"
        f"ls{frame['left_steer']:.2f}_"
        f"ft{frame['forward_throttle']:.2f}_"
        f"b{frame['brake']:.2f}"
    )


def frame_bins_to_token_text(frame_bins):
    """
    Make one readable discrete symbol name from compact integer bins.
    """

    return (
        f"rs{int(frame_bins['right_steer_bin'])}_"
        f"ls{int(frame_bins['left_steer_bin'])}_"
        f"ft{int(frame_bins['forward_throttle_bin'])}_"
        f"b{int(frame_bins['brake_bin'])}"
    )


def make_control_frame_from_gta_state(raw_state):
    """
    Extract one control frame from one GTA state dictionary.

    If the state already has split fields (right_steer, left_steer, etc.)
    they are used directly.  Otherwise the legacy u_steer / agent_steer /
    v_steer values are read and split into right/left components.

    Preference order for legacy fields:
    - raw player inputs: u_*
    - latest automation command: agent_*
    - applied vehicle controls: v_*
    """

    if "right_steer" in raw_state and "left_steer" in raw_state:
        return {
            "right_steer": float(raw_state.get("right_steer", 0.0) or 0.0),
            "left_steer": float(raw_state.get("left_steer", 0.0) or 0.0),
            "forward_throttle": float(raw_state.get("forward_throttle", 0.0) or 0.0),
            "brake": float(raw_state.get("brake", 0.0) or 0.0),
        }

    if "u_steer" in raw_state:
        steer_value = float(raw_state.get("u_steer", 0.0) or 0.0)
    elif "agent_steer" in raw_state:
        steer_value = float(raw_state.get("agent_steer", 0.0) or 0.0)
    else:
        steer_value = float(raw_state.get("v_steer", 0.0) or 0.0)

    if "u_throttle" in raw_state:
        throttle_value = float(raw_state.get("u_throttle", 0.0) or 0.0)
    elif "agent_throttle" in raw_state:
        throttle_value = float(raw_state.get("agent_throttle", 0.0) or 0.0)
    else:
        throttle_value = float(raw_state.get("v_throttle", 0.0) or 0.0)

    if "u_brake" in raw_state:
        brake_value = float(raw_state.get("u_brake", 0.0) or 0.0)
    elif "agent_brake" in raw_state:
        brake_value = float(raw_state.get("agent_brake", 0.0) or 0.0)
    else:
        brake_value = float(raw_state.get("v_brake", 0.0) or 0.0)

    return {
        "right_steer": max(0.0, steer_value),
        "left_steer": max(0.0, -steer_value),
        "forward_throttle": throttle_value,
        "brake": brake_value,
    }


def load_gta_control_frames(frame_dir):
    """
    Read GTA JSON frames from a folder and extract raw player controls.
    """

    frame_dir = Path(frame_dir)
    json_paths = sorted(frame_dir.glob("*.json"))

    control_frames = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as file:
            raw_state = json.load(file)
        control_frames.append(make_control_frame_from_gta_state(raw_state))

    return control_frames


def load_control_capture(capture_path):
    """
    Load control frames from a compact JSONL capture file.

    Each line is one JSON object with steer, throttle, brake, handbrake, enabled.
    Lines starting with '#' are session markers and are skipped.

    Returns all frames from this file as one flat list (one session per file).
    """

    capture_path = Path(capture_path)
    control_frames = []

    with open(capture_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            control_frames.append(json.loads(line))

    return control_frames


def load_all_captures(capture_dir):
    """
    Load all .jsonl capture files from a directory as separate sessions.

    Returns a list of sessions, where each session is a list of control frames.
    Files are sorted by name so session order is preserved.
    """

    capture_dir = Path(capture_dir)
    sessions = []

    for path in sorted(capture_dir.glob("*.jsonl")):
        frames = load_control_capture(path)
        if frames:
            sessions.append(frames)

    return sessions


def load_all_captures_flat(capture_dir):
    """
    Load and concatenate all .jsonl capture files into one flat frame list.

    This is the legacy behavior without session boundaries.
    """

    sessions = load_all_captures(capture_dir)
    all_frames = []
    for session in sessions:
        all_frames.extend(session)
    return all_frames


def count_adjacent_pairs(token_sequence):
    """
    Count all adjacent token pairs in one sequence.

    Pairs that include TOKEN_END are never counted so BPE merges
    cannot bridge across session boundaries.
    """

    pair_counts = {}

    for index in range(len(token_sequence) - 1):
        left = token_sequence[index]
        right = token_sequence[index + 1]
        if left == TOKEN_END or right == TOKEN_END:
            continue
        pair = (left, right)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    return pair_counts


def replace_pair_once(token_sequence, pair, new_token_id):
    """
    Replace non-overlapping occurrences of one adjacent pair.

    Never merges across a TOKEN_END boundary.
    """

    merged = []
    index = 0

    while index < len(token_sequence):
        if index + 1 < len(token_sequence):
            current_pair = (token_sequence[index], token_sequence[index + 1])
            if current_pair == pair and pair[0] != TOKEN_END and pair[1] != TOKEN_END:
                merged.append(new_token_id)
                index += 2
                continue

        merged.append(token_sequence[index])
        index += 1

    return merged


def build_lookup_token_entry(
    token_id,
    token_text,
    raw_control_chunk,
    fps,
    source_pair=None,
    source_count=None,
):
    """
    Build one JSON-friendly lookup entry.
    """

    frame_count = len(raw_control_chunk)
    default_duration_s = float(frame_count) / float(fps)

    return {
        "token_id": int(token_id),
        "token_name": f"tok_{int(token_id):06d}",
        "token_text": str(token_text),
        "raw_control_chunk": raw_control_chunk,
        "frame_count": int(frame_count),
        "default_duration_s": float(default_duration_s),
        "source_pair": list(source_pair) if source_pair is not None else None,
        "source_count": int(source_count) if source_count is not None else None,
    }


def pack_raw_control_chunk(raw_control_chunk, quantization):
    """
    Convert one token chunk into a compact integer-bin list for JSON storage.
    """

    packed_chunk = []

    for frame in raw_control_chunk:
        packed_chunk.append(
            control_frame_to_bins(
                frame,
                steer_step=quantization["steer_step"],
                throttle_step=quantization["throttle_step"],
                brake_step=quantization["brake_step"],
            )
        )

    return packed_chunk


def unpack_raw_control_chunk(raw_control_chunk_bins, quantization):
    """
    Convert one compact integer-bin chunk back into quantized control frames.
    """

    unpacked_chunk = []

    for frame_bins in raw_control_chunk_bins:
        unpacked_chunk.append(
            control_frame_from_bins(
                frame_bins,
                steer_step=quantization["steer_step"],
                throttle_step=quantization["throttle_step"],
                brake_step=quantization["brake_step"],
            )
        )

    return unpacked_chunk


def make_token_chunks_payload(tokenizer_state):
    """
    Build a chunk-only JSON payload that stores every token's chunk compactly.
    """

    quantization = tokenizer_state["quantization"]
    tokens = []

    for token_id in sorted(tokenizer_state["token_lookup"]):
        entry = tokenizer_state["token_lookup"][token_id]
        tokens.append(
            {
                "token_id": int(entry["token_id"]),
                "token_name": entry["token_name"],
                "token_text": entry["token_text"],
                "frame_count": int(entry["frame_count"]),
                "default_duration_s": float(entry["default_duration_s"]),
                "raw_control_chunk_bins": pack_raw_control_chunk(
                    entry["raw_control_chunk"],
                    quantization,
                ),
            }
        )

    return {
        "fps": float(tokenizer_state["fps"]),
        "quantization": dict(quantization),
        "tokens": tokens,
    }


def clone_tokenizer_state(tokenizer_state):
    """
    Make a safe working copy of a tokenizer state.
    """

    token_lookup = {}
    for token_id, entry in tokenizer_state["token_lookup"].items():
        token_lookup[int(token_id)] = dict(entry)

    merge_rules = []
    for merge_rule in tokenizer_state["merge_rules"]:
        merge_rules.append(
            {
                "pair": [int(merge_rule["pair"][0]), int(merge_rule["pair"][1])],
                "token_id": int(merge_rule["token_id"]),
                "count": int(merge_rule["count"]),
            }
        )

    return {
        "fps": float(tokenizer_state["fps"]),
        "control_fields": list(tokenizer_state["control_fields"]),
        "quantization": dict(tokenizer_state["quantization"]),
        "token_lookup": token_lookup,
        "merge_rules": merge_rules,
        "encoded_ids": [int(token_id) for token_id in tokenizer_state.get("encoded_ids", [])],
        "base_vocab_size": int(tokenizer_state["base_vocab_size"]),
        "total_vocab_size": int(tokenizer_state["total_vocab_size"]),
    }


def validate_resume_tokenizer(
    tokenizer_state,
    fps,
    steer_step,
    throttle_step,
    brake_step,
):
    """
    Validate that resumed tokenizer settings match the current build settings.
    """

    quantization = tokenizer_state["quantization"]

    if abs(float(tokenizer_state["fps"]) - float(fps)) > 1e-9:
        raise ValueError("resume tokenizer fps does not match current fps")
    if abs(float(quantization["steer_step"]) - float(steer_step)) > 1e-9:
        raise ValueError("resume tokenizer steer_step does not match current steer_step")
    if abs(float(quantization["throttle_step"]) - float(throttle_step)) > 1e-9:
        raise ValueError("resume tokenizer throttle_step does not match current throttle_step")
    if abs(float(quantization["brake_step"]) - float(brake_step)) > 1e-9:
        raise ValueError("resume tokenizer brake_step does not match current brake_step")


def replay_merge_rules(token_sequence, merge_rules):
    """
    Apply an existing merge-rule list in order.
    """

    merged_sequence = list(token_sequence)

    for merge_rule in merge_rules:
        pair = (int(merge_rule["pair"][0]), int(merge_rule["pair"][1]))
        new_token_id = int(merge_rule["token_id"])
        merged_sequence = replace_pair_once(merged_sequence, pair, new_token_id)

    return merged_sequence


def build_control_tokenizer(
    control_frames,
    fps=20.0,
    max_merges=128,
    min_pair_count=2,
    steer_step=DEFAULT_STEP,
    throttle_step=DEFAULT_STEP,
    brake_step=DEFAULT_STEP,
    existing_tokenizer_state=None,
    sessions=None,
):
    """
    Build a lookup-table tokenizer from raw GTA control frames.

    Output:
    - tokenizer_state["token_lookup"]: token_id -> chunk, frame count, duration
    - tokenizer_state["merge_rules"]: BPE-style merge history
    - tokenizer_state["encoded_ids"]: token ids for the training sequence

    If `existing_tokenizer_state` is provided, old token ids stay fixed.
    Existing merge rules are replayed first, and only new token ids are appended.

    If `sessions` is provided (list of frame lists), TOKEN_END is inserted
    between sessions so BPE merges never bridge across session boundaries.
    The `control_frames` argument is ignored when `sessions` is given.
    """

    if sessions is not None:
        session_list = [s for s in sessions if s]
        if not session_list:
            raise ValueError("sessions is empty")
    else:
        if not control_frames:
            raise ValueError("control_frames is empty")
        session_list = [control_frames]

    if fps <= 0.0:
        raise ValueError("fps must be positive")

    # -------------------------------------------------------------------------
    # Step 1: turn each frame into one discrete base symbol
    # -------------------------------------------------------------------------

    all_frames_flat = []
    for session in session_list:
        all_frames_flat.extend(session)

    base_frames = []
    base_frame_bins = []
    for frame in all_frames_flat:
        quantized = quantize_control_frame(
            frame,
            steer_step=steer_step,
            throttle_step=throttle_step,
            brake_step=brake_step,
        )
        base_frames.append(quantized)
        base_frame_bins.append(
            control_frame_to_bins(
                quantized,
                steer_step=steer_step,
                throttle_step=throttle_step,
                brake_step=brake_step,
            )
        )

    # -------------------------------------------------------------------------
    # Step 2: initialize or resume the vocabulary
    # -------------------------------------------------------------------------

    if existing_tokenizer_state is None:
        token_lookup = {}
        merge_rules = []
        base_text_to_id = {}
        next_token_id = 0
    else:
        validate_resume_tokenizer(
            existing_tokenizer_state,
            fps=fps,
            steer_step=steer_step,
            throttle_step=throttle_step,
            brake_step=brake_step,
        )
        resumed_state = clone_tokenizer_state(existing_tokenizer_state)
        token_lookup = resumed_state["token_lookup"]
        merge_rules = resumed_state["merge_rules"]
        base_text_to_id = {}
        for token_id, entry in token_lookup.items():
            if int(entry["frame_count"]) == 1:
                base_text_to_id[entry["token_text"]] = int(token_id)
        if token_lookup:
            next_token_id = max(token_lookup) + 1
        else:
            next_token_id = 0

    # Build base sequence with TOKEN_END between sessions
    base_sequence = []
    flat_index = 0

    for session in session_list:
        for _ in session:
            frame = base_frames[flat_index]
            frame_bins = base_frame_bins[flat_index]
            token_text = frame_bins_to_token_text(frame_bins)

            if token_text not in base_text_to_id:
                token_id = next_token_id
                next_token_id += 1
                base_text_to_id[token_text] = token_id
                token_lookup[token_id] = build_lookup_token_entry(
                    token_id=token_id,
                    token_text=token_text,
                    raw_control_chunk=[frame],
                    fps=fps,
                )

            base_sequence.append(base_text_to_id[token_text])
            flat_index += 1

        base_sequence.append(TOKEN_END)

    # -------------------------------------------------------------------------
    # Step 3: replay old merge rules, then append new merges only
    # -------------------------------------------------------------------------

    encoded_ids = replay_merge_rules(base_sequence, merge_rules)

    new_merge_budget = max(0, int(max_merges) - len(merge_rules))

    for _ in range(new_merge_budget):
        pair_counts = count_adjacent_pairs(encoded_ids)
        if not pair_counts:
            break

        best_pair = None
        best_count = 0
        for pair, count in pair_counts.items():
            if count > best_count:
                best_pair = pair
                best_count = count

        if best_pair is None or best_count < min_pair_count:
            break

        left_token = token_lookup[best_pair[0]]
        right_token = token_lookup[best_pair[1]]
        merged_chunk = (
            list(left_token["raw_control_chunk"]) + list(right_token["raw_control_chunk"])
        )
        merged_text = left_token["token_name"] + " + " + right_token["token_name"]
        new_token_id = next_token_id
        next_token_id += 1

        token_lookup[new_token_id] = build_lookup_token_entry(
            token_id=new_token_id,
            token_text=merged_text,
            raw_control_chunk=merged_chunk,
            fps=fps,
            source_pair=best_pair,
            source_count=best_count,
        )

        encoded_ids = replace_pair_once(encoded_ids, best_pair, new_token_id)
        merge_rules.append(
            {
                "pair": [int(best_pair[0]), int(best_pair[1])],
                "token_id": int(new_token_id),
                "count": int(best_count),
            }
        )

    # -------------------------------------------------------------------------
    # Step 4: package the lookup-table tokenizer state
    # -------------------------------------------------------------------------

    return {
        "fps": float(fps),
        "control_fields": list(CONTROL_FIELDS),
        "quantization": {
            "steer_step": float(steer_step),
            "throttle_step": float(throttle_step),
            "brake_step": float(brake_step),
        },
        "token_lookup": token_lookup,
        "merge_rules": merge_rules,
        "encoded_ids": [int(token_id) for token_id in encoded_ids],
        "base_vocab_size": int(len(base_text_to_id)),
        "total_vocab_size": int(len(token_lookup)),
    }


def encode_control_frames(control_frames, tokenizer_state, append_end=True):
    """
    Encode raw GTA control frames into token ids with the learned merge rules.

    If append_end is True, a TOKEN_END is appended at the end of the
    sequence to mark the session boundary.
    """

    token_lookup = {}
    for token_id, entry in tokenizer_state["token_lookup"].items():
        token_lookup[int(token_id)] = entry

    quantization = tokenizer_state["quantization"]
    base_text_to_id = {}
    for token_id, entry in token_lookup.items():
        if entry["frame_count"] == 1:
            base_text_to_id[entry["token_text"]] = token_id

    encoded_ids = []
    for frame in control_frames:
        base_frame = quantize_control_frame(
            frame,
            steer_step=quantization["steer_step"],
            throttle_step=quantization["throttle_step"],
            brake_step=quantization["brake_step"],
        )
        base_frame_bins = control_frame_to_bins(
            base_frame,
            steer_step=quantization["steer_step"],
            throttle_step=quantization["throttle_step"],
            brake_step=quantization["brake_step"],
        )
        token_text = frame_bins_to_token_text(base_frame_bins)

        if token_text not in base_text_to_id:
            raise KeyError(f"unknown base control token: {token_text}")

        encoded_ids.append(base_text_to_id[token_text])

    if append_end:
        encoded_ids.append(TOKEN_END)

    for merge_rule in tokenizer_state["merge_rules"]:
        pair = (int(merge_rule["pair"][0]), int(merge_rule["pair"][1]))
        new_token_id = int(merge_rule["token_id"])
        encoded_ids = replace_pair_once(encoded_ids, pair, new_token_id)

    return encoded_ids


def decode_token_ids(token_ids, tokenizer_state):
    """
    Decode token ids back into the raw GTA control chunk frames.

    TOKEN_END tokens are skipped during decoding (they carry no control
    data, they only mark session boundaries).
    """

    token_lookup = {}
    for token_id, entry in tokenizer_state["token_lookup"].items():
        token_lookup[int(token_id)] = entry

    decoded_frames = []

    for token_id in token_ids:
        if int(token_id) == TOKEN_END:
            continue

        if int(token_id) not in token_lookup:
            raise KeyError(f"unknown token id: {token_id}")

        entry = token_lookup[int(token_id)]
        decoded_frames.extend(entry["raw_control_chunk"])

    return decoded_frames


def save_tokenizer(tokenizer_state, output_path):
    """
    Save the tokenizer lookup table to JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_ready = dict(tokenizer_state)
    json_ready["token_lookup"] = {}
    quantization = tokenizer_state["quantization"]

    for token_id, entry in tokenizer_state["token_lookup"].items():
        json_entry = dict(entry)
        json_entry["raw_control_chunk_bins"] = pack_raw_control_chunk(
            entry["raw_control_chunk"],
            quantization,
        )
        json_entry.pop("raw_control_chunk", None)
        json_ready["token_lookup"][str(token_id)] = json_entry

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(json_ready, file, indent=2)


def save_token_chunks(tokenizer_state, output_path):
    """
    Save a token-chunk-only file for easy inspection.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = make_token_chunks_payload(tokenizer_state)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_tokenizer_snapshot(tokenizer_state, snapshot_dir, snapshot_name=None):
    """
    Save one timestamped tokenizer snapshot so earlier token sets are preserved.
    """

    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    if snapshot_name is None:
        snapshot_name = time.strftime("tokenizer_%Y%m%d_%H%M%S")

    lookup_path = snapshot_dir / f"{snapshot_name}_lookup.json"
    chunks_path = snapshot_dir / f"{snapshot_name}_chunks.json"
    summary_path = snapshot_dir / f"{snapshot_name}_summary.json"

    save_tokenizer(tokenizer_state, lookup_path)
    save_token_chunks(tokenizer_state, chunks_path)

    summary_payload = summarize_tokenizer(tokenizer_state)
    summary_payload["snapshot_name"] = snapshot_name
    summary_payload["lookup_path"] = str(lookup_path)
    summary_payload["chunks_path"] = str(chunks_path)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary_payload, file, indent=2)

    return {
        "snapshot_name": snapshot_name,
        "lookup_path": lookup_path,
        "chunks_path": chunks_path,
        "summary_path": summary_path,
    }


def load_tokenizer(input_path):
    """
    Load a tokenizer lookup table from JSON.
    """

    with open(input_path, "r", encoding="utf-8") as file:
        tokenizer_state = json.load(file)

    quantization = tokenizer_state["quantization"]
    token_lookup = {}
    for token_id, entry in tokenizer_state["token_lookup"].items():
        if "raw_control_chunk" not in entry and "raw_control_chunk_bins" in entry:
            entry["raw_control_chunk"] = unpack_raw_control_chunk(
                entry["raw_control_chunk_bins"],
                quantization,
            )
        token_lookup[int(token_id)] = entry
    tokenizer_state["token_lookup"] = token_lookup

    return tokenizer_state


def get_token_chunk(tokenizer_state, token_id):
    """
    Return the chunk metadata for one token id.
    """

    lookup = tokenizer_state["token_lookup"]
    normalized_token_id = int(token_id)

    if normalized_token_id not in lookup:
        raise KeyError(f"unknown token id: {token_id}")

    return lookup[normalized_token_id]


def summarize_tokenizer(tokenizer_state):
    """
    Build a short printable tokenizer summary.
    """

    max_frames = 0
    longest_token_id = None

    for token_id, entry in tokenizer_state["token_lookup"].items():
        if entry["frame_count"] > max_frames:
            max_frames = entry["frame_count"]
            longest_token_id = token_id

    return {
        "base_vocab_size": int(tokenizer_state["base_vocab_size"]),
        "total_vocab_size": int(tokenizer_state["total_vocab_size"]),
        "merge_count": int(len(tokenizer_state["merge_rules"])),
        "encoded_length": int(len(tokenizer_state["encoded_ids"])),
        "longest_token_id": int(longest_token_id) if longest_token_id is not None else None,
        "longest_token_frames": int(max_frames),
    }


def run_capture(args):
    """
    Capture raw controls from the live game into a compact JSONL file.

    Run this while playing.  Each line is one control frame (~80 bytes)
    instead of the full 660 KB state dump.  Press Ctrl-C to stop.
    """

    # Late import so the tokenizer module stays usable without gta_stream
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gta_stream"))
    import queue as _queue
    from game_state_reader import GameStateReader, SCALAR_FIELDS

    capture_dir = Path(args.capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)

    # Show existing capture history
    existing = sorted(capture_dir.glob("*.jsonl"))
    if existing:
        total_lines = 0
        for p in existing:
            with open(p, "r", encoding="utf-8") as fh:
                total_lines += sum(1 for ln in fh if ln.strip() and not ln.startswith("#"))
        total_duration_s = total_lines / max(1.0, args.fps)
        print(f"Existing captures: {len(existing)} session(s), "
              f"{total_lines} frames ({total_duration_s / 3600:.2f}h)")
    else:
        print("No existing captures found.  Starting fresh.")

    session_name = time.strftime("session_%Y%m%d_%H%M%S")
    capture_path = capture_dir / f"{session_name}.jsonl"

    reader = GameStateReader(fields=SCALAR_FIELDS).start()

    frame_count = 0
    last_print = 0.0
    print_every = 10.0

    print(f"\nCapturing controls to {capture_path}")
    print("Play the game normally.  Press Ctrl-C to stop.\n")

    try:
        with open(capture_path, "w", encoding="utf-8") as out:
            out.write(f"# session {session_name}  fps={args.fps}\n")

            while True:
                try:
                    state = reader.get(timeout=5.0)
                except _queue.Empty:
                    continue

                raw = state.raw
                steer = float(raw.get("u_steer", 0.0) or 0.0)
                frame = {
                    "right_steer": round(max(0.0, steer), 4),
                    "left_steer": round(max(0.0, -steer), 4),
                    "forward_throttle": round(float(raw.get("u_throttle", 0.0) or 0.0), 4),
                    "brake": round(float(raw.get("u_brake", 0.0) or 0.0), 4),
                    "speed": round(float(raw.get("v_speed", 0.0) or 0.0), 2),
                    "heading": round(float(raw.get("v_heading", 0.0) or 0.0), 2),
                    "road_heading": round(float(raw.get("road_heading", 0.0) or 0.0), 2),
                    "road_dist": round(float(raw.get("road_dist", 0.0) or 0.0), 2),
                    "wp_dist": round(float(raw.get("wp_dist", 0.0) or 0.0), 2),
                    "wp_x": round(float(raw.get("wp_x", 0.0) or 0.0), 1),
                    "wp_y": round(float(raw.get("wp_y", 0.0) or 0.0), 1),
                    "pos_x": round(float(raw.get("px", 0.0) or 0.0), 1),
                    "pos_y": round(float(raw.get("py", 0.0) or 0.0), 1),
                }
                out.write(json.dumps(frame, separators=(",", ":")) + "\n")
                frame_count += 1

                # Flush every 100 frames so data is safe on crash
                if frame_count % 100 == 0:
                    out.flush()

                now_t = time.time()
                if now_t - last_print >= print_every:
                    last_print = now_t
                    duration_s = frame_count / max(1.0, args.fps)
                    print(
                        f"  frames={frame_count}"
                        f"  duration={duration_s:.1f}s"
                        f"  file_size={os.path.getsize(capture_path) / 1024:.0f}KB"
                    )

    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()

    duration_s = frame_count / max(1.0, args.fps)
    file_kb = os.path.getsize(capture_path) / 1024
    print(f"\nCapture complete: {frame_count} frames ({duration_s:.1f}s at {args.fps}Hz)")
    print(f"Saved to {capture_path} ({file_kb:.0f} KB)")


def run_build(args):
    """
    Build (or resume) the BPE tokenizer from captured control data.

    Run this after playing.  Reads all .jsonl captures from the capture
    directory, optionally also reads legacy stats/ frames, and runs BPE.
    """

    training_data = Path(__file__).resolve().parent / "training_data"

    capture_dir = Path(args.capture_dir)
    output_path = Path(args.output)
    chunk_output = Path(args.chunk_output)
    snapshot_dir = Path(args.snapshot_dir)

    # Load controls from captures as separate sessions
    sessions = []
    if capture_dir.is_dir():
        jsonl_files = sorted(capture_dir.glob("*.jsonl"))
        if jsonl_files:
            print(f"Loading {len(jsonl_files)} capture file(s) from {capture_dir}")
            sessions = load_all_captures(capture_dir)
            total_capture_frames = sum(len(s) for s in sessions)
            print(f"  {total_capture_frames} control frames from {len(sessions)} session(s)")

    # Also load legacy stats/ frames if they exist (treated as one session)
    stats_dir = Path(args.input_dir)
    if stats_dir.is_dir() and any(stats_dir.glob("*.json")):
        print(f"Loading legacy frames from {stats_dir}")
        legacy_frames = load_gta_control_frames(stats_dir)
        print(f"  {len(legacy_frames)} control frames from legacy stats")
        if legacy_frames:
            sessions.insert(0, legacy_frames)

    if not sessions:
        print("No control data found.  Run 'capture' first or provide --input-dir.")
        sys.exit(1)

    total_frames = sum(len(s) for s in sessions)
    duration_s = total_frames / max(1.0, args.fps)
    print(f"Total: {total_frames} frames in {len(sessions)} session(s) ({duration_s:.1f}s / {duration_s / 3600:.2f}h)")

    existing_tokenizer_state = None
    if output_path.exists():
        existing_tokenizer_state = load_tokenizer(output_path)

    tokenizer_state = build_control_tokenizer(
        control_frames=None,
        fps=args.fps,
        max_merges=args.max_merges,
        min_pair_count=args.min_pair_count,
        steer_step=args.steer_step,
        throttle_step=args.throttle_step,
        brake_step=args.brake_step,
        existing_tokenizer_state=existing_tokenizer_state,
        sessions=sessions,
    )

    save_tokenizer(tokenizer_state, output_path)
    save_token_chunks(tokenizer_state, chunk_output)
    snapshot_info = save_tokenizer_snapshot(tokenizer_state, snapshot_dir)

    summary = summarize_tokenizer(tokenizer_state)
    if existing_tokenizer_state is None:
        print("\nbuild_mode: fresh")
    else:
        print("\nbuild_mode: resume")
        print(f"resumed_token_count: {existing_tokenizer_state['total_vocab_size']}")
        print(f"resumed_merge_count: {len(existing_tokenizer_state['merge_rules'])}")
    print(f"saved tokenizer: {output_path}")
    print(f"saved token chunks: {chunk_output}")
    print(f"saved tokenizer snapshot: {snapshot_info['lookup_path']}")
    print(f"saved chunk snapshot: {snapshot_info['chunks_path']}")
    print(f"base_vocab_size: {summary['base_vocab_size']}")
    print(f"total_vocab_size: {summary['total_vocab_size']}")
    print(f"merge_count: {summary['merge_count']}")
    print(f"encoded_length: {summary['encoded_length']}")
    print(f"longest_token_id: {summary['longest_token_id']}")
    print(f"longest_token_frames: {summary['longest_token_frames']}")


def main():
    training_data = Path(__file__).resolve().parent / "training_data"
    default_capture_dir = str(training_data / "captures")

    parser = argparse.ArgumentParser(
        description="GTA control tokenizer — capture controls then build BPE vocabulary",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── capture ──────────────────────────────────────────────────────────
    cap = subparsers.add_parser("capture", help="Capture controls from live game")
    cap.add_argument("--capture-dir", default=default_capture_dir)
    cap.add_argument("--fps", type=float, default=20.0)

    # ── build ────────────────────────────────────────────────────────────
    bld = subparsers.add_parser("build", help="Build BPE tokenizer from captured data")
    bld.add_argument("--capture-dir", default=default_capture_dir)
    bld.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parent.parent / "gta_stream" / "stats"),
    )
    bld.add_argument(
        "--output",
        default=str(training_data / "tokenizer_lookup.json"),
    )
    bld.add_argument(
        "--chunk-output",
        default=str(training_data / "tokenizer_chunks.json"),
    )
    bld.add_argument(
        "--snapshot-dir",
        default=str(training_data / "snapshots"),
    )
    bld.add_argument("--fps", type=float, default=20.0)
    bld.add_argument("--max-merges", type=int, default=512)
    bld.add_argument("--min-pair-count", type=int, default=5)
    bld.add_argument("--steer-step", type=float, default=DEFAULT_STEP)
    bld.add_argument("--throttle-step", type=float, default=DEFAULT_STEP)
    bld.add_argument("--brake-step", type=float, default=DEFAULT_STEP)

    args = parser.parse_args()

    if args.command == "capture":
        run_capture(args)
    elif args.command == "build":
        run_build(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
