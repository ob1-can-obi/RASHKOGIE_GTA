"""
Preprocess raw JSONL training data into compact .pt tensor files.

Raw JSONL records are ~370KB each (verbose JSON, nearby vehicle lists, etc.)
but training only uses ~3.4KB of numeric fields per state. This script
extracts exactly what build_state_tensors() and extract_reward_features()
need and saves pre-built tensors to disk.

Result: 88GB JSONL → ~800MB .pt files that load into RAM in seconds.

Usage:
    python preprocess_data.py                    # process both main_model and reward_head
    python preprocess_data.py --module main      # only main_model
    python preprocess_data.py --module reward     # only reward_head
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "main_model"))
sys.path.insert(0, str(_ROOT / "reward_head"))

from main_model.main_model import build_state_tensors
from reward_head.reward_head import extract_reward_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def preprocess_main_model(data_dir=None, output_path=None):
    """Convert main_model JSONL → single .pt file with pre-built tensors."""
    if data_dir is None:
        data_dir = _ROOT / "main_model" / "training_data"
    else:
        data_dir = Path(data_dir)

    if output_path is None:
        output_path = data_dir / "preprocessed.pt"
    else:
        output_path = Path(output_path)

    ego_t_list, scene_t_list, route_t_list, entities_t_list, mask_t_list = [], [], [], [], []
    ego_t1_list, scene_t1_list, route_t1_list, entities_t1_list, mask_t1_list = [], [], [], [], []
    token_ids = []

    total = 0
    skipped = 0

    for path in sorted(data_dir.glob("*.jsonl")):
        logging.info("Processing %s ...", path.name)
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                t_t = build_state_tensors(record["state_t"])
                t_t1 = build_state_tensors(record["state_t1"])

                ego_t_list.append(t_t["ego"])
                scene_t_list.append(t_t["scene"])
                route_t_list.append(t_t["route"])
                entities_t_list.append(t_t["entities"])
                mask_t_list.append(t_t["mask"])

                ego_t1_list.append(t_t1["ego"])
                scene_t1_list.append(t_t1["scene"])
                route_t1_list.append(t_t1["route"])
                entities_t1_list.append(t_t1["entities"])
                mask_t1_list.append(t_t1["mask"])

                token_ids.append(record["token_id"])
                total += 1

                if total % 10000 == 0:
                    logging.info("  %d records processed...", total)

    if total == 0:
        logging.warning("No records found in %s", data_dir)
        return

    logging.info("Stacking %d records into tensors...", total)
    data = {
        "ego_t": torch.cat(ego_t_list, dim=0),           # [N, 46]
        "scene_t": torch.cat(scene_t_list, dim=0),       # [N, 16]
        "route_t": torch.cat(route_t_list, dim=0),       # [N, 14]
        "entities_t": torch.cat(entities_t_list, dim=0), # [N, 32, 24]
        "mask_t": torch.cat(mask_t_list, dim=0),         # [N, 32]
        "ego_t1": torch.cat(ego_t1_list, dim=0),
        "scene_t1": torch.cat(scene_t1_list, dim=0),
        "route_t1": torch.cat(route_t1_list, dim=0),
        "entities_t1": torch.cat(entities_t1_list, dim=0),
        "mask_t1": torch.cat(mask_t1_list, dim=0),
        "token_ids": torch.tensor(token_ids, dtype=torch.long),  # [N]
    }

    logging.info("Saving to %s ...", output_path)
    torch.save(data, output_path)
    size_mb = output_path.stat().st_size / 1e6
    logging.info("Done. %d records, %.1f MB, skipped %d malformed lines", total, size_mb, skipped)


def preprocess_reward_head(data_dir=None, output_path=None):
    """Convert reward_head JSONL → single .pt file with pre-built tensors."""
    if data_dir is None:
        data_dir = _ROOT / "reward_head" / "training_data"
    else:
        data_dir = Path(data_dir)

    if output_path is None:
        output_path = data_dir / "preprocessed.pt"
    else:
        output_path = Path(output_path)

    ego_before_list, scene_before_list, route_before_list = [], [], []
    entities_before_list, mask_before_list = [], []
    ego_after_list, scene_after_list, route_after_list = [], [], []
    entities_after_list, mask_after_list = [], []
    rf_before_list, rf_after_list = [], []
    durations, returns = [], []

    total = 0
    skipped = 0

    for path in sorted(data_dir.glob("*.jsonl")):
        logging.info("Processing %s ...", path.name)
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                t_before = build_state_tensors(record["state_before"])
                t_after = build_state_tensors(record["state_after"])

                ego_before_list.append(t_before["ego"])
                scene_before_list.append(t_before["scene"])
                route_before_list.append(t_before["route"])
                entities_before_list.append(t_before["entities"])
                mask_before_list.append(t_before["mask"])

                ego_after_list.append(t_after["ego"])
                scene_after_list.append(t_after["scene"])
                route_after_list.append(t_after["route"])
                entities_after_list.append(t_after["entities"])
                mask_after_list.append(t_after["mask"])

                rf_before_list.append(extract_reward_features(record["state_before"]))
                rf_after_list.append(extract_reward_features(record["state_after"]))

                durations.append(record["duration"])
                returns.append(record["realized_return"])
                total += 1

                if total % 10000 == 0:
                    logging.info("  %d records processed...", total)

    if total == 0:
        logging.warning("No records found in %s", data_dir)
        return

    logging.info("Stacking %d records into tensors...", total)
    data = {
        "ego_before": torch.cat(ego_before_list, dim=0),
        "scene_before": torch.cat(scene_before_list, dim=0),
        "route_before": torch.cat(route_before_list, dim=0),
        "entities_before": torch.cat(entities_before_list, dim=0),
        "mask_before": torch.cat(mask_before_list, dim=0),
        "ego_after": torch.cat(ego_after_list, dim=0),
        "scene_after": torch.cat(scene_after_list, dim=0),
        "route_after": torch.cat(route_after_list, dim=0),
        "entities_after": torch.cat(entities_after_list, dim=0),
        "mask_after": torch.cat(mask_after_list, dim=0),
        "rf_before": torch.cat(rf_before_list, dim=0),   # [N, 6]
        "rf_after": torch.cat(rf_after_list, dim=0),     # [N, 6]
        "durations": torch.tensor(durations, dtype=torch.float32),
        "returns": torch.tensor(returns, dtype=torch.float32),
    }

    logging.info("Saving to %s ...", output_path)
    torch.save(data, output_path)
    size_mb = output_path.stat().st_size / 1e6
    logging.info("Done. %d records, %.1f MB, skipped %d malformed lines", total, size_mb, skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess JSONL training data to compact .pt")
    parser.add_argument("--module", choices=["main", "reward", "both"], default="both")
    args = parser.parse_args()

    if args.module in ("main", "both"):
        preprocess_main_model()
    if args.module in ("reward", "both"):
        preprocess_reward_head()
