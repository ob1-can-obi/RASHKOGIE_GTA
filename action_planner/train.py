"""
Stage 3 training script: Action planner imitation learning from player demonstrations.

Trains planner_mlp via cross-entropy loss against player token labels.
The encoder and intuition head are FROZEN feature extractors -- their outputs
are DETACHED so no gradients flow back to encoder or intuition weights.

Usage:
    python action_planner/train.py
    python action_planner/train.py --data-dir path/to/data --max-epochs 50
    python action_planner/train.py --resume path/to/session_dir
    python action_planner/train.py --encoder-checkpoint path/to/enc --intuition-checkpoint path/to/int

Training data format (JSONL, one record per line):
    {"state": {...}, "token_id": int, "session_ts": str, "frame_idx": int}

On convergence:
    - Saves final checkpoint to action_planner/checkpoints/
    - Auto-updates training_status.json (fixes WARNING 2)

Primary metric: Top-1 accuracy (target > 60%) per D-09, D-11
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Sibling module resolution (same pattern as main_model/train.py)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
for _d in ("main_model", "intuition_head", "action_planner"):
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main_model import create_encoder_weights, encode_state
from intuition_head import intuition_head
from action_planner import action_planner
from training_utils import (
    ConvergenceDetector,
    load_training_config,
    update_training_status,
)


# ---------------------------------------------------------------------------
# JSONL data loading (T-04-10: try/except per line, skip malformed)
# ---------------------------------------------------------------------------

def load_data(data_dir):
    """
    Load all JSONL records from a training data directory.

    Iterates sorted .jsonl files, parses one JSON object per line.
    Strips whitespace (handles CRLF), skips empty lines and comments.
    Malformed lines are logged and skipped (T-04-10 mitigation).

    Args:
        data_dir: Path to directory containing .jsonl files

    Returns:
        list[dict]: Parsed training records, each with keys
                    state, token_id, session_ts, frame_idx
    """
    data_dir = Path(data_dir)
    records = []
    for path in sorted(data_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    logging.warning(
                        "Skipping malformed line %d in %s", line_number, path.name
                    )
    return records


# ---------------------------------------------------------------------------
# Preprocessing (Pitfall 4: filter records without token_id)
# ---------------------------------------------------------------------------

def preprocess_data(records):
    """
    Filter out records where token_id is None (raw captures not yet tokenized).

    Per Pitfall 4 and D-06: capture mode saves raw states + player_controls.
    Preprocessing converts player_controls to token_id. Records where
    token_id is still None have not been tokenized and must be skipped.

    Args:
        records: List of dicts from load_data()

    Returns:
        list[dict]: Filtered records where token_id is a valid integer
    """
    filtered = [r for r in records if r.get("token_id") is not None]
    removed = len(records) - len(filtered)
    if removed > 0:
        logging.warning(
            "Filtered %d records with token_id=None (not yet tokenized)", removed
        )
    return filtered


# ---------------------------------------------------------------------------
# Checkpoint save/load (T-04-11, T-04-12: weights_only=True, map_location="cpu")
# ---------------------------------------------------------------------------

def save_planner_checkpoint(checkpoint_dir, planner_mlp, optimizer, step_count):
    """
    Save a training checkpoint for Stage 3 (action planner).

    Creates a session directory under checkpoint_dir with separate files
    for planner_mlp and optimizer state.

    Args:
        checkpoint_dir: Base checkpoint directory
        planner_mlp: Action planner MLP (nn.Sequential)
        optimizer: Adam optimizer
        step_count: Current training step

    Returns:
        Path: Session directory containing the saved checkpoint files
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(checkpoint_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save planner_mlp
    torch.save(
        {"model_state_dict": planner_mlp.state_dict(), "step_count": step_count},
        session_dir / "planner_mlp.pt",
    )

    # Save optimizer
    torch.save(
        {"optimizer_state_dict": optimizer.state_dict(), "step_count": step_count},
        session_dir / "optimizer.pt",
    )

    return session_dir


def load_planner_checkpoint(session_dir, planner_mlp, optimizer):
    """
    Restore training state from a saved checkpoint.

    All torch.load calls use weights_only=True and map_location="cpu"
    per T-04-11 and T-04-12 threat mitigation.

    Args:
        session_dir: Path to session directory containing checkpoint files
        planner_mlp: Action planner MLP to restore into
        optimizer: Optimizer to restore into

    Returns:
        int: Restored step_count
    """
    session_dir = Path(session_dir)

    # Restore planner_mlp
    planner_ckpt = torch.load(
        session_dir / "planner_mlp.pt", weights_only=True, map_location="cpu"
    )
    planner_mlp.load_state_dict(planner_ckpt["model_state_dict"])

    # Restore optimizer + step count
    opt_ckpt = torch.load(
        session_dir / "optimizer.pt", weights_only=True, map_location="cpu"
    )
    optimizer.load_state_dict(opt_ckpt["optimizer_state_dict"])
    step_count = opt_ckpt["step_count"]

    return step_count


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_action_planner_imitation(
    data_dir=None,
    config_path=None,
    checkpoint_dir=None,
    resume_from=None,
    encoder_checkpoint=None,
    intuition_checkpoint=None,
    vocab_size=874,
    max_epochs=100,
):
    """
    Stage 3: Action planner imitation learning from player driving demonstrations.

    Reads JSONL demonstration data (states + player token labels), uses FROZEN
    encoder + intuition head to compute z_t and z_next_pred, trains planner_mlp
    via cross-entropy loss against player token labels, and tracks top-1 accuracy
    as the convergence metric (target > 60%).

    The encoder and intuition head are frozen feature extractors -- their outputs
    are detached so only planner_mlp receives gradient updates.

    Args:
        data_dir: Path to JSONL training data. Defaults to action_planner/training_data/
        config_path: Path to training_config.json. Defaults to project root.
        checkpoint_dir: Path to save checkpoints. Defaults to action_planner/checkpoints/
        resume_from: Path to session directory to resume from. None = fresh start.
        encoder_checkpoint: Path to encoder checkpoint from Stage 1 (.pt file or
                           session directory). None = use fresh random encoder weights.
        intuition_checkpoint: Path to intuition head checkpoint from Stage 1 (.pt file
                             or session directory). None = use fresh random weights.
        vocab_size: Token vocabulary size (default 874)
        max_epochs: Maximum training epochs (default 100)

    Returns:
        dict: Training result with keys:
            converged (bool), final_accuracy (float), total_steps (int)
    """
    # Defaults
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "training_data"
    else:
        data_dir = Path(data_dir)

    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    else:
        checkpoint_dir = Path(checkpoint_dir)

    # Load config
    config_full = load_training_config(config_path)
    config = config_full["action_planner"]

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    # -----------------------------------------------------------------------
    # Initialize encoder (frozen feature extractor)
    # -----------------------------------------------------------------------

    encoder_weights = create_encoder_weights()

    # If encoder checkpoint is provided, load pre-trained encoder weights
    if encoder_checkpoint is not None:
        encoder_ckpt_path = Path(encoder_checkpoint)
        if encoder_ckpt_path.is_dir():
            # Session directory -- load encoder_weights.pt from it
            ckpt_file = encoder_ckpt_path / "encoder_weights.pt"
        else:
            # Direct .pt file
            ckpt_file = encoder_ckpt_path

        encoder_state = torch.load(
            ckpt_file, weights_only=True, map_location="cpu"
        )
        for key in encoder_state:
            if key in encoder_weights and hasattr(encoder_weights[key], "load_state_dict"):
                encoder_weights[key].load_state_dict(encoder_state[key])
        logging.info("Loaded encoder checkpoint from %s", ckpt_file)

    # -----------------------------------------------------------------------
    # Initialize intuition head via dummy forward pass (lazy init pattern)
    # -----------------------------------------------------------------------

    _dummy_z = torch.randn(1, 128)
    _dummy_token = torch.tensor([0])
    _, _, token_embed, intuition_mlp = intuition_head(
        _dummy_z, _dummy_token, vocab_size
    )

    # If intuition checkpoint is provided, load pre-trained weights
    if intuition_checkpoint is not None:
        intuition_ckpt_path = Path(intuition_checkpoint)
        if intuition_ckpt_path.is_dir():
            # Session directory -- load individual files
            intuition_mlp_file = intuition_ckpt_path / "intuition_mlp.pt"
            token_embed_file = intuition_ckpt_path / "token_embed.pt"
        else:
            # Direct .pt file -- assume it contains intuition_mlp state
            intuition_mlp_file = intuition_ckpt_path
            token_embed_file = None

        if intuition_mlp_file.exists():
            ckpt = torch.load(
                intuition_mlp_file, weights_only=True, map_location="cpu"
            )
            intuition_mlp.load_state_dict(ckpt["model_state_dict"])
            logging.info("Loaded intuition_mlp checkpoint from %s", intuition_mlp_file)

        if token_embed_file is not None and token_embed_file.exists():
            ckpt = torch.load(
                token_embed_file, weights_only=True, map_location="cpu"
            )
            token_embed.load_state_dict(ckpt["model_state_dict"])
            logging.info("Loaded token_embed checkpoint from %s", token_embed_file)

    # -----------------------------------------------------------------------
    # Initialize planner_mlp via dummy forward pass (lazy init pattern)
    # -----------------------------------------------------------------------

    result = action_planner(
        _dummy_z.detach(), _dummy_z.detach(), vocab_size
    )
    planner_mlp = result["planner_mlp"]

    # -----------------------------------------------------------------------
    # Optimizer (planner_mlp params ONLY -- encoder + intuition are frozen)
    # -----------------------------------------------------------------------

    optimizer = Adam(planner_mlp.parameters(), lr=config["lr"], eps=1e-5)

    # -----------------------------------------------------------------------
    # Convergence detection (D-11: dual criteria, mode="max" for accuracy)
    # -----------------------------------------------------------------------

    conv_cfg = config["convergence"]
    convergence = ConvergenceDetector(
        threshold=conv_cfg["threshold"],
        patience=conv_cfg["patience"],
        mode=conv_cfg["mode"],
    )

    # -----------------------------------------------------------------------
    # Load and preprocess data
    # -----------------------------------------------------------------------

    records = load_data(data_dir)
    if not records:
        logging.warning("No training records found in %s", data_dir)
        return {"converged": False, "final_accuracy": None, "total_steps": 0}

    records = preprocess_data(records)
    if not records:
        logging.warning("No valid records after preprocessing (all token_id=None)")
        return {"converged": False, "final_accuracy": None, "total_steps": 0}

    logging.info(
        "Loaded %d valid training records from %s", len(records), data_dir
    )

    # -----------------------------------------------------------------------
    # Resume from checkpoint if requested
    # -----------------------------------------------------------------------

    step_count = 0
    if resume_from is not None:
        step_count = load_planner_checkpoint(resume_from, planner_mlp, optimizer)
        logging.info("Resumed from checkpoint at step %d", step_count)

    # -----------------------------------------------------------------------
    # Auto-update training_status.json on start (fixes WARNING 2)
    # -----------------------------------------------------------------------

    update_training_status("action_planner", "training")

    # -----------------------------------------------------------------------
    # Training loop (epoch-based, iterate over records in mini-batches)
    # -----------------------------------------------------------------------

    batch_size = config["batch_size"]
    max_grad_norm = config["max_grad_norm"]
    eval_every = config["eval_every_n_steps"]

    for epoch in range(max_epochs):
        random.shuffle(records)
        epoch_correct = 0
        epoch_total = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)
            batch_correct = 0

            for record in batch:
                state = record["state"]
                target_token_id = record["token_id"]

                # Encoder and intuition head are FROZEN -- detach outputs
                with torch.no_grad():
                    z_t = encode_state(state, encoder_weights)
                    # Use token_id=0 (idle) as prev_token for intuition head
                    prev_token = torch.tensor([0])
                    z_next_pred, _, token_embed, intuition_mlp = intuition_head(
                        z_t, prev_token, vocab_size,
                        token_embed=token_embed, intuition_mlp=intuition_mlp,
                    )

                # Action planner forward (trainable)
                result = action_planner(
                    z_t.detach(), z_next_pred.detach(), vocab_size,
                    planner_mlp=planner_mlp,
                )
                planner_mlp = result["planner_mlp"]
                logits = result["logits"]  # [1, vocab_size]

                # Cross-entropy loss against player token label
                target = torch.tensor([target_token_id], dtype=torch.long)
                loss = F.cross_entropy(logits, target)
                total_loss = total_loss + loss

                # Track top-1 accuracy
                predicted_token = logits.argmax(dim=-1).item()
                if predicted_token == target_token_id:
                    batch_correct += 1

            # Batch-mean loss
            total_loss = total_loss / len(batch)
            total_loss.backward()
            grad_norm = clip_grad_norm_(planner_mlp.parameters(), max_grad_norm)
            clipped = bool(grad_norm.item() > max_grad_norm)
            optimizer.step()
            step_count += 1

            epoch_correct += batch_correct
            epoch_total += len(batch)

            # Logging
            if step_count % 10 == 0:
                batch_acc = batch_correct / len(batch)
                logging.info(
                    "step=%d loss=%.4f batch_acc=%.3f grad_norm=%.4f clipped=%s",
                    step_count,
                    total_loss.item(),
                    batch_acc,
                    grad_norm.item(),
                    clipped,
                )

            # Convergence check
            if step_count % eval_every == 0:
                eval_acc = epoch_correct / max(1, epoch_total)
                logging.info(
                    "eval step=%d accuracy=%.3f", step_count, eval_acc
                )
                if convergence.update(eval_acc):
                    logging.info(
                        "CONVERGED at step %d, accuracy=%.3f",
                        step_count, eval_acc,
                    )
                    # Save final checkpoint
                    ckpt_path = save_planner_checkpoint(
                        checkpoint_dir, planner_mlp, optimizer, step_count
                    )
                    # Auto-update training_status.json (fixes WARNING 2)
                    update_training_status(
                        "action_planner",
                        "converged",
                        metric=eval_acc,
                        steps=step_count,
                        checkpoint=str(ckpt_path),
                    )
                    return {
                        "converged": True,
                        "final_accuracy": eval_acc,
                        "total_steps": step_count,
                    }

        epoch_acc = epoch_correct / max(1, epoch_total)
        logging.info("Epoch %d complete, accuracy=%.3f", epoch + 1, epoch_acc)

    # Max epochs reached without convergence
    logging.warning(
        "Max epochs (%d) reached without convergence at step %d",
        max_epochs, step_count,
    )
    # Save checkpoint even if not converged
    ckpt_path = save_planner_checkpoint(
        checkpoint_dir, planner_mlp, optimizer, step_count
    )
    update_training_status(
        "action_planner",
        "training",
        steps=step_count,
        checkpoint=str(ckpt_path),
    )
    return {"converged": False, "final_accuracy": None, "total_steps": step_count}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train action planner via imitation learning (Stage 3)"
    )
    parser.add_argument(
        "--data-dir", default=None, help="Path to training data directory"
    )
    parser.add_argument(
        "--config", default=None, help="Path to training_config.json"
    )
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--resume", default=None, help="Path to session directory to resume from"
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help="Path to encoder checkpoint from Stage 1",
    )
    parser.add_argument(
        "--intuition-checkpoint",
        default=None,
        help="Path to intuition head checkpoint from Stage 1",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=874)
    args = parser.parse_args()

    result = train_action_planner_imitation(
        data_dir=args.data_dir,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        encoder_checkpoint=args.encoder_checkpoint,
        intuition_checkpoint=args.intuition_checkpoint,
        vocab_size=args.vocab_size,
        max_epochs=args.max_epochs,
    )
    print(json.dumps(result, indent=2))
