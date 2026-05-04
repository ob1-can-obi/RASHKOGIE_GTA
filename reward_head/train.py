"""
Stage 2 training script: Reward head offline training from JSONL data.

Trains reward_mlp and rf_predictor on saved state pairs + realized returns.
The encoder is used as a frozen feature extractor -- its outputs are DETACHED
so no gradients flow back to encoder weights (Pitfall 2).

Usage:
    python reward_head/train.py
    python reward_head/train.py --data-dir path/to/data --max-epochs 50
    python reward_head/train.py --resume path/to/session_dir
    python reward_head/train.py --encoder-checkpoint path/to/encoder_ckpt

Training data format (JSONL, one record per line):
    {"state_before": {...}, "state_after": {...}, "duration": int,
     "realized_return": float}

On convergence:
    - Saves final checkpoint to reward_head/checkpoints/
    - Freezes reward_mlp and rf_predictor (D-12, fixes WARNING 1)
    - Auto-updates training_status.json (fixes WARNING 2)
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

# ---------------------------------------------------------------------------
# Sibling module resolution (same pattern as main_model/train.py)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
for _d in ("main_model", "reward_head"):
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main_model import create_encoder_weights, encode_state, encode_tensors
from reward_head import reward_head, extract_reward_features, predict_reward_features, RF_DIM
from training_utils import (
    ConvergenceDetector,
    StreamingJSONLDataset,
    freeze_module,
    get_device,
    load_training_config,
    to_device,
    update_training_status,
)


# ---------------------------------------------------------------------------
# JSONL data loading -- uses StreamingJSONLDataset (lazy, RAM-safe)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Checkpoint save/load (T-04-08, T-04-09: weights_only=True, map_location="cpu")
# ---------------------------------------------------------------------------

def save_reward_checkpoint(checkpoint_dir, reward_mlp, rf_predictor, optimizer, step_count):
    """
    Save a training checkpoint for Stage 2 (reward head).

    Creates a session directory under checkpoint_dir with separate files
    for reward_mlp, rf_predictor, and optimizer state.

    Args:
        checkpoint_dir: Base checkpoint directory
        reward_mlp: Reward head MLP (nn.Sequential)
        rf_predictor: Reward feature predictor MLP (nn.Sequential)
        optimizer: Adam optimizer
        step_count: Current training step

    Returns:
        Path: Session directory containing the saved checkpoint files
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(checkpoint_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save reward_mlp
    torch.save(
        {"model_state_dict": reward_mlp.state_dict(), "step_count": step_count},
        session_dir / "reward_mlp.pt",
    )

    # Save rf_predictor
    torch.save(
        {"model_state_dict": rf_predictor.state_dict(), "step_count": step_count},
        session_dir / "rf_predictor.pt",
    )

    # Save optimizer
    torch.save(
        {"optimizer_state_dict": optimizer.state_dict(), "step_count": step_count},
        session_dir / "optimizer.pt",
    )

    return session_dir


def load_reward_checkpoint(session_dir, reward_mlp, rf_predictor, optimizer):
    """
    Restore training state from a saved checkpoint.

    All torch.load calls use weights_only=True and map_location="cpu"
    per T-04-08 and T-04-09 threat mitigation.

    Args:
        session_dir: Path to session directory containing checkpoint files
        reward_mlp: Reward head MLP to restore into
        rf_predictor: Reward feature predictor MLP to restore into
        optimizer: Optimizer to restore into

    Returns:
        int: Restored step_count
    """
    session_dir = Path(session_dir)

    # Restore reward_mlp
    reward_ckpt = torch.load(
        session_dir / "reward_mlp.pt", weights_only=True, map_location="cpu"
    )
    reward_mlp.load_state_dict(reward_ckpt["model_state_dict"])

    # Restore rf_predictor
    rf_ckpt = torch.load(
        session_dir / "rf_predictor.pt", weights_only=True, map_location="cpu"
    )
    rf_predictor.load_state_dict(rf_ckpt["model_state_dict"])

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

def train_reward_head_offline(
    data_dir=None,
    config_path=None,
    checkpoint_dir=None,
    resume_from=None,
    encoder_checkpoint=None,
    max_epochs=100,
):
    """
    Stage 2: Reward head offline training from saved JSONL data.

    Reads state pairs + realized returns, computes embeddings through a
    FROZEN encoder (detached outputs per Pitfall 2), trains reward_mlp and
    rf_predictor via combined MSE loss, applies convergence detection with
    checkpoint support, and freezes reward modules on convergence (fixes WARNING 1).

    The encoder is used as a frozen feature extractor -- its outputs are
    detached so no gradients flow back to encoder weights. Only reward_mlp
    and rf_predictor receive gradient updates.

    Args:
        data_dir: Path to JSONL training data. Defaults to reward_head/training_data/
        config_path: Path to training_config.json. Defaults to project root.
        checkpoint_dir: Path to save checkpoints. Defaults to reward_head/checkpoints/
        resume_from: Path to session directory to resume from. None = fresh start.
        encoder_checkpoint: Path to encoder checkpoint from Stage 1 (.pt file).
                           None = use fresh random encoder weights.
        max_epochs: Maximum training epochs (default 100)

    Returns:
        dict: Training result with keys:
            converged (bool), final_mse (float), total_steps (int)
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
    config = config_full["reward_head"]

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    # -----------------------------------------------------------------------
    # Device selection (CUDA when available)
    # -----------------------------------------------------------------------

    device = get_device()
    logging.info("Training device: %s", device)

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
    # Initialize reward modules via dummy forward pass (lazy init pattern)
    # -----------------------------------------------------------------------

    to_device(encoder_weights, device)

    _dummy_state = {
        "near_entities": [], "near_vehs": [], "near_peds": [], "near_objects": [],
    }
    _dummy_z = encode_state(_dummy_state, encoder_weights).detach()
    _dummy_rf = extract_reward_features(_dummy_state).to(device)
    _dummy_delta = torch.zeros_like(_dummy_z)
    fused_dim = _dummy_z.shape[-1]

    # Initialize rf_predictor
    _, rf_predictor = predict_reward_features(
        z_parent=_dummy_z,
        delta_z=_dummy_delta,
        rf_parent=_dummy_rf,
        rf_predictor=None,
        fused_dim=fused_dim,
    )

    # Initialize reward_mlp
    _dummy_duration = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    _dummy_time_left = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    _, reward_mlp = reward_head(
        z_parent=_dummy_z,
        z_child=_dummy_z,
        rf_parent=_dummy_rf,
        rf_child=_dummy_rf,
        duration=_dummy_duration,
        time_left=_dummy_time_left,
        reward_mlp=None,
        fused_dim=fused_dim,
    )

    reward_mlp = reward_mlp.to(device)
    rf_predictor = rf_predictor.to(device)

    # -----------------------------------------------------------------------
    # Optimizer (reward_mlp + rf_predictor only, NOT encoder)
    # -----------------------------------------------------------------------

    all_params = list(reward_mlp.parameters()) + list(rf_predictor.parameters())
    optimizer = Adam(all_params, lr=config["lr"], eps=1e-5)

    # -----------------------------------------------------------------------
    # Convergence detection (D-11: dual criteria)
    # -----------------------------------------------------------------------

    conv_cfg = config["convergence"]
    convergence = ConvergenceDetector(
        threshold=conv_cfg["threshold"],
        patience=conv_cfg["patience"],
        mode=conv_cfg["mode"],
    )

    # -----------------------------------------------------------------------
    # Load data — preprocessed .pt (fast, fits in RAM) or streaming JSONL
    # -----------------------------------------------------------------------

    preprocessed_path = data_dir / "preprocessed.pt"
    use_preprocessed = preprocessed_path.exists()

    if use_preprocessed:
        logging.info("Loading preprocessed data from %s", preprocessed_path)
        pt_data = torch.load(preprocessed_path, weights_only=True, map_location="cpu")
        num_records = pt_data["durations"].shape[0]
        logging.info("Loaded %d preprocessed records", num_records)
    else:
        logging.info("No preprocessed.pt found — using streaming JSONL (slow)")
        logging.info("Run 'python preprocess_data.py --module reward' first for faster training")
        dataset = StreamingJSONLDataset(data_dir)
        num_records = len(dataset)

    if num_records == 0:
        logging.warning("No training records found in %s", data_dir)
        return {"converged": False, "final_mse": None, "total_steps": 0}

    # -----------------------------------------------------------------------
    # Resume from checkpoint if requested
    # -----------------------------------------------------------------------

    step_count = 0
    if resume_from is not None:
        step_count = load_reward_checkpoint(
            resume_from, reward_mlp, rf_predictor, optimizer
        )
        reward_mlp = reward_mlp.to(device)
        rf_predictor = rf_predictor.to(device)
        logging.info("Resumed from checkpoint at step %d", step_count)

    # -----------------------------------------------------------------------
    # Auto-update training_status.json on start (fixes WARNING 2)
    # -----------------------------------------------------------------------

    update_training_status("reward_head", "training")

    # -----------------------------------------------------------------------
    # Training loop (epoch-based, iterate over records in mini-batches)
    # -----------------------------------------------------------------------

    batch_size = config["batch_size"]
    max_grad_norm = config["max_grad_norm"]
    eval_every = config["eval_every_n_steps"]

    for epoch in range(max_epochs):
        indices = list(range(num_records))
        random.shuffle(indices)

        for i in range(0, num_records, batch_size):
            batch_indices = indices[i : i + batch_size]
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)

            for idx in batch_indices:
                if use_preprocessed:
                    def _tensors(prefix):
                        return {
                            "ego": pt_data[f"ego_{prefix}"][idx].unsqueeze(0).to(device),
                            "scene": pt_data[f"scene_{prefix}"][idx].unsqueeze(0).to(device),
                            "route": pt_data[f"route_{prefix}"][idx].unsqueeze(0).to(device),
                            "entities": pt_data[f"entities_{prefix}"][idx].unsqueeze(0).to(device),
                            "mask": pt_data[f"mask_{prefix}"][idx].unsqueeze(0).to(device),
                        }
                    z_parent = encode_tensors(_tensors("before"), encoder_weights).detach()
                    z_child = encode_tensors(_tensors("after"), encoder_weights).detach()
                    rf_parent = pt_data["rf_before"][idx].unsqueeze(0).to(device)
                    rf_child_real = pt_data["rf_after"][idx].unsqueeze(0).to(device)
                    duration = pt_data["durations"][idx].item()
                    realized_return = pt_data["returns"][idx].item()
                else:
                    record = dataset[idx]
                    state_before = record["state_before"]
                    state_after = record["state_after"]
                    duration = record["duration"]
                    realized_return = record["realized_return"]
                    z_parent = encode_state(state_before, encoder_weights).detach()
                    z_child = encode_state(state_after, encoder_weights).detach()
                    rf_parent = extract_reward_features(state_before).to(device)
                    rf_child_real = extract_reward_features(state_after).to(device)

                current_fused_dim = z_parent.shape[-1]
                delta_z = z_child - z_parent

                # Predict child reward features
                rf_child_pred, rf_predictor = predict_reward_features(
                    z_parent=z_parent,
                    delta_z=delta_z,
                    rf_parent=rf_parent,
                    rf_predictor=rf_predictor,
                    fused_dim=current_fused_dim,
                )

                # Predict reward
                duration_t = torch.tensor([[float(duration)]], dtype=torch.float32, device=device)
                time_left_t = torch.tensor([[0.0]], dtype=torch.float32, device=device)

                r_pred, reward_mlp = reward_head(
                    z_parent=z_parent,
                    z_child=z_child,
                    rf_parent=rf_parent,
                    rf_child=rf_child_pred,
                    duration=duration_t,
                    time_left=time_left_t,
                    reward_mlp=reward_mlp,
                    fused_dim=current_fused_dim,
                )

                # Combined loss (matching trainer.py lines 549-560)
                target_return = torch.tensor([[realized_return]], dtype=torch.float32, device=device)
                reward_loss = (r_pred - target_return) ** 2
                rf_loss = ((rf_child_pred - rf_child_real.detach()) ** 2).mean()
                total_loss = total_loss + reward_loss + rf_loss

            # Batch-mean loss
            total_loss = total_loss / len(batch_indices)
            total_loss.backward()

            all_params = list(reward_mlp.parameters()) + list(rf_predictor.parameters())
            grad_norm = clip_grad_norm_(all_params, max_grad_norm)
            clipped = bool(grad_norm.item() > max_grad_norm)
            optimizer.step()
            step_count += 1

            # Logging
            if step_count % 10 == 0:
                logging.info(
                    "step=%d loss=%.6f grad_norm=%.4f clipped=%s",
                    step_count,
                    total_loss.item(),
                    grad_norm.item(),
                    clipped,
                )

            # Convergence check
            if step_count % eval_every == 0:
                eval_loss = total_loss.item()
                if convergence.update(eval_loss):
                    logging.info(
                        "CONVERGED at step %d, MSE=%.6f", step_count, eval_loss
                    )
                    # Save final checkpoint
                    ckpt_path = save_reward_checkpoint(
                        checkpoint_dir, reward_mlp, rf_predictor, optimizer, step_count
                    )
                    # FREEZE reward modules on convergence (fixes WARNING 1)
                    freeze_module(reward_mlp)
                    freeze_module(rf_predictor)
                    # Auto-update training_status.json (fixes WARNING 2)
                    update_training_status(
                        "reward_head",
                        "converged",
                        metric=eval_loss,
                        steps=step_count,
                        checkpoint=str(ckpt_path),
                    )
                    return {
                        "converged": True,
                        "final_mse": eval_loss,
                        "total_steps": step_count,
                    }

        logging.info("Epoch %d complete, step_count=%d", epoch + 1, step_count)

    # Max epochs reached without convergence
    logging.warning(
        "Max epochs (%d) reached without convergence at step %d", max_epochs, step_count
    )
    # Save checkpoint even if not converged
    ckpt_path = save_reward_checkpoint(
        checkpoint_dir, reward_mlp, rf_predictor, optimizer, step_count
    )
    update_training_status(
        "reward_head",
        "training",
        steps=step_count,
        checkpoint=str(ckpt_path),
    )
    return {"converged": False, "final_mse": None, "total_steps": step_count}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train reward head (Stage 2)")
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
    parser.add_argument("--max-epochs", type=int, default=100)
    args = parser.parse_args()

    result = train_reward_head_offline(
        data_dir=args.data_dir,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        encoder_checkpoint=args.encoder_checkpoint,
        max_epochs=args.max_epochs,
    )
    print(json.dumps(result, indent=2))
