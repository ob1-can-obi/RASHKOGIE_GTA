"""
Stage 1 training script: Joint encoder + intuition head training.

Trains both modules simultaneously via next-state prediction MSE loss.
Gradients flow from the intuition head's prediction error back through
the encoder's attention blocks, teaching the encoder to produce useful
representations (z_t embeddings).

Usage:
    python main_model/train.py
    python main_model/train.py --data-dir path/to/data --max-epochs 50
    python main_model/train.py --resume path/to/session_dir

Training data format (JSONL, one record per line):
    {"state_t": {...}, "state_t1": {...}, "token_id": int,
     "session_ts": str, "frame_idx": int}

On convergence:
    - Saves final checkpoint to main_model/checkpoints/
    - Freezes encoder_weights, intuition_mlp, and token_embed (D-12)
    - Auto-updates training_status.json (fixes WARNING 2)
"""

import os
import sys
import json
import random
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# ---------------------------------------------------------------------------
# Sibling module resolution (same pattern as main_model.py)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
for _d in ("main_model", "intuition_head"):
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from main_model import create_encoder_weights, encode_state
from intuition_head import intuition_head
from training_utils import (
    ConvergenceDetector,
    freeze_module,
    load_training_config,
    update_training_status,
)


# ---------------------------------------------------------------------------
# JSONL data loading (T-04-04: try/except per line, skip malformed)
# ---------------------------------------------------------------------------

def load_data(data_dir):
    """
    Load all JSONL records from a training data directory.

    Iterates sorted .jsonl files, parses one JSON object per line.
    Strips whitespace (handles CRLF), skips empty lines and comments.
    Malformed lines are logged and skipped (T-04-04 mitigation).

    Args:
        data_dir: Path to directory containing .jsonl files

    Returns:
        list[dict]: Parsed training records, each with keys
                    state_t, state_t1, token_id, session_ts, frame_idx
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
# Checkpoint save/load (T-04-05: weights_only=True, map_location="cpu")
# ---------------------------------------------------------------------------

def save_training_checkpoint(
    checkpoint_dir, encoder_weights, intuition_mlp, token_embed, optimizer, step_count
):
    """
    Save a full training checkpoint for Stage 1.

    Creates a session directory under checkpoint_dir with separate files
    for encoder weights, intuition MLP, token embedding, and optimizer state.

    Args:
        checkpoint_dir: Base checkpoint directory
        encoder_weights: Encoder weight dict from create_encoder_weights()
        intuition_mlp: Intuition head MLP (nn.Sequential)
        token_embed: Token embedding table (nn.Embedding)
        optimizer: Adam optimizer
        step_count: Current training step

    Returns:
        Path: Session directory containing the saved checkpoint files
    """
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(checkpoint_dir) / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save encoder weights (dict of nn.Module state dicts)
    encoder_state = {}
    for key, val in encoder_weights.items():
        if hasattr(val, "state_dict"):
            encoder_state[key] = val.state_dict()
    torch.save(encoder_state, session_dir / "encoder_weights.pt")

    # Save intuition MLP
    torch.save(
        {"model_state_dict": intuition_mlp.state_dict()},
        session_dir / "intuition_mlp.pt",
    )

    # Save token embedding
    torch.save(
        {"model_state_dict": token_embed.state_dict()},
        session_dir / "token_embed.pt",
    )

    # Save optimizer + step count
    torch.save(
        {"optimizer_state_dict": optimizer.state_dict(), "step_count": step_count},
        session_dir / "optimizer.pt",
    )

    return session_dir


def load_training_checkpoint(
    session_dir, encoder_weights, intuition_mlp, token_embed, optimizer
):
    """
    Restore training state from a saved checkpoint.

    All torch.load calls use weights_only=True and map_location="cpu"
    per T-04-05 threat mitigation.

    Args:
        session_dir: Path to session directory containing checkpoint files
        encoder_weights: Encoder weight dict to restore into
        intuition_mlp: Intuition head MLP to restore into
        token_embed: Token embedding to restore into
        optimizer: Optimizer to restore into

    Returns:
        int: Restored step_count
    """
    session_dir = Path(session_dir)

    # Restore encoder weights
    encoder_state = torch.load(
        session_dir / "encoder_weights.pt", weights_only=True, map_location="cpu"
    )
    for key in encoder_state:
        if key in encoder_weights and hasattr(encoder_weights[key], "load_state_dict"):
            encoder_weights[key].load_state_dict(encoder_state[key])

    # Restore intuition MLP
    intuition_ckpt = torch.load(
        session_dir / "intuition_mlp.pt", weights_only=True, map_location="cpu"
    )
    intuition_mlp.load_state_dict(intuition_ckpt["model_state_dict"])

    # Restore token embedding
    token_ckpt = torch.load(
        session_dir / "token_embed.pt", weights_only=True, map_location="cpu"
    )
    token_embed.load_state_dict(token_ckpt["model_state_dict"])

    # Restore optimizer + step count
    opt_ckpt = torch.load(
        session_dir / "optimizer.pt", weights_only=True, map_location="cpu"
    )
    optimizer.load_state_dict(opt_ckpt["optimizer_state_dict"])
    step_count = opt_ckpt["step_count"]

    return step_count


# ---------------------------------------------------------------------------
# Driving context classification for embedding coloring (D-21b)
# ---------------------------------------------------------------------------

def _derive_driving_context(state: dict) -> str:
    """Classify driving context from GTA state for embedding coloring (D-21b).

    Returns 'straight', 'turn', or 'braking' based on steering and brake inputs.
    Falls back to 'unknown' if state keys are missing.
    """
    try:
        steering = abs(state.get("steering", 0.0))
        brake = state.get("brake", 0.0)
        if brake > 0.3:
            return "braking"
        elif steering > 0.3:
            return "turn"
        else:
            return "straight"
    except (TypeError, KeyError):
        return "unknown"


# ---------------------------------------------------------------------------
# Dashboard WebSocket param receiver for hot-reload (D-06)
# ---------------------------------------------------------------------------

class DashboardParamReceiver:
    """Background WebSocket client that connects to the dashboard server
    to receive hyperparameter updates (D-06).

    Runs in a daemon thread. If the dashboard is not running, silently
    retries every 10 seconds. When a param_update message arrives, it
    updates the optimizer learning rate and stores other params for the
    training loop to read.
    """

    def __init__(self, module_name: str, optimizer):
        self.module_name = module_name
        self.optimizer = optimizer
        self.pending_params = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        # Dashboard URL from env or default
        host = os.environ.get("DASHBOARD_HOST", "localhost")
        port = os.environ.get("DASHBOARD_PORT", "8000")
        self.ws_url = f"ws://{host}:{port}/ws/train"

    def start(self):
        """Start the background receiver thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="dashboard-ws")
        self._thread.start()
        logging.info("Dashboard param receiver started | url=%s", self.ws_url)

    def stop(self):
        """Signal the thread to stop."""
        self._stop.set()

    def get_pending_params(self) -> dict:
        """Retrieve and clear any pending parameter updates."""
        with self._lock:
            params = self.pending_params.copy()
            self.pending_params.clear()
        return params

    def _run(self):
        """Background loop: connect, register, listen for param updates."""
        import websockets.sync.client as ws_client

        while not self._stop.is_set():
            try:
                with ws_client.connect(self.ws_url, close_timeout=5) as ws:
                    # Register with module name
                    ws.send(json.dumps({"type": "register", "module": self.module_name}))
                    logging.info("Connected to dashboard WS | module=%s", self.module_name)

                    while not self._stop.is_set():
                        try:
                            msg = ws.recv(timeout=2.0)
                            data = json.loads(msg)
                            if data.get("type") == "param_update":
                                self._apply_params(data.get("params", {}))
                        except TimeoutError:
                            # No message received in timeout -- just keep waiting
                            continue
            except Exception as exc:
                if not self._stop.is_set():
                    logging.debug("Dashboard WS not available: %s (retrying in 10s)", exc)
                    self._stop.wait(10.0)

    def _apply_params(self, params: dict):
        """Apply parameter updates. lr goes directly to optimizer; others are stored.

        T-05-20 mitigation: Only applies known param keys (lr, entropy_coeff,
        think_cost, batch_size); validates lr > 0, batch_size >= 1 before applying.
        """
        target_module = params.get("module", "")
        if target_module and target_module != self.module_name:
            return  # Not for this module

        with self._lock:
            if "lr" in params:
                new_lr = float(params["lr"])
                if new_lr > 0:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = new_lr
                    logging.info("Hot-reload: lr updated to %g", new_lr)

            # Store other params for the training loop to read
            for key in ("entropy_coeff", "think_cost", "batch_size"):
                if key in params:
                    self.pending_params[key] = params[key]
                    logging.info("Hot-reload: %s queued = %s", key, params[key])


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_encoder_intuition(
    data_dir=None,
    config_path=None,
    checkpoint_dir=None,
    resume_from=None,
    vocab_size=874,
    max_epochs=100,
):
    """
    Stage 1: Joint encoder + intuition head training via next-state prediction MSE.

    Both z_t and z_t1_real are computed with live autograd graphs through the
    same encoder weights (Pitfall 1: do NOT detach z_t1_real). The MSE loss
    flows gradients to both the intuition head AND the encoder.

    Args:
        data_dir: Path to JSONL training data. Defaults to main_model/training_data/
        config_path: Path to training_config.json. Defaults to project root.
        checkpoint_dir: Path to save checkpoints. Defaults to main_model/checkpoints/
        resume_from: Path to session directory to resume from. None = fresh start.
        vocab_size: Token vocabulary size (default 874)
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
    config = config_full["encoder_intuition"]

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    # -----------------------------------------------------------------------
    # Initialize modules
    # -----------------------------------------------------------------------

    # Create encoder weights
    encoder_weights = create_encoder_weights()

    # Initialize intuition head by calling once to create token_embed and intuition_mlp
    _dummy_z = torch.randn(1, 128)
    _dummy_token = torch.tensor([0])
    _, _, token_embed, intuition_mlp = intuition_head(
        _dummy_z, _dummy_token, vocab_size
    )

    # -----------------------------------------------------------------------
    # Collect all trainable parameters from both modules
    # -----------------------------------------------------------------------

    encoder_params = []
    for key in encoder_weights:
        if hasattr(encoder_weights[key], "parameters"):
            encoder_params.extend(encoder_weights[key].parameters())

    intuition_params = list(intuition_mlp.parameters()) + list(
        token_embed.parameters()
    )

    all_params = encoder_params + intuition_params

    # -----------------------------------------------------------------------
    # Optimizer and convergence detection
    # -----------------------------------------------------------------------

    optimizer = Adam(all_params, lr=config["lr"], eps=1e-5)

    conv_cfg = config["convergence"]
    convergence = ConvergenceDetector(
        threshold=conv_cfg["threshold"],
        patience=conv_cfg["patience"],
        mode=conv_cfg["mode"],
    )

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------

    records = load_data(data_dir)
    if not records:
        logging.warning("No training records found in %s", data_dir)
        return {"converged": False, "final_mse": None, "total_steps": 0}

    logging.info("Loaded %d training records from %s", len(records), data_dir)

    # -----------------------------------------------------------------------
    # Resume from checkpoint if requested
    # -----------------------------------------------------------------------

    step_count = 0
    if resume_from is not None:
        step_count = load_training_checkpoint(
            resume_from, encoder_weights, intuition_mlp, token_embed, optimizer
        )
        logging.info("Resumed from checkpoint at step %d", step_count)

    # -----------------------------------------------------------------------
    # Auto-update training_status.json on start (fixes WARNING 2)
    # -----------------------------------------------------------------------

    update_training_status("encoder_intuition", "training")

    # -----------------------------------------------------------------------
    # JSONL output for dashboard collector (D-05)
    # -----------------------------------------------------------------------

    session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    jsonl_dir = data_dir  # same directory as training data input
    jsonl_path = jsonl_dir / f"{session_id}.jsonl"
    jsonl_file = open(jsonl_path, "a", encoding="utf-8")

    def write_jsonl(row: dict) -> None:
        """Append a single JSON line to the JSONL output file."""
        jsonl_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        jsonl_file.flush()

    # -----------------------------------------------------------------------
    # Start dashboard WebSocket param receiver (D-06)
    # -----------------------------------------------------------------------

    param_receiver = DashboardParamReceiver("encoder_intuition", optimizer)
    param_receiver.start()

    # -----------------------------------------------------------------------
    # Training loop (epoch-based, iterate over records in mini-batches)
    # -----------------------------------------------------------------------

    batch_size = config["batch_size"]
    max_grad_norm = config["max_grad_norm"]
    eval_every = config["eval_every_n_steps"]

    for epoch in range(max_epochs):
        random.shuffle(records)

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            optimizer.zero_grad()

            # Apply any pending hot-reload params (D-06)
            pending = param_receiver.get_pending_params()
            if pending and "batch_size" in pending:
                new_bs = int(pending["batch_size"])
                if new_bs >= 1:
                    batch_size = new_bs
                    logging.info("Hot-reload: batch_size updated to %d", batch_size)

            total_loss = torch.tensor(0.0)

            for record in batch:
                state_t = record["state_t"]
                state_t1 = record["state_t1"]
                token_id = record["token_id"]

                # CRITICAL: Both z_t and z_t1_real computed WITH gradients
                # per D-02 and Pitfall 1 -- do NOT detach z_t1_real
                z_t = encode_state(state_t, encoder_weights)  # [1, 128]
                z_t1_real = encode_state(state_t1, encoder_weights)  # [1, 128] WITH grad

                prev_token = torch.tensor([token_id])
                z_next_pred, _, token_embed, intuition_mlp = intuition_head(
                    z_t,
                    prev_token,
                    vocab_size,
                    token_embed=token_embed,
                    intuition_mlp=intuition_mlp,
                )

                loss = torch.nn.functional.mse_loss(z_next_pred, z_t1_real)
                total_loss = total_loss + loss

            # Batch-mean loss
            total_loss = total_loss / len(batch)
            total_loss.backward()
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

                # JSONL metric row for dashboard collector (D-05)
                write_jsonl({
                    "type": "metric",
                    "step": step_count,
                    "loss": total_loss.item(),
                    "grad_norm": grad_norm.item(),
                    "clipped": clipped,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            # Embedding snapshot for PCA visualization (D-05)
            if step_count % 500 == 0 and step_count > 0:
                # z_t is already computed above in the training loop
                # Use the z_t from the LAST record in the current batch
                z_t_list = z_t.detach().cpu().squeeze().tolist()  # [128] floats
                # Derive driving context from state_t (if speed/steering available)
                driving_context = _derive_driving_context(state_t)
                write_jsonl({
                    "type": "embedding",
                    "step": step_count,
                    "z_t": z_t_list,
                    "driving_context": driving_context,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            # Convergence check
            if step_count % eval_every == 0:
                eval_loss = total_loss.item()
                if convergence.update(eval_loss):
                    logging.info(
                        "CONVERGED at step %d, MSE=%.6f", step_count, eval_loss
                    )
                    # Save final checkpoint
                    ckpt_path = save_training_checkpoint(
                        checkpoint_dir,
                        encoder_weights,
                        intuition_mlp,
                        token_embed,
                        optimizer,
                        step_count,
                    )
                    # Freeze per D-12
                    freeze_module(encoder_weights)
                    freeze_module(intuition_mlp)
                    freeze_module(token_embed)
                    # Auto-update training_status.json (fixes WARNING 2)
                    update_training_status(
                        "encoder_intuition",
                        "converged",
                        metric=eval_loss,
                        steps=step_count,
                        checkpoint=str(ckpt_path),
                    )
                    # Cleanup JSONL and WS receiver
                    jsonl_file.close()
                    param_receiver.stop()
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
    ckpt_path = save_training_checkpoint(
        checkpoint_dir,
        encoder_weights,
        intuition_mlp,
        token_embed,
        optimizer,
        step_count,
    )
    update_training_status(
        "encoder_intuition",
        "training",
        steps=step_count,
        checkpoint=str(ckpt_path),
    )
    # Cleanup JSONL and WS receiver
    jsonl_file.close()
    param_receiver.stop()
    return {"converged": False, "final_mse": None, "total_steps": step_count}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train encoder + intuition head (Stage 1)"
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
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--vocab-size", type=int, default=874)
    args = parser.parse_args()

    result = train_encoder_intuition(
        data_dir=args.data_dir,
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        vocab_size=args.vocab_size,
        max_epochs=args.max_epochs,
    )
    print(json.dumps(result, indent=2))
