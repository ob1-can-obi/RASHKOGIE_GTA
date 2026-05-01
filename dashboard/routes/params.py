"""REST + WebSocket endpoints for hyperparameter control.

DASH-04: hyperparameter control panel (GET/PUT /api/params).
D-07: hot-reloadable params: lr, entropy_coeff, think_cost, batch_size + convergence.
D-08: writes changes to training_config.json for persistence.
D-06: broadcasts changes to training scripts via WebSocket.

Uses asyncio.Lock to protect training_config.json read-modify-write
(per RESEARCH.md anti-pattern: no locking).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from dashboard.auth import verify_ws_token

_log = logging.getLogger("dashboard.routes.params")

params_router = APIRouter(tags=["params"])

# Lock for config file read-modify-write (RESEARCH.md anti-pattern: no locking)
_config_lock = asyncio.Lock()


class ParamUpdateRequest(BaseModel):
    """Request body for PUT /api/params. Per D-07 hot-reloadable params."""

    module: str
    lr: Optional[float] = Field(None, gt=0, description="Learning rate > 0")
    entropy_coeff: Optional[float] = Field(None, ge=0, description="Entropy coefficient >= 0")
    think_cost: Optional[float] = Field(None, ge=0, description="Think cost >= 0")
    batch_size: Optional[int] = Field(None, ge=1, description="Batch size >= 1")
    convergence_threshold: Optional[float] = Field(None, gt=0, description="Convergence threshold > 0")
    convergence_patience: Optional[int] = Field(None, ge=1, description="Convergence patience >= 1")


@params_router.get("/params")
async def get_params(request: Request):
    """Return current training_config.json contents.

    DASH-04: populates hyperparameter control panel.
    """
    config_path = Path(request.app.state.project_root) / "training_config.json"
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        _log.warning("Could not read training_config.json: %s", e)
        config = {}
    return {"config": config, "config_version": request.app.state.ws.config_version}


@params_router.put("/params")
async def update_params(request: Request, update: ParamUpdateRequest):
    """Update training_config.json and broadcast to training scripts.

    DASH-04: hyperparameter hot-reload.
    D-07: hot-reloadable params: lr, entropy_coeff, think_cost, batch_size + convergence.
    D-08: writes changes to training_config.json for persistence.
    """
    config_path = Path(request.app.state.project_root) / "training_config.json"
    ws = request.app.state.ws

    async with _config_lock:
        # Read current config
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}

        module_config = config.get(update.module, {})

        # Apply updates (only non-None fields)
        changed: dict = {}
        if update.lr is not None:
            module_config["lr"] = update.lr
            changed["lr"] = update.lr
        if update.entropy_coeff is not None:
            module_config["entropy_coeff"] = update.entropy_coeff
            changed["entropy_coeff"] = update.entropy_coeff
        if update.think_cost is not None:
            module_config["think_cost"] = update.think_cost
            changed["think_cost"] = update.think_cost
        if update.batch_size is not None:
            module_config["batch_size"] = update.batch_size
            changed["batch_size"] = update.batch_size
        if update.convergence_threshold is not None:
            conv = module_config.setdefault("convergence", {})
            conv["threshold"] = update.convergence_threshold
            changed["convergence_threshold"] = update.convergence_threshold
        if update.convergence_patience is not None:
            conv = module_config.setdefault("convergence", {})
            conv["patience"] = update.convergence_patience
            changed["convergence_patience"] = update.convergence_patience

        config[update.module] = module_config

        # Write back to file (D-08)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
            f.write("\n")

    # Broadcast to training scripts via WebSocket (D-06)
    if changed:
        await ws.broadcast_param_change({"module": update.module, **changed})
        _log.info("Params updated | module=%s | changed=%s", update.module, list(changed.keys()))

    return {
        "status": "ok",
        "module": update.module,
        "changed": changed,
        "config_version": ws.config_version,
    }
