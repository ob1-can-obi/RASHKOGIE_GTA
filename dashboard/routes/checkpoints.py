"""Checkpoint listing, download, and restore endpoints.

D-17: checkpoints in per-session subdirectories.
D-18: checkpoint list per module per session with file size, timestamp, metrics.
D-19: download checkpoints from browser.

Path traversal protection: reject ".." in path components AND verify resolved
path is within expected checkpoint directory (T-05-12, T-05-15).

Checkpoint download uses HTTP FileResponse (not WebSocket) per RESEARCH.md
Open Question 3 -- binary file transfer over HTTP is simpler and more reliable.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse

_log = logging.getLogger("dashboard.routes.checkpoints")

# Module -> checkpoint directory mapping (from train.py / trainer.py patterns)
MODULE_CHECKPOINT_DIRS = {
    "encoder_intuition": "main_model/checkpoints",
    "reward_head": "reward_head/checkpoints",
    "action_planner": "action_planner/checkpoints",
    "metacontroller": "metacontroller/checkpoints",
}

checkpoints_router = APIRouter(tags=["checkpoints"])


@checkpoints_router.get("/checkpoints")
async def list_checkpoints(
    request: Request,
    module: Optional[str] = None,
):
    """List all checkpoint files with metadata.

    D-18: shows checkpoint list per module per session with file size,
    timestamp, and metrics at save time.
    D-17: checkpoints in per-session subdirectories.
    """
    project_root = Path(request.app.state.project_root)
    checkpoints = []

    dirs_to_scan = MODULE_CHECKPOINT_DIRS
    if module and module in MODULE_CHECKPOINT_DIRS:
        dirs_to_scan = {module: MODULE_CHECKPOINT_DIRS[module]}

    for mod, rel_dir in dirs_to_scan.items():
        checkpoint_dir = project_root / rel_dir
        if not checkpoint_dir.exists():
            continue
        for session_dir in sorted(checkpoint_dir.glob("session_*")):
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name  # e.g., "session_20260501_120000"
            for pt_file in sorted(session_dir.glob("*.pt")):
                stat = pt_file.stat()
                checkpoints.append({
                    "session_id": session_id,
                    "module": mod,
                    "filename": pt_file.name,
                    "file_size": stat.st_size,
                    "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(pt_file.relative_to(project_root)),
                })

    return {"checkpoints": checkpoints, "count": len(checkpoints)}


@checkpoints_router.get("/checkpoints/{module}/{session_id}/{filename}")
async def download_checkpoint(
    request: Request,
    module: str,
    session_id: str,
    filename: str,
):
    """Download a checkpoint file.

    D-19: download checkpoints from browser.
    Uses HTTP (not WebSocket) for binary file transfer per RESEARCH.md Open Question 3.
    """
    # Path traversal protection (T-05-12)
    if ".." in module or ".." in session_id or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path component")
    if module not in MODULE_CHECKPOINT_DIRS:
        raise HTTPException(status_code=404, detail=f"Unknown module: {module}")

    project_root = Path(request.app.state.project_root)
    file_path = project_root / MODULE_CHECKPOINT_DIRS[module] / session_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Verify the resolved path is within the expected directory (defense in depth, T-05-15)
    try:
        file_path.resolve().relative_to(
            (project_root / MODULE_CHECKPOINT_DIRS[module]).resolve()
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream",
    )
