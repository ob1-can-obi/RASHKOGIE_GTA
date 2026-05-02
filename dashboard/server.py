"""
FastAPI server for the RASHKOGIE Training Dashboard.

Entry point for the web application. Manages lifespan (database init/close,
collector background task, WebSocket manager), mounts middleware, hosts API
routes and WebSocket endpoints.

Usage:
    python -m dashboard.server [--host 0.0.0.0] [--port 8000] [--log-level info]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
import uvicorn

from dashboard.database import init_database
from dashboard.auth import PasswordAuthMiddleware, verify_ws_token
from dashboard.ws_manager import WSManager
from dashboard.collector import JSONLCollector
from dashboard.routes.metrics import metrics_router
from dashboard.routes.params import params_router
from dashboard.routes.sessions import sessions_router
from dashboard.routes.checkpoints import checkpoints_router
from dashboard.routes.embeddings import embeddings_router
from dashboard.routes.predictions import predictions_router

_log = logging.getLogger("dashboard")
_DASHBOARD_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database, collector, and WebSocket manager on startup."""
    # Init database (D-01)
    db = await init_database()
    app.state.db = db
    app.state.project_root = str(_PROJECT_ROOT)
    _log.info("Database initialized")

    # Init WebSocket manager
    ws = WSManager()
    app.state.ws = ws
    _log.info("WebSocket manager initialized")

    # Start JSONL collector as background task (D-02)
    collector = JSONLCollector(db=db, ws_manager=ws, project_root=_PROJECT_ROOT)
    collector_task = asyncio.create_task(collector.run(), name="jsonl-collector")
    app.state.collector_task = collector_task
    _log.info("Collector started")

    yield

    # Shutdown
    collector_task.cancel()
    try:
        await collector_task
    except asyncio.CancelledError:
        pass
    await db.close()
    _log.info("Shutdown complete")


app = FastAPI(title="RASHKOGIE Training Dashboard", lifespan=lifespan)
app.add_middleware(PasswordAuthMiddleware)

# REST routes
app.include_router(metrics_router, prefix="/api")
app.include_router(params_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(checkpoints_router, prefix="/api")
app.include_router(embeddings_router, prefix="/api")
app.include_router(predictions_router, prefix="/api")


# Health check endpoint (no auth needed -- excluded in PasswordAuthMiddleware)
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# WebSocket endpoint for browser clients (D-04)
@app.websocket("/ws/browser")
async def ws_browser(websocket: WebSocket, token: str = Query("")):
    """Browser WebSocket: receives metric updates, sends param commands."""
    await verify_ws_token(token)
    ws = websocket.app.state.ws
    await ws.connect_browser(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")
            if msg_type == "set_params":
                # Forward to params route logic
                from dashboard.routes.params import _config_lock
                config_path = Path(websocket.app.state.project_root) / "training_config.json"
                params = data.get("params", {})
                module = params.pop("module", None)
                if module and params:
                    async with _config_lock:
                        try:
                            with open(config_path, encoding="utf-8") as f:
                                config = json.load(f)
                        except (FileNotFoundError, json.JSONDecodeError):
                            config = {}
                        module_config = config.get(module, {})
                        for key, value in params.items():
                            if key.startswith("convergence_"):
                                conv = module_config.setdefault("convergence", {})
                                conv[key.replace("convergence_", "")] = value
                            else:
                                module_config[key] = value
                        config[module] = module_config
                        with open(config_path, "w", encoding="utf-8") as f:
                            json.dump(config, f, indent=4)
                            f.write("\n")
                    await ws.broadcast_param_change({"module": module, **params})
                    await ws.send_param_ack(websocket, {"module": module, **params})
            elif msg_type == "restore_checkpoint":
                session_id = data.get("session_id", "")
                module = data.get("module", "")
                # Build checkpoint path
                checkpoint_path = f"{module}/checkpoints/session_{session_id}"
                await ws.forward_restore_to_train(checkpoint_path)
                await ws.send_restore_ack(websocket, session_id, module, True)
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws.disconnect_browser(websocket)
    except Exception:
        ws.disconnect_browser(websocket)


# WebSocket endpoint for training script clients (D-06)
@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket):
    """Training script WebSocket: receives param updates and restore commands."""
    ws = websocket.app.state.ws
    await ws.connect_train(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")
            if msg_type == "register":
                _log.info("Train client registered | module=%s", data.get("module"))
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws.disconnect_train(websocket)
    except Exception:
        ws.disconnect_train(websocket)


# SPA mount LAST (activated in Plan 05 when frontend is built)
frontend_dist = _DASHBOARD_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="spa")


def main():
    """CLI entry point for running the dashboard server."""
    parser = argparse.ArgumentParser(description="RASHKOGIE Training Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0 per D-15)")
    parser.add_argument("--port", default=8000, type=int, help="Bind port")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s.%(msecs)03d [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
