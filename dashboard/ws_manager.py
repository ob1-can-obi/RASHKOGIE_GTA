"""WebSocket connection manager for the training dashboard.

Manages two pools of WebSocket clients:
- Browser clients: receive metric updates, decision distributions, session events
- Training script clients: receive parameter change commands, checkpoint restore commands

Follows the broadcast pattern from gta_stream/gta_ws_bridge.py with stale client cleanup.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

_log = logging.getLogger("dashboard.ws")


class WSManager:
    """Manages WebSocket connections for browser clients and training script clients.

    Browser clients receive metric updates (push from collector).
    Training script clients receive parameter change commands (push from browser).
    """

    def __init__(self, max_browser_clients: int = 20, max_train_clients: int = 10):
        self.browser_clients: list[WebSocket] = []
        self.train_clients: list[WebSocket] = []
        self._max_browser = max_browser_clients
        self._max_train = max_train_clients
        self._config_version: int = 0

    # ---- Connection lifecycle ----

    async def connect_browser(self, ws: WebSocket) -> None:
        """Accept a browser WebSocket connection."""
        if len(self.browser_clients) >= self._max_browser:
            await ws.close(code=1013, reason="Too many connections")
            return
        await ws.accept()
        self.browser_clients.append(ws)
        _log.info("Browser client connected | total=%d", len(self.browser_clients))
        await self._send_json(ws, {
            "type": "hello",
            "config_version": self._config_version,
        })

    async def connect_train(self, ws: WebSocket) -> None:
        """Accept a training script WebSocket connection."""
        if len(self.train_clients) >= self._max_train:
            await ws.close(code=1013, reason="Too many connections")
            return
        await ws.accept()
        self.train_clients.append(ws)
        _log.info("Train client connected | total=%d", len(self.train_clients))

    def disconnect_browser(self, ws: WebSocket) -> None:
        """Remove a browser client."""
        if ws in self.browser_clients:
            self.browser_clients.remove(ws)
        _log.info("Browser client disconnected | total=%d", len(self.browser_clients))

    def disconnect_train(self, ws: WebSocket) -> None:
        """Remove a training script client."""
        if ws in self.train_clients:
            self.train_clients.remove(ws)
        _log.info("Train client disconnected | total=%d", len(self.train_clients))

    # ---- Broadcast methods ----

    async def broadcast_metrics(self, data: dict) -> None:
        """Push new metrics to all browser clients (DASH-02, DASH-07, DASH-08)."""
        msg = json.dumps({"type": "metrics_update", **data}, separators=(",", ":"))
        await self._broadcast_to(self.browser_clients, msg)

    async def broadcast_decisions(self, data: dict) -> None:
        """Push decision distribution update to browser clients (DASH-03)."""
        msg = json.dumps({"type": "decision_update", **data}, separators=(",", ":"))
        await self._broadcast_to(self.browser_clients, msg)

    async def broadcast_session_event(self, event_type: str, session_data: dict) -> None:
        """Push session lifecycle events (started/ended/converged) to browser clients."""
        msg = json.dumps({"type": event_type, **session_data}, separators=(",", ":"))
        await self._broadcast_to(self.browser_clients, msg)

    async def broadcast_param_change(self, params: dict) -> None:
        """Forward parameter changes to all training script clients (D-06)."""
        self._config_version += 1
        msg = json.dumps({
            "type": "param_update",
            "params": params,
            "config_version": self._config_version,
        }, separators=(",", ":"))
        await self._broadcast_to(self.train_clients, msg)
        _log.info(
            "Param change broadcast | version=%d | params=%s",
            self._config_version, list(params.keys()),
        )

    async def send_param_ack(self, ws: WebSocket, params: dict) -> None:
        """Acknowledge a param change back to the requesting browser client."""
        await self._send_json(ws, {
            "type": "param_ack",
            "params": params,
            "config_version": self._config_version,
        })

    async def send_restore_ack(
        self, ws: WebSocket, session_id: str, module: str, success: bool,
    ) -> None:
        """Acknowledge a checkpoint restore request."""
        await self._send_json(ws, {
            "type": "restore_ack",
            "session_id": session_id,
            "module": module,
            "success": success,
        })

    async def forward_restore_to_train(self, checkpoint_path: str) -> None:
        """Forward restore command to training script clients."""
        msg = json.dumps({
            "type": "restore_checkpoint",
            "checkpoint_path": checkpoint_path,
        }, separators=(",", ":"))
        await self._broadcast_to(self.train_clients, msg)

    # ---- Internal helpers ----

    async def _broadcast_to(self, clients: list[WebSocket], msg: str) -> None:
        """Broadcast a message to a client list, removing stale clients."""
        stale = []
        for ws in list(clients):
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)
        for ws in stale:
            if ws in clients:
                clients.remove(ws)
        if stale:
            _log.debug("Removed %d stale clients", len(stale))

    async def _send_json(self, ws: WebSocket, payload: dict) -> None:
        """Send a JSON message to a single client."""
        await ws.send_text(json.dumps(payload, separators=(",", ":")))

    @property
    def config_version(self) -> int:
        return self._config_version
