"""
gta_ws_bridge.py
================
Bridges the GTA V named-pipe state/control path to a single full-duplex
WebSocket server.

Run this from Windows Python on the same machine as GTA V. Clients can then:

- receive live `state` messages from GTA V
- send `control` messages with steer/throttle/brake commands
- disable automation without touching Windows named pipes directly

Message schema
--------------

Server -> client:

    {"type":"hello","client_id":"...","has_state":false}
    {"type":"state","seq":1,"data":{...}}
    {"type":"ack","action":"control","sent_at_ms":...}
    {"type":"error","message":"..."}

Client -> server:

    {"type":"ping"}
    {"type":"get_state"}
    {"type":"disable"}
    {"type":"neutral"}
    {"type":"control","steer":0.0,"throttle":0.4,"brake":0.0,"handbrake":false}

`control` also accepts a nested payload:

    {"type":"control","payload":{"steer":0.0,"throttle":0.4,"brake":0.0}}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import queue
import secrets
import signal
import time
from typing import Any, Dict, Optional, Set

from game_control import GameControlClient
from game_state_reader import GameStateReader

websockets = None


def _require_websockets():
    global websockets
    if websockets is not None:
        return websockets

    try:
        import websockets as websockets_module
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise SystemExit(
            "Missing dependency: websockets\n"
            "Install it in the Windows Python environment with:\n"
            "  pip install websockets"
        ) from exc

    websockets = websockets_module
    return websockets


def _build_logger(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gta_ws_bridge.log")

    logger = logging.getLogger("gta5_ws_bridge")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class GTAWebSocketBridge:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        log_dir: str = "logs",
        reader_maxsize: int = 10,
    ):
        self.host = host
        self.port = int(port)
        self._log = _build_logger(log_dir)
        self._reader = GameStateReader(maxsize=reader_maxsize, log_dir=log_dir)
        self._control = GameControlClient(log_dir=log_dir)
        self._clients: Set[Any] = set()
        self._latest_state: Optional[Dict[str, Any]] = None
        self._state_seq = 0
        self._stop = asyncio.Event()
        self._server = None
        self._state_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        websockets_module = _require_websockets()
        self._reader.start()
        self._state_task = asyncio.create_task(self._pump_state(), name="gta-state-pump")
        self._server = await websockets_module.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20,
            max_size=2**23,
        )
        self._log.info("WebSocket bridge listening on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._stop.is_set():
            return

        self._stop.set()
        self._reader.stop()
        self._control.close()

        if self._state_task is not None:
            self._state_task.cancel()
            try:
                await self._state_task
            except asyncio.CancelledError:
                pass

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        if self._clients:
            await asyncio.gather(
                *(client.close(code=1001, reason="bridge shutting down") for client in list(self._clients)),
                return_exceptions=True,
            )
        self._log.info("WebSocket bridge stopped")

    async def wait_closed(self) -> None:
        await self._stop.wait()

    async def _pump_state(self) -> None:
        while not self._stop.is_set():
            try:
                state = await asyncio.to_thread(self._reader.get, 1.0)
            except queue.Empty:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log.warning("State pump error: %s", exc)
                await asyncio.sleep(1.0)
                continue

            self._state_seq += 1
            self._latest_state = state.raw
            if not self._clients:
                continue

            message = json.dumps(
                {"type": "state", "seq": self._state_seq, "data": state.raw},
                separators=(",", ":"),
            )
            await self._broadcast(message)

    async def _handle_client(self, websocket) -> None:
        client_id = secrets.token_hex(4)
        self._clients.add(websocket)
        self._log.info("Client connected | id=%s | clients=%d", client_id, len(self._clients))

        try:
            await self._send_json(
                websocket,
                {"type": "hello", "client_id": client_id, "has_state": self._latest_state is not None},
            )
            if self._latest_state is not None:
                await self._send_json(
                    websocket,
                    {"type": "state", "seq": self._state_seq, "data": self._latest_state},
                )

            async for raw in websocket:
                await self._handle_message(websocket, raw)
        except Exception as exc:
            self._log.info("Client disconnected | id=%s | reason=%s", client_id, exc)
        finally:
            self._clients.discard(websocket)
            self._log.info("Client closed | id=%s | clients=%d", client_id, len(self._clients))

    async def _handle_message(self, websocket, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "invalid JSON")
            return

        if not isinstance(data, dict):
            await self._send_error(websocket, "message must be a JSON object")
            return

        msg_type = str(data.get("type", "")).strip().lower()

        if msg_type == "ping":
            await self._send_json(websocket, {"type": "pong", "ts_ms": int(time.time() * 1000)})
            return

        if msg_type == "get_state":
            if self._latest_state is None:
                await self._send_error(websocket, "no GTA state available yet")
            else:
                await self._send_json(
                    websocket,
                    {"type": "state", "seq": self._state_seq, "data": self._latest_state},
                )
            return

        if msg_type == "disable":
            cmd = self._control.disable()
            await self._send_json(
                websocket,
                {"type": "ack", "action": "disable", "sent_at_ms": cmd.sent_at_ms},
            )
            return

        if msg_type == "neutral":
            cmd = self._control.neutral()
            await self._send_json(
                websocket,
                {"type": "ack", "action": "neutral", "sent_at_ms": cmd.sent_at_ms},
            )
            return

        if msg_type == "control":
            payload = data.get("payload")
            if payload is None:
                payload = data
            if not isinstance(payload, dict):
                await self._send_error(websocket, "control payload must be an object")
                return

            try:
                cmd = self._control.send(
                    steer=float(payload.get("steer", 0.0)),
                    throttle=float(payload.get("throttle", 0.0)),
                    brake=float(payload.get("brake", 0.0)),
                    handbrake=bool(payload.get("handbrake", False)),
                    enabled=bool(payload.get("enabled", True)),
                )
            except (TypeError, ValueError):
                await self._send_error(websocket, "control values must be numeric")
                return
            except Exception as exc:
                await self._send_error(websocket, f"control send failed: {exc}")
                return

            await self._send_json(
                websocket,
                {
                    "type": "ack",
                    "action": "control",
                    "sent_at_ms": cmd.sent_at_ms,
                    "applied": cmd.to_payload(),
                },
            )
            return

        await self._send_error(websocket, f"unknown message type: {msg_type or '<missing>'}")

    async def _broadcast(self, message: str) -> None:
        stale = []
        for client in list(self._clients):
            try:
                await client.send(message)
            except Exception:
                stale.append(client)
        for client in stale:
            self._clients.discard(client)

    async def _send_json(self, websocket, payload: Dict[str, Any]) -> None:
        await websocket.send(json.dumps(payload, separators=(",", ":")))

    async def _send_error(self, websocket, message: str) -> None:
        await self._send_json(websocket, {"type": "error", "message": message})


async def _run_bridge(args: argparse.Namespace) -> None:
    bridge = GTAWebSocketBridge(
        host=args.host,
        port=args.port,
        log_dir=args.log_dir,
        reader_maxsize=args.reader_maxsize,
    )
    await bridge.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        if not stop_event.is_set():
            stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    finally:
        await bridge.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose GTA V state/control over WebSockets")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for the WebSocket server")
    parser.add_argument("--port", default=8765, type=int, help="Bind port for the WebSocket server")
    parser.add_argument("--log-dir", default="logs", help="Directory for bridge logs")
    parser.add_argument(
        "--reader-maxsize",
        default=10,
        type=int,
        help="Queue depth for incoming GTA state frames",
    )
    args = parser.parse_args()

    if os.name != "nt":
        raise SystemExit("Run gta_ws_bridge.py from Windows Python, not WSL/Linux Python")

    try:
        asyncio.run(_run_bridge(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
