"""
game_control.py
===============
Named-pipe client for programmatic GTA V driving commands.

The paired SHVDN script listens on ``\\\\.\\pipe\\GTA5Control`` for one JSON
object per line with the following shape:

    {"enabled": true, "steer": 0.1, "throttle": 0.4, "brake": 0.0, "handbrake": false}

Commands should be sent continuously, typically at 20 Hz or faster. The game
script applies a watchdog and falls back to manual control if commands stop
arriving for more than a short timeout.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence


def _build_logger(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "game_control.log")

    logger = logging.getLogger("gta5_control")
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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


@dataclass(frozen=True)
class DriveCommand:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    handbrake: bool = False
    enabled: bool = True
    sent_at_ms: int = 0

    def to_payload(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "steer": _clamp(self.steer, -1.0, 1.0),
            "throttle": _clamp(self.throttle, 0.0, 1.0),
            "brake": _clamp(self.brake, 0.0, 1.0),
            "handbrake": bool(self.handbrake),
            "sent_at_ms": int(self.sent_at_ms or time.time() * 1000),
        }


class GameControlClient:
    PIPE_PATH = r"\\.\pipe\GTA5Control"

    def __init__(
        self,
        pipe_name: str = PIPE_PATH,
        connect_timeout: float = 5.0,
        log_dir: str = "logs",
    ):
        if os.name != "nt":
            raise OSError("GameControlClient must be run from Windows Python")

        self._pipe = pipe_name
        self._connect_timeout = float(connect_timeout)
        self._handle = None
        self._log = _build_logger(log_dir)

        import ctypes
        import ctypes.wintypes as wt

        self._ctypes = ctypes
        self._wt = wt
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateFileW.restype = wt.HANDLE
        self._k32.WriteFile.restype = wt.BOOL
        self._k32.CloseHandle.restype = wt.BOOL

        self._GENERIC_WRITE = 0x40000000
        self._OPEN_EXISTING = 3
        self._INVALID_HANDLE = wt.HANDLE(-1).value

    def connect(self) -> "GameControlClient":
        if self._handle not in (None, self._INVALID_HANDLE):
            return self

        deadline = time.time() + self._connect_timeout
        last_err: Optional[int] = None

        while time.time() < deadline:
            handle = self._k32.CreateFileW(
                self._pipe,
                self._GENERIC_WRITE,
                0,
                None,
                self._OPEN_EXISTING,
                0,
                None,
            )
            if handle not in (None, self._INVALID_HANDLE):
                self._handle = handle
                self._log.info("Pipe connected — control stream ready")
                return self

            last_err = self._ctypes.get_last_error()
            time.sleep(0.1)

        raise OSError(f"Unable to connect to {self._pipe} (last WinError={last_err})")

    def close(self) -> None:
        if self._handle in (None, self._INVALID_HANDLE):
            self._handle = None
            return
        try:
            self._k32.CloseHandle(self._handle)
        finally:
            self._handle = None

    def send(
        self,
        steer: float = 0.0,
        throttle: float = 0.0,
        brake: float = 0.0,
        handbrake: bool = False,
        enabled: bool = True,
    ) -> DriveCommand:
        cmd = DriveCommand(
            steer=steer,
            throttle=throttle,
            brake=brake,
            handbrake=handbrake,
            enabled=enabled,
            sent_at_ms=int(time.time() * 1000),
        )
        payload = json.dumps(cmd.to_payload(), separators=(",", ":")).encode("utf-8") + b"\n"
        self._write(payload)
        return cmd

    def neutral(self) -> DriveCommand:
        return self.send(steer=0.0, throttle=0.0, brake=0.0, handbrake=False, enabled=True)

    def disable(self) -> DriveCommand:
        return self.send(steer=0.0, throttle=0.0, brake=0.0, handbrake=False, enabled=False)

    def execute_timeline(
        self,
        frames: Sequence[Any],
        rate_hz: float = 20.0,
        neutral_before: bool = True,
        neutral_after: bool = True,
    ) -> DriveCommand:
        """
        Execute decoded planner frames.

        Each frame can be a dict or an object with these fields:
        duration_s, steer, throttle, brake, handbrake, enabled.
        """

        if neutral_before:
            last_cmd = self.neutral()
        else:
            last_cmd = DriveCommand()
        for frame in frames:
            if isinstance(frame, dict):
                duration_s = frame.get("duration_s", 0.05)
                steer = frame.get("steer", 0.0)
                throttle = frame.get("throttle", 0.0)
                brake = frame.get("brake", 0.0)
                handbrake = frame.get("handbrake", False)
                enabled = frame.get("enabled", True)
            else:
                duration_s = getattr(frame, "duration_s", 0.05)
                steer = getattr(frame, "steer", 0.0)
                throttle = getattr(frame, "throttle", 0.0)
                brake = getattr(frame, "brake", 0.0)
                handbrake = getattr(frame, "handbrake", False)
                enabled = getattr(frame, "enabled", True)

            duration_s = max(0.0, float(duration_s))
            period_s = 1.0 / max(1.0, float(rate_hz))
            end_time = time.monotonic() + duration_s

            last_cmd = self.send(
                steer=steer,
                throttle=throttle,
                brake=brake,
                handbrake=bool(handbrake),
                enabled=bool(enabled),
            )

            while time.monotonic() + period_s < end_time:
                time.sleep(period_s)
                last_cmd = self.send(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    handbrake=bool(handbrake),
                    enabled=bool(enabled),
                )

            remaining_s = end_time - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)

        if neutral_after:
            last_cmd = self.neutral()
        return last_cmd

    def _write(self, data: bytes) -> None:
        for attempt in range(2):
            if self._handle in (None, self._INVALID_HANDLE):
                self.connect()

            payload = self._ctypes.create_string_buffer(data)
            n_written = self._wt.DWORD(0)
            ok = self._k32.WriteFile(
                self._handle,
                payload,
                len(data),
                self._ctypes.byref(n_written),
                None,
            )
            if ok and n_written.value == len(data):
                return

            err = self._ctypes.get_last_error()
            self._log.warning("Write failed (attempt %d, WinError=%s)", attempt + 1, err)
            self.close()

        raise OSError(f"WriteFile failed for {self._pipe}")

    def __enter__(self) -> "GameControlClient":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send drive commands to GTA V")
    parser.add_argument("--steer", type=float, default=0.0, help="Steering command in [-1, 1]")
    parser.add_argument("--throttle", type=float, default=0.0, help="Throttle command in [0, 1]")
    parser.add_argument("--brake", type=float, default=0.0, help="Brake command in [0, 1]")
    parser.add_argument("--handbrake", action="store_true", help="Apply handbrake for this command")
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable automation immediately and return control to the player",
    )
    args = parser.parse_args()

    with GameControlClient() as client:
        if args.disable:
            client.disable()
        else:
            client.send(
                steer=args.steer,
                throttle=args.throttle,
                brake=args.brake,
                handbrake=args.handbrake,
                enabled=True,
            )
