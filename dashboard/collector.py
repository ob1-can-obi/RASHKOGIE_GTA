"""JSONL-to-SQLite collector background task.

Polls module training_data directories for new JSONL lines, ingests them into
SQLite via database.py helpers, and broadcasts updates to browser WebSocket clients.

Per D-02: runs as a background asyncio task, polls every 1.5 seconds.
Per D-03: training scripts keep writing JSONL -- collector bridges JSONL -> SQLite.
Per D-05: handles embedding snapshots and decision distribution counts.

Follows RESEARCH.md Pattern 3 (JSONL Collector with File Position Tracking).
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dashboard.database import (
    insert_session,
    insert_metrics_batch,
    insert_decision_counts,
    insert_embeddings,
    insert_predictions,
)

_log = logging.getLogger("dashboard.collector")

# Module directories to scan for JSONL training data
# Matches coordinator.py STAGE_ORDER -> directory mapping
MODULE_DIRS: dict[str, str] = {
    "encoder_intuition": "main_model/training_data",
    "reward_head": "reward_head/training_data",
    "action_planner": "action_planner/training_data",
    "metacontroller": "metacontroller/training_data",
}


class JSONLCollector:
    """Background task that polls JSONL files and ingests new lines into SQLite.

    Per D-02: runs as a background asyncio task, polls every 1.5 seconds.
    Per D-03: training scripts keep writing JSONL -- collector bridges JSONL -> SQLite.
    Per D-05: handles embedding snapshots and decision distribution counts.
    """

    def __init__(
        self,
        db: Any,
        ws_manager: Any,
        project_root: Path,
        poll_interval: float = 1.5,
    ):
        self.db = db
        self.ws = ws_manager
        self.project_root = project_root
        self.poll_interval = poll_interval
        self.file_positions: dict[str, int] = {}  # filepath -> byte offset
        self._active_sessions: dict[str, str] = {}  # module -> session_id

    async def run(self) -> None:
        """Main polling loop. Runs until cancelled."""
        _log.info(
            "Collector started | poll_interval=%.1fs | root=%s",
            self.poll_interval, self.project_root,
        )
        while True:
            try:
                await self._poll_all_modules()
            except asyncio.CancelledError:
                _log.info("Collector stopped")
                raise
            except Exception as exc:
                _log.warning("Collector error: %s", exc, exc_info=True)
            await asyncio.sleep(self.poll_interval)

    async def _poll_all_modules(self) -> None:
        """Scan all module training_data directories for new JSONL lines."""
        for module, rel_dir in MODULE_DIRS.items():
            data_dir = self.project_root / rel_dir
            if not data_dir.exists():
                continue
            for jsonl_path in sorted(data_dir.glob("*.jsonl")):
                new_rows = self._read_new_lines(jsonl_path)
                if not new_rows:
                    continue

                # Derive session_id from filename (e.g., "session_20260501_120000.jsonl")
                session_id = self._extract_session_id(jsonl_path, module)

                # Classify and ingest rows
                metrics_rows: list[dict] = []
                decision_rows: list[dict] = []
                embedding_rows: list[dict] = []
                prediction_rows: list[dict] = []

                now = datetime.utcnow().isoformat()

                for row in new_rows:
                    row_type = row.get("type", "metric")
                    ts = row.get("timestamp", now)

                    if row_type == "decision_counts":
                        decision_rows.append({
                            "session_id": session_id,
                            "step": row.get("step", 0),
                            "explore": row.get("explore", 0),
                            "rollback": row.get("rollback", 0),
                            "interrupt": row.get("interrupt", 0),
                            "commit_next": row.get("commit_next", 0),
                            "nodes_expanded": row.get("nodes_expanded"),
                            "search_depth": row.get("search_depth"),
                            "timestamp": ts,
                        })
                    elif row_type == "embedding":
                        # z_t comes as JSON array of 128 floats, convert to bytes for BLOB
                        z_t_list = row.get("z_t", [])
                        z_t_blob = struct.pack(f"{len(z_t_list)}f", *z_t_list)
                        embedding_rows.append({
                            "session_id": session_id,
                            "step": row.get("step", 0),
                            "z_t": z_t_blob,
                            "driving_context": row.get("driving_context"),
                            "timestamp": ts,
                        })
                    elif row_type == "prediction":
                        pred_list = row.get("z_next_pred", [])
                        real_list = row.get("z_next_real", [])
                        pred_blob = struct.pack(f"{len(pred_list)}f", *pred_list)
                        real_blob = struct.pack(f"{len(real_list)}f", *real_list)
                        prediction_rows.append({
                            "session_id": session_id,
                            "step": row.get("step", 0),
                            "z_next_pred": pred_blob,
                            "z_next_real": real_blob,
                            "mse": row.get("mse", 0.0),
                            "timestamp": ts,
                        })
                    else:
                        # Default: training metric row
                        metrics_rows.append({
                            "session_id": session_id,
                            "module": module,
                            "step": row.get("step", 0),
                            "epoch": row.get("epoch"),
                            "loss": row.get("loss"),
                            "reward": row.get("reward"),
                            "episode_return": row.get("episode_return"),
                            "grad_norm": row.get("grad_norm"),
                            "clipped": (
                                1 if row.get("clipped")
                                else 0 if row.get("clipped") is not None
                                else None
                            ),
                            "lr": row.get("lr"),
                            "timestamp": ts,
                        })

                # Ensure session exists in DB via database.py helper
                await self._ensure_session(session_id, module)

                # Batch insert into SQLite via database.py helpers
                if metrics_rows:
                    await insert_metrics_batch(self.db, metrics_rows)
                    # Broadcast last 10 rows to browser clients
                    await self.ws.broadcast_metrics({
                        "module": module,
                        "data": metrics_rows[-10:],
                    })

                if decision_rows:
                    await insert_decision_counts(self.db, decision_rows)
                    await self.ws.broadcast_decisions({
                        "data": decision_rows[-10:],
                    })

                if embedding_rows:
                    await insert_embeddings(self.db, embedding_rows)

                if prediction_rows:
                    await insert_predictions(self.db, prediction_rows)

                _log.debug(
                    "Ingested | module=%s | metrics=%d decisions=%d embeddings=%d predictions=%d",
                    module, len(metrics_rows), len(decision_rows),
                    len(embedding_rows), len(prediction_rows),
                )

    def _read_new_lines(self, path: Path) -> list[dict]:
        """Read only new lines since last poll, handling partial writes (Pitfall 1).

        Uses byte-offset tracking per file to avoid re-reading old data.
        On JSONDecodeError, stops at the partial line and retries from that
        position on the next poll cycle.
        """
        key = str(path)
        offset = self.file_positions.get(key, 0)
        rows: list[dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(offset)
                last_good_pos = offset
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        last_good_pos = f.tell()
                        continue
                    try:
                        rows.append(json.loads(stripped))
                        last_good_pos = f.tell()
                    except json.JSONDecodeError:
                        # Partial write from concurrent training script -- stop here
                        # Retry from this position next poll (Pitfall 1)
                        _log.debug(
                            "Partial line in %s at offset %d -- will retry",
                            path.name, last_good_pos,
                        )
                        break
                self.file_positions[key] = last_good_pos
        except FileNotFoundError:
            pass
        return rows

    def _extract_session_id(self, path: Path, module: str) -> str:
        """Extract session ID from JSONL filename or generate one.

        Convention from main_model/train.py: session_20260501_120000.jsonl
        Falls back to filename stem if pattern doesn't match.
        """
        stem = path.stem
        if stem.startswith("session_"):
            return stem  # e.g., "session_20260501_120000"
        return f"{module}_{stem}"

    async def _ensure_session(self, session_id: str, module: str) -> None:
        """Create session record if it doesn't exist yet.

        Uses database.py insert_session helper. Handles duplicate insert
        gracefully (race condition with concurrent modules writing to same
        session file).
        """
        if session_id in self._active_sessions:
            return
        try:
            await insert_session(
                self.db, session_id, module, datetime.utcnow().isoformat(),
            )
            self._active_sessions[session_id] = module
            _log.info("Session created | id=%s | module=%s", session_id, module)
            await self.ws.broadcast_session_event("session_started", {
                "session_id": session_id,
                "module": module,
            })
        except Exception:
            # Already exists (race condition) -- that's fine
            self._active_sessions[session_id] = module
