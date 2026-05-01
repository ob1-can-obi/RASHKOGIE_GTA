"""Database helpers stub -- actual implementation provided by Plan 05-01.

This file exists only to satisfy import resolution during parallel development.
It will be replaced by the full implementation when Plan 01 worktree merges.
"""
from __future__ import annotations

from typing import Any


async def insert_session(db: Any, session_id: str, module: str, started_at: str, status: str = "running") -> None:
    """Insert a training session record."""
    raise NotImplementedError("Stub -- see Plan 05-01")


async def insert_metrics_batch(db: Any, rows: list[dict]) -> None:
    """Batch-insert training metric rows."""
    raise NotImplementedError("Stub -- see Plan 05-01")


async def insert_decision_counts(db: Any, rows: list[dict]) -> None:
    """Batch-insert decision distribution counts."""
    raise NotImplementedError("Stub -- see Plan 05-01")


async def insert_embeddings(db: Any, rows: list[dict]) -> None:
    """Batch-insert embedding snapshot rows."""
    raise NotImplementedError("Stub -- see Plan 05-01")


async def insert_predictions(db: Any, rows: list[dict]) -> None:
    """Batch-insert prediction rows."""
    raise NotImplementedError("Stub -- see Plan 05-01")
