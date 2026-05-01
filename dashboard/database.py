"""
SQLite database layer for the RASHKOGIE Training Dashboard.

Schema: 5 tables (sessions, metrics, decision_counts, embeddings, predictions)
         with 6 indices for fast lookups.

All queries use parameterized statements (?) -- never f-string interpolation (T-05-02).
Database runs in WAL mode for concurrent reads during training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import aiosqlite

_log = logging.getLogger("dashboard.database")

_DASHBOARD_DIR = Path(__file__).resolve().parent
_DEFAULT_DB_PATH = _DASHBOARD_DIR / "training.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT UNIQUE NOT NULL,
    module TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT DEFAULT 'running',
    final_metric REAL,
    total_steps INTEGER,
    checkpoint_path TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    module TEXT NOT NULL,
    step INTEGER NOT NULL,
    epoch INTEGER,
    loss REAL,
    reward REAL,
    episode_return REAL,
    grad_norm REAL,
    clipped INTEGER,
    lr REAL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS decision_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    explore INTEGER DEFAULT 0,
    rollback INTEGER DEFAULT 0,
    interrupt INTEGER DEFAULT 0,
    commit_next INTEGER DEFAULT 0,
    nodes_expanded INTEGER,
    search_depth INTEGER,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    z_t BLOB NOT NULL,
    driving_context TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    z_next_pred BLOB NOT NULL,
    z_next_real BLOB NOT NULL,
    mse REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_metrics_session ON metrics(session_id, step);
CREATE INDEX IF NOT EXISTS idx_metrics_module ON metrics(module, step);
CREATE INDEX IF NOT EXISTS idx_decision_session ON decision_counts(session_id, step);
CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings(session_id, step);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id, step);
CREATE INDEX IF NOT EXISTS idx_sessions_module ON sessions(module);
"""


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

async def init_database(db_path=None) -> aiosqlite.Connection:
    """
    Open (or create) the SQLite database and apply the schema.

    Args:
        db_path: Optional path override. Defaults to dashboard/training.db.

    Returns:
        aiosqlite.Connection with WAL mode enabled and row_factory set.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    db = await aiosqlite.connect(str(db_path))
    db.row_factory = aiosqlite.Row

    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.executescript(SCHEMA_SQL)
    await db.commit()

    _log.info("Database initialized at %s", db_path)
    return db


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

async def insert_session(db, session_id, module, started_at, status="running"):
    """Insert a new training session row."""
    await db.execute(
        "INSERT INTO sessions (session_id, module, started_at, status) VALUES (?, ?, ?, ?)",
        (session_id, module, started_at, status),
    )
    await db.commit()
    _log.debug("sessions: inserted 1 row (session_id=%s)", session_id)


async def update_session(db, session_id, **kwargs):
    """Update a session row with arbitrary column values."""
    if not kwargs:
        return
    cols = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [session_id]
    await db.execute(
        f"UPDATE sessions SET {cols} WHERE session_id = ?",
        vals,
    )
    await db.commit()
    _log.debug("sessions: updated session_id=%s (%s)", session_id, list(kwargs.keys()))


async def get_sessions(db, module=None):
    """Retrieve all sessions, optionally filtered by module."""
    if module:
        cursor = await db.execute(
            "SELECT * FROM sessions WHERE module = ? ORDER BY started_at DESC", (module,)
        )
    else:
        cursor = await db.execute("SELECT * FROM sessions ORDER BY started_at DESC")
    rows = await cursor.fetchall()
    _log.debug("sessions: fetched %d rows", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

async def insert_metrics_batch(db, rows: list[dict]):
    """Batch-insert metric rows using executemany."""
    if not rows:
        return
    await db.executemany(
        "INSERT INTO metrics (session_id, module, step, epoch, loss, reward, "
        "episode_return, grad_norm, clipped, lr, timestamp) "
        "VALUES (:session_id, :module, :step, :epoch, :loss, :reward, "
        ":episode_return, :grad_norm, :clipped, :lr, :timestamp)",
        rows,
    )
    await db.commit()
    _log.debug("metrics: inserted %d rows", len(rows))


async def get_metrics(db, session_id, module=None, step_from=0, limit=2000):
    """Retrieve metrics for a session with optional module filter and pagination."""
    if module:
        cursor = await db.execute(
            "SELECT * FROM metrics WHERE session_id = ? AND module = ? AND step >= ? "
            "ORDER BY step ASC LIMIT ?",
            (session_id, module, step_from, limit),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM metrics WHERE session_id = ? AND step >= ? "
            "ORDER BY step ASC LIMIT ?",
            (session_id, step_from, limit),
        )
    rows = await cursor.fetchall()
    _log.debug("metrics: fetched %d rows (session_id=%s)", len(rows), session_id)
    return rows


async def get_latest_metrics(db, module, limit=100):
    """Retrieve the most recent metric rows for a module (newest first)."""
    cursor = await db.execute(
        "SELECT * FROM metrics WHERE module = ? ORDER BY step DESC LIMIT ?",
        (module, limit),
    )
    rows = await cursor.fetchall()
    _log.debug("metrics: fetched %d latest rows (module=%s)", len(rows), module)
    return rows


# ---------------------------------------------------------------------------
# Decision counts helpers
# ---------------------------------------------------------------------------

async def insert_decision_counts(db, rows: list[dict]):
    """Batch-insert decision count rows."""
    if not rows:
        return
    await db.executemany(
        "INSERT INTO decision_counts (session_id, step, explore, rollback, "
        "interrupt, commit_next, nodes_expanded, search_depth, timestamp) "
        "VALUES (:session_id, :step, :explore, :rollback, :interrupt, "
        ":commit_next, :nodes_expanded, :search_depth, :timestamp)",
        rows,
    )
    await db.commit()
    _log.debug("decision_counts: inserted %d rows", len(rows))


# ---------------------------------------------------------------------------
# Embeddings helpers
# ---------------------------------------------------------------------------

async def insert_embeddings(db, rows: list[dict]):
    """Batch-insert embedding rows. z_t is stored as BLOB (numpy float32 tobytes())."""
    if not rows:
        return
    await db.executemany(
        "INSERT INTO embeddings (session_id, step, z_t, driving_context, timestamp) "
        "VALUES (:session_id, :step, :z_t, :driving_context, :timestamp)",
        rows,
    )
    await db.commit()
    _log.debug("embeddings: inserted %d rows", len(rows))


async def get_embeddings(db, session_id, limit=500):
    """Retrieve embeddings for PCA endpoint."""
    cursor = await db.execute(
        "SELECT * FROM embeddings WHERE session_id = ? ORDER BY step ASC LIMIT ?",
        (session_id, limit),
    )
    rows = await cursor.fetchall()
    _log.debug("embeddings: fetched %d rows (session_id=%s)", len(rows), session_id)
    return rows


# ---------------------------------------------------------------------------
# Predictions helpers
# ---------------------------------------------------------------------------

async def insert_predictions(db, rows: list[dict]):
    """Batch-insert prediction rows. z_next_pred/z_next_real stored as BLOBs."""
    if not rows:
        return
    await db.executemany(
        "INSERT INTO predictions (session_id, step, z_next_pred, z_next_real, mse, timestamp) "
        "VALUES (:session_id, :step, :z_next_pred, :z_next_real, :mse, :timestamp)",
        rows,
    )
    await db.commit()
    _log.debug("predictions: inserted %d rows", len(rows))


async def get_predictions(db, session_id, limit=500):
    """Retrieve predictions for scatter plot endpoint."""
    cursor = await db.execute(
        "SELECT * FROM predictions WHERE session_id = ? ORDER BY step ASC LIMIT ?",
        (session_id, limit),
    )
    rows = await cursor.fetchall()
    _log.debug("predictions: fetched %d rows (session_id=%s)", len(rows), session_id)
    return rows
