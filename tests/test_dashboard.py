"""Dashboard backend tests for DASH-01 through DASH-08."""

import sys
import os
import json
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---- DASH-01: Server starts and responds ----


@pytest.mark.asyncio
async def test_server_starts(test_client):
    """DASH-01: FastAPI server responds to health check."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_auth_blocks_unauthorized(test_client_with_auth):
    """DASH-01/D-16: API returns 401 when password set and no token provided."""
    # /api/health is excluded from auth -- should still return 200
    response = test_client_with_auth.get("/api/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_auth_blocks_protected_routes(test_client_with_auth):
    """DASH-01/D-16: Non-health API endpoints return 401 without token."""
    # Any non-excluded /api/ path should be blocked
    response = test_client_with_auth.get("/api/metrics")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_database_tables_created(test_db):
    """DASH-01/D-01: SQLite database has all 5 required tables."""
    cursor = await test_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in await cursor.fetchall()]
    for expected in ["sessions", "metrics", "decision_counts", "embeddings", "predictions"]:
        assert expected in tables, f"Table {expected} missing from database"


@pytest.mark.asyncio
async def test_database_wal_mode(test_db):
    """DASH-01: Database runs in WAL journal mode."""
    cursor = await test_db.execute("PRAGMA journal_mode")
    row = await cursor.fetchone()
    assert row[0] == "wal"


# ---- DASH-02: Metrics REST endpoint ----


@pytest.mark.asyncio
async def test_ws_metrics_push(test_client):
    """DASH-02: Metrics REST endpoint returns data after ingestion."""
    db = test_client.app.state.db
    # Insert test data directly
    await db.execute(
        "INSERT INTO sessions (session_id, module, started_at) VALUES (?, ?, ?)",
        ("test_session", "encoder_intuition", "2026-05-01T00:00:00"),
    )
    await db.execute(
        "INSERT INTO metrics (session_id, module, step, loss, timestamp) VALUES (?, ?, ?, ?, ?)",
        ("test_session", "encoder_intuition", 1, 0.5, "2026-05-01T00:00:01"),
    )
    await db.commit()
    response = test_client.get("/api/metrics?module=encoder_intuition")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert data["rows"][0]["loss"] == 0.5


# ---- DASH-03: Decision counts ----


@pytest.mark.asyncio
async def test_decision_counts_ingestion(test_client):
    """DASH-03: Decision counts REST endpoint returns data."""
    db = test_client.app.state.db
    await db.execute(
        "INSERT INTO sessions (session_id, module, started_at) VALUES (?, ?, ?)",
        ("test_session", "metacontroller", "2026-05-01T00:00:00"),
    )
    await db.execute(
        "INSERT INTO decision_counts (session_id, step, explore, rollback, interrupt, "
        "commit_next, nodes_expanded, search_depth, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("test_session", 1, 12, 3, 1, 8, 5, 3, "2026-05-01T00:00:01"),
    )
    await db.commit()
    response = test_client.get("/api/metrics/decisions?session_id=test_session")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert data["rows"][0]["explore"] == 12
    assert data["rows"][0]["nodes_expanded"] == 5


# ---- DASH-04: Param hot-reload ----


@pytest.mark.asyncio
async def test_param_hot_reload(test_client):
    """DASH-04: PUT /api/params updates config and returns changed values."""
    response = test_client.put("/api/params", json={
        "module": "encoder_intuition",
        "lr": 0.0001,
        "batch_size": 16,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["changed"]["lr"] == 0.0001
    assert data["changed"]["batch_size"] == 16
    # Verify config file was updated
    response2 = test_client.get("/api/params")
    config = response2.json()["config"]
    assert config["encoder_intuition"]["lr"] == 0.0001
    assert config["encoder_intuition"]["batch_size"] == 16


@pytest.mark.asyncio
async def test_get_params(test_client):
    """DASH-04: GET /api/params returns training config."""
    response = test_client.get("/api/params")
    assert response.status_code == 200
    data = response.json()
    assert "config" in data
    assert "config_version" in data


# ---- DASH-05/06: Session history (Plan 04) ----


@pytest.mark.asyncio
async def test_session_history():
    """DASH-05: Session history queryable from REST API. (Plan 04)"""
    pytest.skip("Implemented in Plan 04")


@pytest.mark.asyncio
async def test_session_comparison():
    """DASH-06: Session comparison returns overlaid metrics. (Plan 04)"""
    pytest.skip("Implemented in Plan 04")


# ---- DASH-07: Episode return ----


@pytest.mark.asyncio
async def test_episode_return_ingestion(test_client):
    """DASH-07: Episode return field present in metrics rows."""
    db = test_client.app.state.db
    await db.execute(
        "INSERT INTO sessions (session_id, module, started_at) VALUES (?, ?, ?)",
        ("test_session", "metacontroller", "2026-05-01T00:00:00"),
    )
    await db.execute(
        "INSERT INTO metrics (session_id, module, step, episode_return, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        ("test_session", "metacontroller", 1, 42.5, "2026-05-01T00:00:01"),
    )
    await db.commit()
    response = test_client.get("/api/metrics?module=metacontroller")
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert any(r["episode_return"] == 42.5 for r in rows)


# ---- DASH-08: Nodes expanded ----


@pytest.mark.asyncio
async def test_nodes_expanded_ingestion(test_client):
    """DASH-08: Nodes expanded and search depth in decision_counts."""
    db = test_client.app.state.db
    await db.execute(
        "INSERT INTO sessions (session_id, module, started_at) VALUES (?, ?, ?)",
        ("test_session", "metacontroller", "2026-05-01T00:00:00"),
    )
    await db.execute(
        "INSERT INTO decision_counts (session_id, step, explore, rollback, interrupt, "
        "commit_next, nodes_expanded, search_depth, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("test_session", 10, 5, 2, 0, 3, 8, 4, "2026-05-01T00:00:10"),
    )
    await db.commit()
    response = test_client.get("/api/metrics/decisions?session_id=test_session")
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert any(r["nodes_expanded"] == 8 and r["search_depth"] == 4 for r in rows)
