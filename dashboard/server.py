"""
FastAPI server for the RASHKOGIE Training Dashboard.

Entry point for the web application. Manages lifespan (database init/close),
mounts middleware, and will host API routes + SPA frontend.

Usage:
    python -m dashboard.server [--host 0.0.0.0] [--port 8000] [--log-level info]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from dashboard.database import init_database
from dashboard.auth import PasswordAuthMiddleware

_log = logging.getLogger("dashboard")
_DASHBOARD_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and background tasks on startup, cleanup on shutdown."""
    db = await init_database()
    app.state.db = db
    _log.info("Database initialized")
    # Collector and WS manager will be added in Plan 02
    yield
    await db.close()
    _log.info("Database closed")


app = FastAPI(title="RASHKOGIE Training Dashboard", lifespan=lifespan)
app.add_middleware(PasswordAuthMiddleware)


# Health check endpoint (no auth needed -- excluded in PasswordAuthMiddleware)
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Routes will be included here in Plan 03 and 04
# app.include_router(metrics_router, prefix="/api")
# ...

# SPA mount (LAST -- after all API routes) -- will be activated in Plan 05
# frontend_dist = _DASHBOARD_DIR / "frontend" / "dist"
# if frontend_dist.exists():
#     app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="spa")


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
