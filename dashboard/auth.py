"""
Password authentication middleware for the RASHKOGIE Training Dashboard.

Security model (per D-16, T-05-01, T-05-03):
  - DASHBOARD_PASSWORD env var controls access
  - Empty/unset password disables auth (open access)
  - Timing-safe comparison via hmac.compare_digest
  - /api/health excluded from auth for connectivity checks
"""

import hmac
import logging
import os

from fastapi import WebSocket, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_log = logging.getLogger("dashboard.auth")

# Paths excluded from password auth (must start with /api/ to be in scope)
_AUTH_EXCLUDED_PATHS = {"/api/health"}


def get_password() -> str:
    """Read dashboard password from environment. Empty string means auth disabled."""
    return os.environ.get("DASHBOARD_PASSWORD", "")


class PasswordAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware that requires Bearer token matching DASHBOARD_PASSWORD
    for all /api/ paths except excluded ones.

    When DASHBOARD_PASSWORD is empty or unset, all requests pass through.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            # Skip auth for excluded paths (e.g., /api/health)
            if request.url.path in _AUTH_EXCLUDED_PATHS:
                return await call_next(request)

            password = get_password()
            if password:
                token = (
                    request.headers.get("Authorization", "")
                    .removeprefix("Bearer ")
                    .strip()
                )
                if not hmac.compare_digest(token, password):
                    _log.warning("Unauthorized request to %s", request.url.path)
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized"},
                    )

        return await call_next(request)


async def verify_ws_token(token: str) -> None:
    """
    Verify a WebSocket authentication token.

    Raises WebSocketException with WS_1008_POLICY_VIOLATION if the token
    does not match DASHBOARD_PASSWORD. No-op when password is unset.
    """
    password = get_password()
    if password and not hmac.compare_digest(token, password):
        from fastapi import WebSocketException as WSException

        raise WSException(code=status.WS_1008_POLICY_VIOLATION)
