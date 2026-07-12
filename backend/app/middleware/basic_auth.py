"""
Basic auth middleware — protects demo from unauthorized access.

When DEMO_BASIC_AUTH_USER and DEMO_BASIC_AUTH_PASS are set in env,
all requests must include basic auth credentials.

Disable by leaving both env vars empty (default — open demo).
"""
from __future__ import annotations

import base64
import os
import secrets

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """HTTP Basic Auth middleware.

    Only enforces auth when DEMO_BASIC_AUTH_USER and DEMO_BASIC_AUTH_PASS
    are both set in the environment. Otherwise, all requests pass through.
    """

    # Paths that don't require auth (health checks, etc.)
    PUBLIC_PATHS = frozenset({"/health"})

    def __init__(self, app):
        super().__init__(app)
        self.username = os.environ.get("DEMO_BASIC_AUTH_USER", "")
        self.password = os.environ.get("DEMO_BASIC_AUTH_PASS", "")
        self.enabled = bool(self.username and self.password)
        if self.enabled:
            logger.info(f"Basic auth enabled (user: {self.username})")
        else:
            logger.info("Basic auth disabled (open demo)")

    async def dispatch(self, request: Request, call_next):
        # Skip if auth disabled
        if not self.enabled:
            return await call_next(request)

        # Skip public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return self._unauthorized_response()

        try:
            # Decode credentials
            encoded = auth_header[6:]  # strip "Basic "
            decoded = base64.b64decode(encoded).decode("utf-8")
            username, password = decoded.split(":", 1)

            # Constant-time comparison to prevent timing attacks
            user_ok = secrets.compare_digest(username, self.username)
            pass_ok = secrets.compare_digest(password, self.password)

            if not (user_ok and pass_ok):
                logger.warning(f"Auth failed for username: {username}")
                return self._unauthorized_response()

        except Exception as e:
            logger.warning(f"Auth header parse error: {e}")
            return self._unauthorized_response()

        return await call_next(request)

    def _unauthorized_response(self) -> Response:
        return Response(
            content='{"detail": "Unauthorized. Provide valid Basic auth credentials."}',
            status_code=401,
            media_type="application/json",
            headers={"WWW-Authenticate": 'Basic realm="Crossfire Demo"'},
        )
