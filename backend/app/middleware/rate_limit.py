"""
Rate limiting middleware — protects demo from burning API credits.

Implements a simple in-memory rate limiter (token bucket per client IP).
For production, replace with Redis-backed limiter.

Default: 10 requests per minute per IP, burst of 20.
"""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter per client IP."""

    def __init__(self, app, requests_per_minute: int = 10, burst: int = 20):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        # bucket[ip] = (tokens, last_refill_time)
        self._buckets: dict[str, tuple[float, float]] = defaultdict(
            lambda: (float(burst), time.time())
        )

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and static assets
        if request.url.path in ("/health", "/", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)

        # Get client IP (check X-Forwarded-For for proxies)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"

        # Refill tokens
        now = time.time()
        tokens, last_refill = self._buckets[client_ip]
        elapsed = now - last_refill
        tokens = min(self.burst, tokens + elapsed * (self.requests_per_minute / 60.0))

        if tokens < 1:
            logger.warning(f"Rate limit exceeded for IP {client_ip}")
            return Response(
                content='{"detail": "Rate limit exceeded. Max 10 requests per minute."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        # Consume one token
        self._buckets[client_ip] = (tokens - 1, now)
        return await call_next(request)
