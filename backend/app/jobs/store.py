"""
Redis-backed job store — persists translation jobs across restarts.

Falls back to in-memory dict if Redis is unavailable (dev mode).
This replaces the old in-memory `_jobs` dict in routers/translate.py.
"""
from __future__ import annotations

import threading
from typing import Optional

from loguru import logger

from app.config import settings
from app.schemas import TranslationResult


class JobStore:
    """Persistent job store with Redis backend + in-memory fallback."""

    def __init__(self):
        self._redis = None
        self._memory: dict[str, str] = {}  # job_id -> JSON string
        self._lock = threading.Lock()
        self._use_redis = False
        self._init_redis()

    def _init_redis(self):
        """Try to connect to Redis; fall back to in-memory if unavailable."""
        try:
            import redis
            self._redis = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._redis.ping()
            self._use_redis = True
            logger.info(f"JobStore: Redis connected at {settings.redis_url}")
        except Exception as e:
            logger.warning(f"JobStore: Redis unavailable ({e}), using in-memory fallback")
            self._use_redis = False

    def _key(self, job_id: str) -> str:
        return f"crossfire:job:{job_id}"

    def set(self, job_id: str, result: TranslationResult) -> None:
        """Save a translation result."""
        data = result.model_dump_json()
        if self._use_redis:
            self._redis.setex(self._key(job_id), 86400, data)  # TTL: 24 hours
        else:
            with self._lock:
                self._memory[job_id] = data

    def get(self, job_id: str) -> Optional[TranslationResult]:
        """Get a translation result by job_id."""
        try:
            if self._use_redis:
                data = self._redis.get(self._key(job_id))
            else:
                with self._lock:
                    data = self._memory.get(job_id)

            if not data:
                return None

            return TranslationResult.model_validate_json(data)
        except Exception as e:
            logger.error(f"JobStore.get failed for {job_id}: {e}")
            return None

    def delete(self, job_id: str) -> bool:
        """Delete a job. Returns True if deleted, False if not found."""
        if self._use_redis:
            deleted = self._redis.delete(self._key(job_id))
            return deleted > 0
        else:
            with self._lock:
                if job_id in self._memory:
                    del self._memory[job_id]
                    return True
                return False

    def list(self, limit: int = 50) -> list[TranslationResult]:
        """List recent jobs (most recent first)."""
        if self._use_redis:
            keys = self._redis.keys("crossfire:job:*")
            results = []
            for key in keys[-limit:]:
                data = self._redis.get(key)
                if data:
                    try:
                        results.append(TranslationResult.model_validate_json(data))
                    except Exception:
                        pass
            return results
        else:
            with self._lock:
                items = list(self._memory.values())[-limit:]
            results = []
            for data in items:
                try:
                    results.append(TranslationResult.model_validate_json(data))
                except Exception:
                    pass
            return results

    def clear(self) -> int:
        """Clear all jobs. Returns count cleared."""
        if self._use_redis:
            keys = self._redis.keys("crossfire:job:*")
            if keys:
                return self._redis.delete(*keys)
            return 0
        else:
            with self._lock:
                count = len(self._memory)
                self._memory.clear()
                return count

    def count(self) -> int:
        """Count total jobs."""
        if self._use_redis:
            return len(self._redis.keys("crossfire:job:*"))
        else:
            with self._lock:
                return len(self._memory)


# Singleton
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get the singleton JobStore instance."""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store
