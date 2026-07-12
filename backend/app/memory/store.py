"""
Translation memory — SQLite-backed cache of successful CUDA→ROCm translations.

When the agent successfully translates a CUDA file, the result is cached.
On subsequent requests with the same (or similar) CUDA source, the cached
translation is returned instantly — no model call, no compile, no run.

Translation memory cache for repeated requests: the first cuDNN translation
takes 90 seconds, but the second identical request returns in <100ms.

Cache key: SHA256 hash of normalized CUDA source.
Cache value: translated ROCm code + metadata (iterations, tokens, timestamp).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


# ============================================================
# Schema
# ============================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS translation_memory (
    cache_key TEXT PRIMARY KEY,
    cuda_source_hash TEXT NOT NULL,
    cuda_source TEXT NOT NULL,
    rocm_source TEXT NOT NULL,
    filename TEXT,
    patterns TEXT,  -- JSON array of pattern strings
    difficulty_score REAL,
    iterations INTEGER,
    tokens_used INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    model_used TEXT,
    max_abs_error REAL,
    created_at TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_accessed TEXT
);

CREATE INDEX IF NOT EXISTS idx_cuda_hash ON translation_memory(cuda_source_hash);
CREATE INDEX IF NOT EXISTS idx_created ON translation_memory(created_at);
"""


# ============================================================
# Normalization (so trivial whitespace changes don't bust the cache)
# ============================================================

def _normalize_source(source: str) -> str:
    """Normalize CUDA source for hashing.

    Strips comments, trailing whitespace, and collapses blank lines.
    This means minor formatting changes don't bust the cache.
    """
    # Remove single-line comments
    source = re.sub(r"//[^\n]*", "", source)
    # Remove multi-line comments
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in source.splitlines()]
    # Collapse multiple blank lines
    result = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        result.append(line)
        prev_blank = is_blank
    return "\n".join(result).strip()


def _hash_source(source: str) -> str:
    """Compute SHA256 hash of normalized source."""
    normalized = _normalize_source(source)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ============================================================
# Translation memory store
# ============================================================

class TranslationMemory:
    """SQLite-backed translation memory."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "translation_memory.db")
        self.db_path = db_path
        self._local = threading.local()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"TranslationMemory initialized at {db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def lookup(self, cuda_source: str) -> Optional[dict]:
        """Look up a cached translation.

        Returns dict with cached translation info, or None if not found.
        Updates hit_count and last_accessed on hit.
        """
        cache_key = _hash_source(cuda_source)

        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM translation_memory WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()

        if row is None:
            return None

        # Update hit count
        conn.execute(
            "UPDATE translation_memory SET hit_count = hit_count + 1, last_accessed = ? WHERE cache_key = ?",
            (datetime.utcnow().isoformat(), cache_key),
        )
        conn.commit()

        logger.info(f"Translation memory HIT for key {cache_key[:8]}... (hits: {row['hit_count'] + 1})")
        return {
            "cache_key": row["cache_key"],
            "cuda_source": row["cuda_source"],
            "rocm_source": row["rocm_source"],
            "filename": row["filename"],
            "patterns": json.loads(row["patterns"]) if row["patterns"] else [],
            "difficulty_score": row["difficulty_score"],
            "iterations": row["iterations"],
            "tokens_used": row["tokens_used"],
            "cost_usd": row["cost_usd"],
            "latency_ms": row["latency_ms"],
            "model_used": row["model_used"],
            "max_abs_error": row["max_abs_error"],
            "created_at": row["created_at"],
            "hit_count": row["hit_count"] + 1,
        }

    def store(
        self,
        cuda_source: str,
        rocm_source: str,
        filename: str = "",
        patterns: list[str] = None,
        difficulty_score: float = 0.0,
        iterations: int = 0,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        model_used: str = "local",
        max_abs_error: Optional[float] = None,
    ) -> str:
        """Store a successful translation in memory.

        Returns the cache_key.
        """
        cache_key = _hash_source(cuda_source)
        now = datetime.utcnow().isoformat()

        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO translation_memory
            (cache_key, cuda_source_hash, cuda_source, rocm_source, filename,
             patterns, difficulty_score, iterations, tokens_used, cost_usd,
             latency_ms, model_used, max_abs_error, created_at, hit_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                cache_key,
                cache_key,
                cuda_source,
                rocm_source,
                filename,
                json.dumps(patterns or []),
                difficulty_score,
                iterations,
                tokens_used,
                cost_usd,
                latency_ms,
                model_used,
                max_abs_error,
                now,
                now,
            ),
        )
        conn.commit()

        logger.info(f"Translation memory STORED for key {cache_key[:8]}...")
        return cache_key

    def list(self, limit: int = 50) -> list[dict]:
        """List recent cached translations (most recent first)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT cache_key, filename, patterns, difficulty_score, iterations, "
            "tokens_used, cost_usd, latency_ms, model_used, max_abs_error, "
            "created_at, hit_count, last_accessed "
            "FROM translation_memory ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        return [
            {
                "cache_key": row["cache_key"][:16] + "...",
                "filename": row["filename"],
                "patterns": json.loads(row["patterns"]) if row["patterns"] else [],
                "difficulty_score": row["difficulty_score"],
                "iterations": row["iterations"],
                "tokens_used": row["tokens_used"],
                "cost_usd": row["cost_usd"],
                "latency_ms": row["latency_ms"],
                "model_used": row["model_used"],
                "max_abs_error": row["max_abs_error"],
                "created_at": row["created_at"],
                "hit_count": row["hit_count"],
                "last_accessed": row["last_accessed"],
            }
            for row in rows
        ]

    def stats(self) -> dict:
        """Get memory statistics."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as count, "
            "SUM(hit_count) as total_hits, "
            "AVG(latency_ms) as avg_latency, "
            "SUM(tokens_used) as total_tokens, "
            "SUM(cost_usd) as total_cost "
            "FROM translation_memory"
        ).fetchone()

        return {
            "total_entries": row["count"] or 0,
            "total_hits": row["total_hits"] or 0,
            "avg_latency_ms": round(row["avg_latency"], 2) if row["avg_latency"] else 0,
            "total_tokens_saved": row["total_tokens"] or 0,
            "total_cost_saved_usd": round(row["total_cost"], 4) if row["total_cost"] else 0.0,
        }

    def clear(self) -> int:
        """Clear all entries. Returns count deleted."""
        conn = self._get_conn()
        result = conn.execute("DELETE FROM translation_memory")
        conn.commit()
        count = result.rowcount
        logger.info(f"Translation memory cleared: {count} entries deleted")
        return count

    def count(self) -> int:
        """Count total entries."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM translation_memory").fetchone()
        return row[0]


# ============================================================
# Singleton
# ============================================================

_memory: Optional[TranslationMemory] = None


def get_memory() -> TranslationMemory:
    """Get the singleton TranslationMemory instance."""
    global _memory
    if _memory is None:
        _memory = TranslationMemory()
    return _memory
