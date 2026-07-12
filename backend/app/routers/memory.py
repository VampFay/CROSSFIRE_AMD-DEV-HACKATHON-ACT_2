"""
Translation memory routes — GET /api/memory (list), DELETE /api/memory (clear),
GET /api/memory/stats, GET /api/memory/{cache_key}.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.memory.store import get_memory

router = APIRouter()


@router.get("/memory")
async def list_memory(limit: int = 50) -> dict:
    """List cached translations (most recent first)."""
    memory = get_memory()
    entries = memory.list(limit=limit)
    return {"entries": entries, "count": len(entries), "total": memory.count()}


@router.get("/memory/stats")
async def memory_stats() -> dict:
    """Get translation memory statistics."""
    memory = get_memory()
    return memory.stats()


@router.delete("/memory")
async def clear_memory() -> dict:
    """Clear all cached translations."""
    memory = get_memory()
    count = memory.clear()
    logger.info(f"Cleared translation memory: {count} entries")
    return {"cleared": count}


@router.get("/memory/{cache_key_prefix}")
async def get_memory_entry(cache_key_prefix: str) -> dict:
    """Get a specific cached translation by key prefix.

    Note: cache_key is a SHA256 hash; we match by prefix for convenience.
    """
    memory = get_memory()
    entries = memory.list(limit=1000)

    for entry in entries:
        # entries have truncated keys like "abc123def456..."
        # we stored the full key in the DB, so this lookup is best-effort
        full_key = entry.get("cache_key", "").rstrip(".")
        if full_key.startswith(cache_key_prefix):
            return entry

    raise HTTPException(404, f"No cache entry matching prefix '{cache_key_prefix}'")
