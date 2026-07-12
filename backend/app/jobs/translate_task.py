"""
RQ background task for translation jobs.

Uses asyncio.run() for clean event loop management — no resource leaks.
"""
from __future__ import annotations

import asyncio

from loguru import logger

from app.agents.graph import run_agent
from app.jobs.store import get_job_store
from app.schemas import TranslationResult


def translate_task(
    job_id: str,
    cuda_source: str,
    filename: str = "input.cu",
    force_remote: bool = False,
    max_iterations: int = 5,
) -> str:
    """Background translation task (runs in RQ worker).

    Uses asyncio.run() which creates and cleans up the event loop properly.
    No resource leaks across multiple jobs.

    Returns:
        job_id (for RQ result tracking).
    """
    logger.info(f"[RQ] Starting translation job {job_id} for {filename}")

    try:
        result: TranslationResult = asyncio.run(
            run_agent(
                job_id=job_id,
                cuda_source=cuda_source,
                filename=filename,
                force_remote=force_remote,
                max_iterations=max_iterations,
            )
        )

        # Persist to job store
        store = get_job_store()
        store.set(job_id, result)

        logger.info(
            f"[RQ] Job {job_id} complete: status={result.status}, "
            f"iterations={len(result.iterations)}"
        )
        return job_id

    except Exception as e:
        logger.exception(f"[RQ] Job {job_id} failed: {e}")

        # Store failure result
        result = TranslationResult(
            job_id=job_id,
            status="failed",
            cuda_source=cuda_source,
            error=f"Background task error: {e}",
        )
        get_job_store().set(job_id, result)
        return job_id


def health_check_task() -> dict:
    """Simple health check task for RQ worker."""
    return {"status": "ok", "worker": "crossfire"}
