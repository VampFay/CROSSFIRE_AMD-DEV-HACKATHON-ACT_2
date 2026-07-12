"""
Job management routes — uses Redis-backed JobStore.

IMPORTANT: Static routes (/jobs/stats, /jobs/clear) MUST be defined BEFORE
dynamic routes (/jobs/{job_id}) so FastAPI matches them first.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.jobs.store import get_job_store
from app.schemas import JobStatusResponse, TranslationResult, VerificationLevel

router = APIRouter()


# ============================================================
# Static routes — MUST come before /jobs/{job_id}
# ============================================================

@router.get("/jobs/stats")
async def job_stats() -> dict:
    """Get job store statistics."""
    store = get_job_store()
    jobs = store.list(limit=1000)

    status_counts = {}
    verification_counts = {}
    for j in jobs:
        status_counts[j.status] = status_counts.get(j.status, 0) + 1
        vl = j.verification_level.value if hasattr(j.verification_level, "value") else str(j.verification_level)
        verification_counts[vl] = verification_counts.get(vl, 0) + 1

    return {
        "total_jobs": store.count(),
        "by_status": status_counts,
        "by_verification_level": verification_counts,
        "storage": "redis" if store._use_redis else "memory",
    }


@router.post("/jobs/clear")
async def clear_jobs() -> dict:
    """Clear all jobs (admin/debug)."""
    store = get_job_store()
    count = store.clear()
    logger.info(f"Cleared {count} jobs")
    return {"cleared": count}


@router.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs(limit: int = 50) -> List[JobStatusResponse]:
    """List recent jobs with real progress and verification level."""
    store = get_job_store()
    jobs = store.list(limit=limit)
    return [
        JobStatusResponse(
            job_id=j.job_id,
            status=j.status,
            verification_level=j.verification_level,
            progress=_compute_progress(j),
            current_state=j.status,
            iteration=len(j.iterations),
            message=j.error,
            stub_mode=j.stub_mode,
        )
        for j in jobs
    ]


def _compute_progress(job: TranslationResult) -> float:
    """Derive a 0-1 progress value from job state + iteration count."""
    if job.status == "done":
        return 1.0
    if job.status == "failed":
        return 1.0  # terminal state
    if job.cache_hit:
        return 1.0
    state_progress = {
        "queued": 0.0, "analyzing": 0.1, "translating": 0.25,
        "compiling": 0.5, "running": 0.75, "diffing": 0.9, "debugging": 0.4,
    }
    base = state_progress.get(job.status, 0.0)
    if job.budget and job.iterations:
        iter_frac = len(job.iterations) / max(job.budget.max_attempts, 1)
        base = min(base + 0.1 * iter_frac, 0.95)
    return base


# ============================================================
# Dynamic routes — MUST come after static routes
# ============================================================

@router.get("/jobs/{job_id}", response_model=TranslationResult)
async def get_job(job_id: str) -> TranslationResult:
    """Get full result of a translation job."""
    store = get_job_store()
    result = store.get(job_id)
    if result is None:
        raise HTTPException(404, f"Job {job_id} not found")
    return result


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    """Delete a job from the store."""
    store = get_job_store()
    if store.delete(job_id):
        logger.info(f"Deleted job {job_id}")
        return {"status": "deleted", "job_id": job_id}
    raise HTTPException(404, f"Job {job_id} not found")
