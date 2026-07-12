"""
Translation routes — POST /api/translate (always async, returns job_id immediately).

The endpoint NEVER blocks. It either:
1. Enqueues to RQ worker (production, with Redis)
2. Starts a background asyncio task (dev, without Redis)

Frontend polls GET /api/jobs/{job_id} or subscribes to:
- WebSocket: /ws/jobs/{job_id}
- SSE: /api/stream/{job_id}
"""
from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from app.config import settings
from app.jobs.store import get_job_store
from app.schemas import (
    TranslateRepoRequest,
    TranslateRequest,
    TranslationResult,
)

router = APIRouter()


@router.post("/translate")
async def translate(
    request: TranslateRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Translate a single CUDA file to ROCm.

    Always returns immediately with a job_id. Never blocks the HTTP request.
    The frontend should connect to WebSocket /ws/jobs/{job_id} for live updates,
    or poll GET /api/jobs/{job_id} for the result.
    """
    job_id = str(uuid.uuid4())

    logger.info(
        f"Translate request: job_id={job_id}, filename={request.filename}, "
        f"size={len(request.cuda_source)} chars"
    )

    max_iters = request.max_iterations or settings.agent_max_iterations

    # Try RQ first (production with Redis)
    rq_enqueued = False
    try:
        import redis
        from rq import Queue

        redis_conn = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        redis_conn.ping()

        queue = Queue("crossfire-queue", connection=redis_conn)
        from app.jobs.translate_task import translate_task

        queue.enqueue(
            translate_task,
            job_id=job_id,
            cuda_source=request.cuda_source,
            filename=request.filename,
            force_remote=request.force_remote,
            max_iterations=max_iters,
            job_timeout=300,
            result_ttl=86400,
        )
        rq_enqueued = True
        logger.info(f"Enqueued RQ job {job_id}")
    except Exception as e:
        logger.warning(f"RQ unavailable ({e}), using background task")

    # Fallback: run as FastAPI background task (non-blocking)
    if not rq_enqueued:
        async def _run_translation():
            """Background translation task — does not block the HTTP response."""
            try:
                from app.agents.graph import run_agent
                result = await run_agent(
                    job_id=job_id,
                    cuda_source=request.cuda_source,
                    filename=request.filename,
                    force_remote=request.force_remote,
                    max_iterations=max_iters,
                )
                get_job_store().set(job_id, result)
                logger.info(f"Background job {job_id} complete: {result.status}")
            except Exception as e:
                logger.exception(f"Background job {job_id} failed: {e}")
                from app.schemas import JobStatus
                from datetime import datetime
                result = TranslationResult(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    cuda_source=request.cuda_source,
                    error=str(e),
                    completed_at=datetime.utcnow(),
                )
                get_job_store().set(job_id, result)

        background_tasks.add_task(_run_translation)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Translation queued. Connect to WebSocket or poll for updates.",
        "poll_url": f"/api/jobs/{job_id}",
        "ws_url": f"/ws/jobs/{job_id}",
        "sse_url": f"/api/stream/{job_id}",
    }


@router.post("/translate-sync")
async def translate_sync(request: TranslateRequest) -> TranslationResult:
    """
    Synchronous translation — blocks until complete.

    Use only for small files or testing. For production, use POST /translate
    which returns immediately with a job_id.
    """
    job_id = str(uuid.uuid4())
    from app.agents.graph import run_agent

    try:
        result = await run_agent(
            job_id=job_id,
            cuda_source=request.cuda_source,
            filename=request.filename,
            force_remote=request.force_remote,
            max_iterations=request.max_iterations or settings.agent_max_iterations,
        )
        get_job_store().set(job_id, result)
        return result
    except Exception as e:
        logger.exception(f"Sync translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/translate-repo")
async def translate_repo(request: TranslateRepoRequest) -> Dict[str, Any]:
    """Translate an entire GitHub repository (batch)."""
    from app.utils.repo_cloner import fetch_repo_contents, parse_github_url

    try:
        owner, repo = parse_github_url(request.repo_url)
    except ValueError as e:
        raise HTTPException(400, str(e))

    batch_id = str(uuid.uuid4())
    logger.info(f"Repo translate request: batch={batch_id}, repo={owner}/{repo}")

    try:
        repo_data = await fetch_repo_contents(request.repo_url, request.file_pattern)

        if repo_data["file_count"] == 0:
            return {
                "batch_id": batch_id,
                "repo": f"{owner}/{repo}",
                "file_pattern": request.file_pattern,
                "status": "no_files",
                "message": f"No CUDA files found matching pattern '{request.file_pattern}'",
            }

        store = get_job_store()
        job_ids = []

        for file_info in repo_data["files"]:
            job_id = str(uuid.uuid4())
            job_ids.append(job_id)

            rq_ok = False
            try:
                import redis
                from rq import Queue
                redis_conn = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                queue = Queue("crossfire-queue", connection=redis_conn)
                from app.jobs.translate_task import translate_task
                queue.enqueue(
                    translate_task,
                    job_id=job_id,
                    cuda_source=file_info["source"],
                    filename=file_info["filename"],
                    job_timeout=300,
                    result_ttl=86400,
                )
                rq_ok = True
            except Exception:
                pass

            if not rq_ok:
                # Inline for dev (will block, but repo translation is less frequent)
                from app.agents.graph import run_agent
                result = await run_agent(
                    job_id=job_id,
                    cuda_source=file_info["source"],
                    filename=file_info["filename"],
                )
                store.set(job_id, result)

        return {
            "batch_id": batch_id,
            "repo": f"{owner}/{repo}",
            "file_pattern": request.file_pattern,
            "status": "queued",
            "file_count": repo_data["file_count"],
            "job_ids": job_ids,
            "message": f"Queued {len(job_ids)} CUDA files for translation.",
        }

    except Exception as e:
        logger.exception(f"Repo translation failed: {e}")
        raise HTTPException(500, f"Repo translation failed: {str(e)}")


@router.get("/samples")
async def list_samples() -> Dict[str, Any]:
    """List available CUDA sample programs."""
    from pathlib import Path

    samples_dir = Path(__file__).parent.parent.parent.parent / "samples" / "cuda"
    if not samples_dir.exists():
        return {"samples": []}

    samples = []
    for f in sorted(samples_dir.glob("*.cu")):
        size = f.stat().st_size
        samples.append({
            "filename": f.name,
            "size_bytes": size,
            "size_human": f"{size/1024:.1f} KB" if size > 1024 else f"{size} B",
            "path": f"/api/samples/{f.name}",
        })

    return {"samples": samples, "count": len(samples)}


@router.get("/samples/{filename}")
async def get_sample(filename: str) -> Dict[str, str]:
    """Get a CUDA sample by filename."""
    from pathlib import Path

    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")

    samples_dir = Path(__file__).parent.parent.parent.parent / "samples" / "cuda"
    sample_path = samples_dir / filename

    if not sample_path.exists():
        raise HTTPException(404, f"Sample {filename} not found")

    return {"filename": filename, "source": sample_path.read_text()}
