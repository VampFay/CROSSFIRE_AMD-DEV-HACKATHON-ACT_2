"""
FastAPI application entry point.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger

from app import __version__
from app.config import settings
from app.routers import jobs, memory, stream, translate
from app.ws.handler import router as ws_router


# ============================================================
# Lifespan — startup and shutdown hooks
# ============================================================

# Module-global so /health can report real uptime
_STARTUP_TIME: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler with graceful shutdown."""
    global _STARTUP_TIME
    logger.info(f"Starting {settings.app_name} v{__version__}")
    startup_time = time.time()
    _STARTUP_TIME = startup_time

    logger.info(f"  AMD GPU enabled: {settings.amd_gpu_enabled}")
    logger.info(f"  vLLM URL: {settings.vllm_url}")
    logger.info(f"  Fireworks configured: {bool(settings.fireworks_api_key)}")
    logger.info(f"  ChromaDB: {settings.chroma_host or 'local file mode'}")
    logger.info(f"  Sandbox: {settings.sandbox_container}")

    # Check sandbox availability (non-blocking)
    try:
        from app.sandbox.compiler import SandboxClient
        client = SandboxClient()
        if client.is_available():
            logger.info("  Sandbox: AVAILABLE (Docker mode)")
        elif client.is_direct_mode():
            logger.info("  Sandbox: AVAILABLE (Direct hipcc mode)")
        else:
            logger.warning("  Sandbox: NOT AVAILABLE")
    except Exception as e:
        logger.warning(f"  Sandbox check failed: {e}")

    # Check vLLM availability
    try:
        import httpx
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{settings.vllm_url}/health", timeout=2.0)
            if resp.status_code == 200:
                logger.info("  vLLM: AVAILABLE")
            else:
                logger.warning(f"  vLLM: status {resp.status_code}")
    except Exception as e:
        logger.warning(f"  vLLM not reachable: {e}")

    # Check HIPIFY availability
    try:
        import shutil
        if shutil.which("hipify-clang") or shutil.which("hipify-perl"):
            logger.info("  HIPIFY: AVAILABLE")
        else:
            logger.info("  HIPIFY: not on PATH (will use LLM-only mode)")
    except Exception:
        pass

    logger.info(f"Startup complete in {time.time() - startup_time:.2f}s")

    yield

    # ---- Graceful Shutdown ----
    logger.info("Shutting down gracefully...")
    try:
        from app.agents.routing import ModelRouter
        router = ModelRouter()
        await router.close()
    except Exception as e:
        logger.warning(f"Error closing model clients: {e}")

    try:
        from app.agents.checkpointer import close_checkpointer
        await close_checkpointer()
    except Exception:
        pass

    try:
        from app.jobs.store import get_job_store
        store = get_job_store()
        if store._use_redis and store._redis is not None:
            # redis-py async clients use aclose(); sync use close()
            aclose = getattr(store._redis, "aclose", None)
            if aclose:
                await aclose()
            else:
                store._redis.close()
    except Exception:
        pass

    logger.info("Shutdown complete.")


# ============================================================
# App
# ============================================================

app = FastAPI(
    title=settings.app_name,
    description="""
**Crossfire** — Autonomous CUDA-to-ROCm Translation Agent

Takes a CUDA codebase, produces a validated, compiled, and benchmarked ROCm
equivalent running on AMD MI300X GPUs. Built for AMD Developer Hackathon: ACT II.
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---- Middleware (order matters: auth runs INSIDE rate-limit so failed auth
# doesn't burn rate-limit tokens for legit users sharing a NAT IP) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

from app.middleware.basic_auth import BasicAuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
# RateLimit is the OUTERMOST middleware (runs first), Auth is INSIDE it.
app.add_middleware(BasicAuthMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=10, burst=20)


# ============================================================
# Routes
# ============================================================

@app.get("/", tags=["meta"])
async def root():
    """Root endpoint — basic info."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "ui": "/ui",
    }


@app.get("/health", tags=["meta"], response_model=None)
async def health():
    """Health check — used by Docker and load balancers. Async throughout."""
    import httpx
    import shutil
    import subprocess

    from app.schemas import HealthResponse

    amd_available = False
    vllm_available = False
    sandbox_available = False
    hipify_available = False

    # Check AMD GPU (async via to_thread)
    async def _check_amd():
        try:
            result = await asyncio.to_thread(
                subprocess.run, ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=2,
            )
            return result.returncode == 0
        except Exception:
            return False

    # Check sandbox (async via to_thread)
    async def _check_sandbox():
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "ps", "--filter", f"name={settings.sandbox_container}", "-q"],
                capture_output=True, text=True, timeout=2,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    amd_available = await _check_amd()
    sandbox_available = await _check_sandbox()

    # Direct-mode sandbox (hipcc on PATH) also counts as available
    if not sandbox_available and shutil.which("hipcc"):
        sandbox_available = True

    # Check vLLM (already async)
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{settings.vllm_url}/health", timeout=2.0)
            vllm_available = resp.status_code == 200
    except Exception:
        pass

    # Check HIPIFY
    hipify_available = bool(shutil.which("hipify-clang") or shutil.which("hipify-perl"))

    # Real uptime from module-global set in lifespan
    uptime = time.time() - _STARTUP_TIME if _STARTUP_TIME else 0.0

    return HealthResponse(
        status="ok",
        version=__version__,
        amd_gpu_available=amd_available,
        vllm_available=vllm_available,
        fireworks_configured=bool(settings.fireworks_api_key),
        chroma_available=settings.chroma_available,  # ← was `or True` (always True)
        sandbox_available=sandbox_available,
        hipify_available=hipify_available,
        uptime_seconds=uptime,
    )


# ---- Routers ----
app.include_router(translate.router, prefix="/api", tags=["translation"])
app.include_router(jobs.router, prefix="/api", tags=["jobs"])
app.include_router(memory.router, prefix="/api", tags=["memory"])
app.include_router(stream.router, prefix="/api", tags=["streaming"])
app.include_router(ws_router, tags=["websocket"])


# ============================================================
# Standalone UI (served from backend/app/ui/index.html)
# ============================================================

@app.get("/ui", tags=["ui"])
async def ui_redirect():
    """Redirect /ui to /ui/ for the static UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/")


@app.get("/ui/", tags=["ui"])
async def ui_index():
    """Serve the standalone Crossfire UI (no Node.js build required).

    Includes cache-busting headers so browsers always fetch the latest version.
    """
    from pathlib import Path
    from fastapi.responses import HTMLResponse
    ui_path = Path(__file__).parent / "ui" / "index.html"
    if ui_path.exists():
        return HTMLResponse(
            ui_path.read_text(),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return HTMLResponse("<h1>UI not found</h1>", status_code=404)


# ============================================================
# Exception handlers
# ============================================================

from fastapi import Request
from fastapi.responses import JSONResponse


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning",
    )
