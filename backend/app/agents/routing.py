"""
Routing logic — decides which model to use (local vs. remote).

ModelRouter is a singleton: model clients are created once and reused
across all translations, enabling HTTP connection pooling.
"""
from __future__ import annotations

import threading
from typing import Optional

from loguru import logger

from app.config import settings
from app.models import FireworksClient, VLLMClient
from app.schemas import AnalysisResult, ModelChoice


class ModelRouter:
    """Singleton router that caches model clients for connection reuse."""

    _instance: Optional["ModelRouter"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelRouter":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._local_model: Optional[VLLMClient] = None
        self._remote_model: Optional[FireworksClient] = None
        self._initialized = True

    def reset_counters(self):
        """Reset per-job state. Call before each translation.

        NOTE: We intentionally drop the cached model clients here rather than
        calling ``reset_counters()`` on them. In the RQ worker (separate
        process), each translation runs inside its own ``asyncio.run()``.
        ``AsyncOpenAI`` clients bind to the event loop that was active at
        construction time, so reusing a client across ``asyncio.run()`` calls
        raises "RuntimeError: Future attached to a different loop". Clearing
        the references here forces ``get_model()`` to build a fresh client
        bound to the current job's event loop. We lose HTTP connection
        pooling across jobs as a trade-off, which is acceptable for a
        hackathon. Per-job token/cost tracking is handled by the caller,
        which reads ``model.tokens_used`` after ``translate_node`` and
        accumulates it into the job's running total.
        """
        self._local_model = None
        self._remote_model = None

    def route(
        self,
        analysis: AnalysisResult,
        force_remote: bool = False,
    ) -> ModelChoice:
        """Decide which model to use based on difficulty score and config.

        IMPORTANT: If Fireworks API key is not configured, ALWAYS fall back to
        LOCAL — never route to REMOTE when it can't work.
        """
        # Check if remote is even available
        remote_available = bool(settings.fireworks_api_key)

        if force_remote and remote_available:
            logger.debug("Routing: forced to REMOTE (available)")
            return ModelChoice.REMOTE
        elif force_remote and not remote_available:
            logger.warning("Routing: force_remote=True but Fireworks not configured — using LOCAL")

        if not remote_available:
            # Remote not configured — always use local (HIPIFY + vLLM)
            logger.debug(f"Routing: LOCAL (remote not configured, difficulty={analysis.difficulty_score:.2f})")
            return ModelChoice.LOCAL

        from app.schemas import TranslationPattern
        hard_patterns = {
            TranslationPattern.CUDNN,
            TranslationPattern.TRITON,
            TranslationPattern.WARP_SHUFFLE,
        }
        if hard_patterns & set(analysis.patterns):
            logger.debug(f"Routing: REMOTE (hard patterns: {hard_patterns & set(analysis.patterns)})")
            return ModelChoice.REMOTE

        if analysis.difficulty_score >= settings.agent_routing_threshold:
            logger.debug(f"Routing: REMOTE (difficulty {analysis.difficulty_score:.2f} >= {settings.agent_routing_threshold})")
            return ModelChoice.REMOTE

        logger.debug(f"Routing: LOCAL (difficulty {analysis.difficulty_score:.2f} < {settings.agent_routing_threshold})")
        return ModelChoice.LOCAL

    def get_model(self, choice: ModelChoice):
        """Get the cached model client for the given choice."""
        if choice == ModelChoice.LOCAL:
            if self._local_model is None:
                self._local_model = VLLMClient()
            return self._local_model
        elif choice == ModelChoice.REMOTE:
            if self._remote_model is None:
                self._remote_model = FireworksClient()
            return self._remote_model
        else:
            raise ValueError(f"Unknown model choice: {choice}")

    async def close(self):
        """Close model client connections. Call on app shutdown."""
        # AsyncOpenAI clients don't need explicit close, but we clear refs
        self._local_model = None
        self._remote_model = None
        logger.info("Model clients cleared")
