"""Base model client interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseModelClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self):
        self.tokens_used: int = 0
        self.cost_usd: float = 0.0

    @abstractmethod
    async def translate(
        self,
        cuda_source: str,
        rag_context: str = "",
        error_feedback: Optional[str] = None,
    ) -> str:
        """Translate CUDA source to ROCm.

        Args:
            cuda_source: CUDA source code.
            rag_context: Retrieved ROCm docs to inject as context.
            error_feedback: If retrying, the error feedback from previous iteration.

        Returns:
            Translated ROCm/HIP source code as a string.
        """
        ...

    def reset_counters(self):
        """Reset token/cost counters."""
        self.tokens_used = 0
        self.cost_usd = 0.0
