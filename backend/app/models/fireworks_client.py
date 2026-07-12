"""
Fireworks AI API client — for remote model calls (Gemma 27B, DeepSeek-Coder).

Used for hard semantic translations: cuDNN→MIOpen mapping, warp shuffle
primitives, custom kernel logic, Triton kernels.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI
from openai import (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.models.base import BaseModelClient


# ============================================================
# Prompt loading
# ============================================================

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / name
    if not path.exists():
        logger.warning(f"Prompt file not found: {path}")
        return ""
    return path.read_text()


# ============================================================
# Fireworks client
# ============================================================

class FireworksClient(BaseModelClient):
    """Client for Fireworks AI API (OpenAI-compatible)."""

    # Pricing (approximate, per million tokens)
    # Source: https://fireworks.ai/pricing
    PRICING = {
        "accounts/fireworks/models/gemma2-27b-it": {"input": 0.70, "output": 1.40},
        "accounts/fireworks/models/deepseek-coder-v3-instruct": {"input": 0.50, "output": 1.20},
        "default": {"input": 0.70, "output": 1.40},
    }

    def __init__(self, model: Optional[str] = None):
        super().__init__()
        self.model = model or settings.fireworks_model
        self.client = AsyncOpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=settings.fireworks_api_key,
        )
        self._simple_prompt = _load_prompt("simple_syntactic.txt")
        self._complex_prompt = _load_prompt("complex_semantic.txt")
        self._debug_prompt = _load_prompt("debug_feedback.txt")

    @retry(
        retry=retry_if_exception_type(
            (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def translate(
        self,
        cuda_source: str,
        rag_context: str = "",
        error_feedback: Optional[str] = None,
    ) -> str:
        """Translate CUDA source via Fireworks AI API.

        Raises:
            AuthenticationError: If API key is invalid (not retried).
            BadRequestError: If request is malformed (not retried).
            RateLimitError: If rate limited (retried up to 3 times).
            APITimeoutError/APIConnectionError: Network issues (retried).
            InternalServerError: Fireworks server error (retried).
        """
        # Validate API key before making the call
        if not settings.fireworks_api_key:
            raise RuntimeError(
                "FIREWORKS_API_KEY not set. Cannot call Fireworks AI API. "
                "Set it in .env or environment."
            )

        # Choose prompt based on whether this is a retry
        if error_feedback:
            system_prompt = self._debug_prompt or self._complex_prompt
        elif rag_context:
            system_prompt = self._complex_prompt
        else:
            system_prompt = self._simple_prompt or self._complex_prompt

        # Format prompt
        system = system_prompt.replace("{{RAG_CONTEXT}}", rag_context)

        user = f"Translate this CUDA code to ROCm/HIP:\n\n```cuda\n{cuda_source}\n```"

        if error_feedback:
            user += f"\n\nPrevious attempt failed. Fix these issues:\n```\n{error_feedback}\n```"

        logger.debug(f"Fireworks translate: model={self.model}, system_prompt={len(system)} chars")

        # Call API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=settings.fireworks_temperature,
            max_tokens=settings.fireworks_max_tokens,
        )

        # Update counters (guard against None usage)
        if response.usage is not None:
            self.tokens_used += response.usage.total_tokens or 0
            pricing = self.PRICING.get(self.model, self.PRICING["default"])
            self.cost_usd += (
                (response.usage.prompt_tokens or 0) * pricing["input"] / 1_000_000
                + (response.usage.completion_tokens or 0) * pricing["output"] / 1_000_000
            )

        # Guard against empty choices (rare but possible on content-filter responses)
        if not response.choices:
            logger.warning("Fireworks returned no choices (possible content filter)")
            return ""

        content = response.choices[0].message.content or ""
        translated = self._extract_code(content)

        logger.debug(f"Fireworks response: {len(translated)} chars, {self.tokens_used} tokens total, ${self.cost_usd:.4f}")
        return translated

    def _extract_code(self, content: str) -> str:
        """Extract code from markdown code blocks in the response."""
        import re

        # Try to extract from ```cpp ... ``` or ```hip ... ``` blocks (and variants)
        pattern = r"```(?:cpp|hip|cuda|c\+\+|c|hip-cpp)?\s*\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return matches[0].strip()

        # If no code block, return the whole content stripped
        return content.strip()


# ============================================================
# Smoke test
# ============================================================

async def _smoke_test():
    """Quick smoke test — run with: python -m app.models.fireworks_client"""
    if not settings.fireworks_api_key:
        print("FIREWORKS_API_KEY not set, skipping smoke test")
        return

    client = FireworksClient()
    result = await client.translate(
        cuda_source="""__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}""",
    )
    print("Translation:")
    print(result)
    print(f"\nTokens: {client.tokens_used}, Cost: ${client.cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
