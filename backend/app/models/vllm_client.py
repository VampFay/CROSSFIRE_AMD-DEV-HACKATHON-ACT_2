"""
vLLM client — for local model calls (Gemma 4 12B via vLLM).

Served via vLLM on AMD MI300X. Local tokens are FREE per hackathon rules.

CRITICAL DESIGN RULES (Phase 0 of the rebuild):
  1. ALLOW_STUB_FALLBACK defaults to FALSE. A regex swap is NOT a translation.
  2. Connection errors ARE retried by the @retry decorator (the local try/except
     no longer swallows them — it only catches AFTER retries are exhausted).
  3. _load_prompt raises on missing file — an empty system prompt is a bug.
  4. used_stub is set on the client so callers can propagate it to the result.
"""
from __future__ import annotations

import asyncio
import os
import re
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


PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt file. Raises FileNotFoundError if missing — empty prompts are bugs."""
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    return path.read_text()


class VLLMClient(BaseModelClient):
    """Client for local vLLM server (OpenAI-compatible API)."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__()
        self.model = model or settings.vllm_model
        self.base_url = base_url or settings.vllm_url
        normalized_url = self.base_url.rstrip("/")
        if normalized_url.endswith("/v1"):
            normalized_url = normalized_url[:-3]
        self.client = AsyncOpenAI(
            base_url=f"{normalized_url}/v1",
            api_key="EMPTY",
            timeout=60.0,
            max_retries=0,  # we use tenacity for retries
        )
        try:
            self._simple_prompt = _load_prompt("simple_syntactic.txt")
            self._complex_prompt = _load_prompt("complex_semantic.txt")
            self._debug_prompt = _load_prompt("debug_feedback.txt")
        except FileNotFoundError as e:
            logger.error(f"Prompt file missing: {e}. Using minimal fallback.")
            self._simple_prompt = "You are a CUDA-to-ROCm translator. Output only valid HIP C++ code."
            self._complex_prompt = self._simple_prompt
            self._debug_prompt = self._simple_prompt
        self.used_stub: bool = False

    @retry(
        retry=retry_if_exception_type(
            (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _call_vllm(
        self,
        system: str,
        user: str,
    ) -> str:
        """Internal: call vLLM with retry. Returns extracted code. Raises on exhaustion.

        NOTE: Gemma does not support the 'system' role. We merge the system
        prompt into the user message instead.
        """
        # Check if model is Gemma (doesn't support system role)
        is_gemma = "gemma" in self.model.lower()

        if is_gemma:
            # Merge system prompt into user message
            merged_user = f"{system}\n\n---\n\n{user}" if system else user
            messages = [{"role": "user", "content": merged_user}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=settings.vllm_temperature,
            max_tokens=settings.vllm_max_tokens,
        )

        if response.usage is not None:
            self.tokens_used += response.usage.total_tokens or 0
        self.cost_usd += 0.0  # local = free

        if not response.choices:
            logger.warning("vLLM returned no choices")
            return ""

        content = response.choices[0].message.content or ""
        return self._extract_code(content)

    async def translate(
        self,
        cuda_source: str,
        rag_context: str = "",
        error_feedback: Optional[str] = None,
    ) -> str:
        """Translate CUDA source via local vLLM server.

        Behavior:
          - Retries connection/timeout errors 3x with exponential backoff.
          - After retries exhausted: if ALLOW_STUB_FALLBACK=true (dev only),
            returns a regex-swap stub and sets used_stub=True.
          - Otherwise re-raises — the agent loop fails visibly. We NEVER
            silently fabricate a "translation".
        """
        if error_feedback:
            system_prompt = self._debug_prompt or self._simple_prompt
        else:
            system_prompt = self._simple_prompt

        system = system_prompt.replace("{{RAG_CONTEXT}}", rag_context)
        user = f"Translate this CUDA code to ROCm/HIP:\n\n```cuda\n{cuda_source}\n```"
        if error_feedback:
            user += f"\n\nPrevious attempt failed. Fix these issues:\n```\n{error_feedback}\n```"

        logger.debug(f"vLLM translate: model={self.model}, prompt={len(system)} chars")
        self.used_stub = False

        try:
            translated = await self._call_vllm(system, user)
            logger.debug(
                f"vLLM response: {len(translated)} chars, {self.tokens_used} tokens total (FREE)"
            )
            return translated
        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
            # Retries exhausted — only NOW consider stub fallback
            if self._allow_stub_fallback():
                logger.warning(
                    f"vLLM unreachable after retries ({type(e).__name__}: {e}). "
                    f"ALLOW_STUB_FALLBACK=true → STUB translator. "
                    f"THIS IS NOT A REAL MODEL TRANSLATION — result will be flagged stub_mode=True."
                )
                self.used_stub = True
                return self._stub_translate(cuda_source)
            logger.error(f"vLLM unreachable after retries and stub fallback disabled: {e}")
            raise

    def _allow_stub_fallback(self) -> bool:
        """Stub fallback is OFF by default. Enable explicitly for dev only."""
        return os.environ.get("ALLOW_STUB_FALLBACK", "false").lower() in ("true", "1", "yes")

    async def translate_stream(
        self,
        cuda_source: str,
        rag_context: str = "",
        error_feedback: Optional[str] = None,
    ):
        """Stream translation tokens from vLLM. Yields token strings."""
        system_prompt = self._debug_prompt if error_feedback else self._simple_prompt
        system = system_prompt.replace("{{RAG_CONTEXT}}", rag_context)
        user = f"Translate this CUDA code to ROCm/HIP:\n\n```cuda\n{cuda_source}\n```"
        if error_feedback:
            user += f"\n\nPrevious attempt failed. Fix these issues:\n```\n{error_feedback}\n```"

        self.used_stub = False

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=settings.vllm_temperature,
                max_tokens=settings.vllm_max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                # Capture usage from the final chunk
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    self.tokens_used += chunk.usage.total_tokens or 0

        except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
            if self._allow_stub_fallback():
                self.used_stub = True
                yield self._stub_translate(cuda_source)
            else:
                raise

    def _extract_code(self, content: str) -> str:
        """Extract code from markdown code blocks."""
        pattern = r"```(?:cpp|hip|cuda|c\+\+|cpp17|hip-cpp|rocm|c)?\s*\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return matches[0].strip()
        return content.strip()

    def _stub_translate(self, cuda_source: str) -> str:
        """
        Stub translator — simple syntactic swaps. DEV ONLY.
        Used when vLLM is unreachable AND ALLOW_STUB_FALLBACK=true.
        The result is flagged with stub_mode=True so the UI never lies.
        """
        logger.warning("Using STUB translator (no vLLM available). Result will be flagged stub_mode=True.")
        result = cuda_source

        swaps = [
            ("cudaMalloc", "hipMalloc"),
            ("cudaFree", "hipFree"),
            ("cudaMemcpy", "hipMemcpy"),
            ("cudaDeviceSynchronize", "hipDeviceSynchronize"),
            ("cudaGetLastError", "hipGetLastError"),
            ("cudaError_t", "hipError_t"),
            ("cudaSuccess", "hipSuccess"),
            ("cudaStreamCreate", "hipStreamCreate"),
            ("cudaStreamDestroy", "hipStreamDestroy"),
            ("cudaStreamSynchronize", "hipStreamSynchronize"),
            ("cudaEventCreate", "hipEventCreate"),
            ("cudaEventRecord", "hipEventRecord"),
            ("cudaEventSynchronize", "hipEventSynchronize"),
            ("cudaEventElapsedTime", "hipEventElapsedTime"),
            ("cudaSetDevice", "hipSetDevice"),
            ("cudaGetDeviceCount", "hipGetDeviceCount"),
            ("cudaGetDeviceProperties", "hipGetDeviceProperties"),
            ("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
            ("#include <cuda.h>", "#include <hip/hip_runtime.h>"),
            ("#include <cublas_v2.h>", "#include <hipblas/hipblas.h>"),
            ("cublasHandle_t", "hipblasHandle_t"),
            ("cublasCreate", "hipblasCreate"),
            ("cublasDestroy", "hipblasDestroy"),
            ("cublasSgemm", "hipblasSgemm"),
            ("cublasDgemm", "hipblasDgemm"),
            ("CUBLAS_OP_N", "HIPBLAS_OP_N"),
            ("CUBLAS_OP_T", "HIPBLAS_OP_T"),
            ("CUBLAS_OP_C", "HIPBLAS_OP_C"),
        ]

        for old, new in swaps:
            result = result.replace(old, new)

        launch_pattern = re.compile(
            r"(\w+)\s*<<<\s*([^,>]+)\s*,\s*([^,>]+)\s*(?:,\s*([^,>]+))?\s*>>>\s*\(([^)]*)\)"
        )

        def replacer(m):
            kernel = m.group(1)
            blocks = m.group(2).strip()
            threads = m.group(3).strip()
            args = m.group(5).strip()
            return f"hipLaunchKernelGGL({kernel}, dim3({blocks}), dim3({threads}), 0, 0, {args})"

        result = launch_pattern.sub(replacer, result)
        return result


# ============================================================
# Smoke test
# ============================================================

async def _smoke_test():
    """Run with: python -m app.models.vllm_client"""
    client = VLLMClient()
    result = await client.translate(
        cuda_source="""__global__ void vector_add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *c;
    cudaMalloc(&a, 1024 * sizeof(float));
    cudaMalloc(&b, 1024 * sizeof(float));
    cudaMalloc(&c, 1024 * sizeof(float));
    vector_add<<<4, 256>>>(a, b, c, 1024);
    cudaDeviceSynchronize();
    return 0;
}""",
    )
    print("Translation:")
    print(result)
    print(f"\nTokens: {client.tokens_used}, Cost: ${client.cost_usd:.4f}, Stub: {client.used_stub}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
