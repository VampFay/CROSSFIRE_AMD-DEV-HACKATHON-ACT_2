"""
Custom exception hierarchy for structured error handling.

Convention:
- Backend functions raise specific exceptions
- API boundary catches TranslationError subclasses, returns structured HTTP error
- Never return None for error cases — raise
"""
from __future__ import annotations

from typing import Optional


class TranslationError(Exception):
    """Base exception for all translation-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None, suggestion: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion


class CompileError(TranslationError):
    """Raised when hipcc compilation fails."""
    pass


class RunError(TranslationError):
    """Raised when running the translated binary fails."""
    pass


class ValidationError(TranslationError):
    """Raised when numerical diff against baseline fails."""
    pass


class ModelUnavailableError(TranslationError):
    """Raised when the LLM model (local or remote) is unavailable."""
    pass


class SandboxUnavailableError(TranslationError):
    """Raised when the ROCm sandbox container is not running."""
    pass


class AnalysisError(TranslationError):
    """Raised when static analysis of CUDA source fails."""
    pass


class BudgetExceeded(TranslationError):
    """Raised when a job budget limit is hit. Stops the agent loop immediately."""
    pass


class UnsupportedInputError(TranslationError):
    """Raised when input CUDA contains constructs outside the supported capability matrix."""
    pass


# ============================================================
# Error classification for frontend hints
# ============================================================

ERROR_CATEGORIES = {
    "missing_hip_include": {
        "hint": "Add #include <hip/hip_runtime.h> at the top of the file.",
        "severity": "compile",
    },
    "cudnn_miopen_mismatch": {
        "hint": "MIOpen API differs from cuDNN. Check RAG context for correct signature. MIOpen requires workspace pre-allocation.",
        "severity": "compile",
    },
    "cublas_hipblas_mismatch": {
        "hint": "cuBLAS maps to hipBLAS. Check that CUBLAS_OP_N is replaced with HIPBLAS_OP_N.",
        "severity": "compile",
    },
    "runtime_segfault": {
        "hint": "Segmentation fault. Likely a memory access issue — check array bounds and pointer initialization.",
        "severity": "runtime",
    },
    "runtime_timeout": {
        "hint": "Execution timed out. The kernel may have an infinite loop or deadlock.",
        "severity": "runtime",
    },
    "diff_numerical_mismatch": {
        "hint": "Output values differ from CUDA baseline. Check algorithm correctness — floating point associativity may differ.",
        "severity": "diff",
    },
    "diff_missing_output": {
        "hint": "Expected output key not found. The translated code may not be printing results in the correct format.",
        "severity": "diff",
    },
    "model_unavailable": {
        "hint": "Model server is not running. Check that vLLM or Fireworks AI is accessible.",
        "severity": "model",
    },
}


def classify_error(error: str) -> tuple[str, Optional[str]]:
    """Classify an error string into a category and return a hint.

    Returns (category_key, hint) or ("unknown", None).
    """
    error_lower = error.lower()

    if "hip/hip_runtime" in error_lower or "hip_runtime.h" in error_lower:
        return "missing_hip_include", ERROR_CATEGORIES["missing_hip_include"]["hint"]

    if "cudnn" in error_lower and ("miopen" in error_lower or "undefined reference" in error_lower):
        return "cudnn_miopen_mismatch", ERROR_CATEGORIES["cudnn_miopen_mismatch"]["hint"]

    if "cublas" in error_lower and ("hipblas" in error_lower or "undefined reference" in error_lower):
        return "cublas_hipblas_mismatch", ERROR_CATEGORIES["cublas_hipblas_mismatch"]["hint"]

    if "segmentation fault" in error_lower or "sigsegv" in error_lower:
        return "runtime_segfault", ERROR_CATEGORIES["runtime_segfault"]["hint"]

    if "timeout" in error_lower:
        return "runtime_timeout", ERROR_CATEGORIES["runtime_timeout"]["hint"]

    if "max_abs_error" in error_lower or "mismatch" in error_lower:
        return "diff_numerical_mismatch", ERROR_CATEGORIES["diff_numerical_mismatch"]["hint"]

    if "no baseline" in error_lower or "__no_baseline__" in error_lower:
        return "diff_missing_output", ERROR_CATEGORIES["diff_missing_output"]["hint"]

    if "connection" in error_lower or "unavailable" in error_lower or "refused" in error_lower:
        return "model_unavailable", ERROR_CATEGORIES["model_unavailable"]["hint"]

    return "unknown", None
