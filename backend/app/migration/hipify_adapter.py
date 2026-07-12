"""
HIPIFY adapter — deterministic CUDA-to-HIP translation.

Two-tier approach:
  1. Try hipify-clang (AST-based, most accurate) — needs CUDA headers
  2. Fallback to regex_hipify (deterministic regex swaps) — always works

The LLM is only invoked when BOTH methods fail to produce compilable code.
This is dramatically more reliable than asking a 7B model to rewrite CUDA
from scratch.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger


def _find_hipify() -> Optional[str]:
    """Find hipify-clang or hipify-perl on PATH."""
    for tool in ["hipify-clang", "hipify-perl"]:
        path = shutil.which(tool)
        if path:
            return path
    for candidate in ["/opt/rocm/bin/hipify-clang", "/opt/rocm/hipify/bin/hipify-clang"]:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


async def hipify_file(cuda_source: str, filename: str = "input.cu") -> Optional[str]:
    """
    Translate CUDA to HIP deterministically.

    Tries hipify-clang first (AST-based). If that fails (e.g., missing CUDA
    headers), falls back to regex_hipify. Only returns None if both fail.

    Returns:
        HIP source code, or None if both methods failed.
    """
    hipify_bin = _find_hipify()

    # Try hipify-clang first
    if hipify_bin is not None:
        result = await _try_hipify_clang(hipify_bin, cuda_source, filename)
        if result is not None:
            logger.info("HIPIFY: hipify-clang succeeded")
            return result
        logger.warning("HIPIFY: hipify-clang failed, falling back to regex_hipify")
    else:
        logger.debug("HIPIFY: hipify-clang not on PATH, using regex fallback")

    # Fallback: regex_hipify (always works, no CUDA headers needed)
    try:
        result = regex_hipify(cuda_source)
        if result and result.strip():
            logger.info(
                f"HIPIFY: regex_hipify succeeded ({len(cuda_source)} → {len(result)} chars)"
            )
            return result
    except Exception as e:
        logger.warning(f"HIPIFY: regex_hipify failed: {e}")

    return None


async def _try_hipify_clang(
    hipify_bin: str, cuda_source: str, filename: str
) -> Optional[str]:
    """Try hipify-clang. Returns HIP source or None on failure."""
    tmp_dir = Path(tempfile.gettempdir())
    job_id = uuid.uuid4().hex[:8]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    in_path = tmp_dir / f"{job_id}_{safe_name}"
    out_path = tmp_dir / f"{job_id}_hipified.{safe_name.replace('.cu', '.hip')}"

    try:
        in_path.write_text(cuda_source)
    except OSError as e:
        logger.warning(f"HIPIFY: failed to write temp file: {e}")
        return None

    try:
        if "hipify-clang" in hipify_bin:
            cmd = [
                hipify_bin,
                str(in_path),
                "-o", str(out_path),
                "--no-cuda-include",
                "--examine",
            ]
        else:
            cmd = [hipify_bin, str(in_path)]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            err = stderr.decode(errors="replace")[:500]
            logger.warning(f"HIPIFY: hipify-clang failed (rc={proc.returncode}): {err}")
            return None

        if "hipify-clang" in hipify_bin:
            if out_path.exists():
                hip_source = out_path.read_text()
            else:
                logger.warning("HIPIFY: output file not created")
                return None
        else:
            hip_source = stdout.decode(errors="replace")

        if not hip_source.strip():
            logger.warning("HIPIFY: produced empty output")
            return None

        # Ensure hip runtime include is present
        if "#include <hip/hip_runtime.h>" not in hip_source:
            if "#include <cuda_runtime.h>" in hip_source:
                hip_source = hip_source.replace(
                    "#include <cuda_runtime.h>",
                    "#include <hip/hip_runtime.h>",
                )
            elif "#include <cuda.h>" in hip_source:
                hip_source = hip_source.replace(
                    "#include <cuda.h>",
                    "#include <hip/hip_runtime.h>",
                )
            else:
                hip_source = "#include <hip/hip_runtime.h>\n" + hip_source

        logger.info(
            f"HIPIFY: hipify-clang success: {len(cuda_source)} → {len(hip_source)} chars"
        )
        return hip_source

    except asyncio.TimeoutError:
        logger.warning("HIPIFY: hipify-clang timed out (30s)")
        return None
    except Exception as e:
        logger.warning(f"HIPIFY: hipify-clang exception: {e}")
        return None
    finally:
        for p in [in_path, out_path]:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


def hipify_available() -> bool:
    """Quick check if hipify is installed (or regex fallback is available)."""
    return _find_hipify() is not None or True  # regex_hipify always available


# ============================================================
# Regex-based HIPIFY fallback (deterministic, no CUDA headers needed)
# ============================================================

CUDA_HIP_SWAPS: list[tuple[str, str]] = [
    # Headers
    ("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
    ("#include <cuda.h>", "#include <hip/hip_runtime.h>"),
    ("#include <cuda_runtime_api.h>", "#include <hip/hip_runtime_api.h>"),
    ("#include <cublas_v2.h>", "#include <hipblas/hipblas.h>"),
    ("#include <cublas.h>", "#include <hipblas/hipblas.h>"),
    ("#include <cufft.h>", "#include <hipfft/hipfft.h>"),
    ("#include <cufftw.h>", "#include <hipfft/hipfftw.h>"),
    ("#include <cudnn.h>", "#include <miopen/miopen.h>"),
    ("#include <curand.h>", "#include <hiprand/hiprand.h>"),
    ("#include <nvrtc.h>", "#include <hiprtc.h>"),

    # Runtime API
    ("cudaMalloc", "hipMalloc"),
    ("cudaFree", "hipFree"),
    ("cudaMemcpy", "hipMemcpy"),
    ("cudaMemset", "hipMemset"),
    ("cudaMallocManaged", "hipMallocManaged"),
    ("cudaDeviceSynchronize", "hipDeviceSynchronize"),
    ("cudaGetLastError", "hipGetLastError"),
    ("cudaError_t", "hipError_t"),
    ("cudaSuccess", "hipSuccess"),
    ("cudaErrorMemoryAllocation", "hipErrorMemoryAllocation"),

    # Streams & events
    ("cudaStreamCreate", "hipStreamCreate"),
    ("cudaStreamDestroy", "hipStreamDestroy"),
    ("cudaStreamSynchronize", "hipStreamSynchronize"),
    ("cudaStreamWaitEvent", "hipStreamWaitEvent"),
    ("cudaEventCreate", "hipEventCreate"),
    ("cudaEventDestroy", "hipEventDestroy"),
    ("cudaEventRecord", "hipEventRecord"),
    ("cudaEventSynchronize", "hipEventSynchronize"),
    ("cudaEventElapsedTime", "hipEventElapsedTime"),

    # Device management
    ("cudaSetDevice", "hipSetDevice"),
    ("cudaGetDevice", "hipGetDevice"),
    ("cudaGetDeviceCount", "hipGetDeviceCount"),
    ("cudaGetDeviceProperties", "hipGetDeviceProperties"),
    ("cudaDeviceReset", "hipDeviceReset"),
    ("cudaDeviceGetAttribute", "hipDeviceGetAttribute"),

    # Memory copy modes
    ("cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"),
    ("cudaMemcpyDeviceToHost", "hipMemcpyDeviceToHost"),
    ("cudaMemcpyDeviceToDevice", "hipMemcpyDeviceToDevice"),

    # cuBLAS → hipBLAS
    ("cublasHandle_t", "hipblasHandle_t"),
    ("cublasCreate", "hipblasCreate"),
    ("cublasDestroy", "hipblasDestroy"),
    ("cublasSgemm", "hipblasSgemm"),
    ("cublasDgemm", "hipblasDgemm"),
    ("cublasGemmEx", "hipblasGemmEx"),
    ("cublasSetStream", "hipblasSetStream"),
    ("CUBLAS_OP_N", "HIPBLAS_OP_N"),
    ("CUBLAS_OP_T", "HIPBLAS_OP_T"),
    ("CUBLAS_OP_C", "HIPBLAS_OP_C"),
    ("CUBLAS_STATUS_SUCCESS", "HIPBLAS_STATUS_SUCCESS"),

    # cuFFT → hipFFT
    ("cufftHandle", "hipfftHandle"),
    ("cufftPlan1d", "hipfftPlan1d"),
    ("cufftPlan2d", "hipfftPlan2d"),
    ("cufftPlan3d", "hipfftPlan3d"),
    ("cufftExecC2C", "hipfftExecC2C"),
    ("cufftExecZ2Z", "hipfftExecZ2Z"),
    ("cufftDestroy", "hipfftDestroy"),
    ("CUFFT_C2C", "HIPFFT_C2C"),
    ("CUFFT_Z2Z", "HIPFFT_Z2Z"),

    # cuDNN → MIOpen (partial — semantics differ)
    ("cudnnHandle_t", "miopenHandle_t"),
    ("cudnnCreate", "miopenCreate"),
    ("cudnnDestroy", "miopenDestroy"),
    ("cudnnSetStream", "miopenSetStream"),

    # Types
    ("cudaDeviceProp", "hipDeviceProp_t"),
]


def regex_hipify(cuda_source: str) -> str:
    """Apply regex-based CUDA→HIP swaps. Always works, no CUDA headers needed.

    This is a deterministic mechanical translation that handles the most
    common CUDA patterns. It's NOT as accurate as AST-based hipify-clang
    but works reliably for standard CUDA code.
    """
    result = cuda_source
    for old, new in CUDA_HIP_SWAPS:
        result = result.replace(old, new)

    # Kernel launch syntax: kernel<<<blocks, threads>>>(args)
    # HIP supports <<<>>> natively — leave as-is. hipcc accepts it directly.
    return result
