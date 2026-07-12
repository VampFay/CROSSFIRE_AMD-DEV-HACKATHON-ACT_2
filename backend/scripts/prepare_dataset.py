#!/usr/bin/env python3
"""
Prepare the CUDA-to-ROCm translation dataset for fine-tuning.

Sources:
1. HIPIFY tool output on GitHub CUDA samples (syntax-only ground truth)
2. ROCm sample ports (manually curated)
3. Synthetic pairs (LLM-generated)
4. Manual hero examples

Outputs a JSONL file with one example per line:
    {"cuda": "...", "rocm": "...", "pattern": "kernel", "difficulty": 0.4, "source": "hipify"}

Usage:
    python scripts/prepare_dataset.py --output data/cuda_rocm_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


# ============================================================
# Pattern detector (mirrors cuda_analyzer.py for tagging)
# ============================================================

def detect_pattern(cuda_source: str) -> str:
    """Detect the dominant pattern in a CUDA source."""
    if re.search(r"@\s*triton\.jit", cuda_source):
        return "triton"
    if re.search(r"\bcudnn\w+", cuda_source):
        return "cudnn"
    if re.search(r"\bcublas\w+", cuda_source):
        return "cublas"
    if re.search(r"\bthrust::", cuda_source):
        return "thrust"
    if re.search(r"__shfl_\w+_sync", cuda_source):
        return "warp_shuffle"
    if re.search(r"__shared__", cuda_source):
        return "shared_memory"
    if re.search(r"__global__", cuda_source):
        return "kernel"
    return "other"


def compute_difficulty(cuda_source: str, pattern: str) -> float:
    """Estimate translation difficulty (0-1)."""
    base = {
        "kernel": 0.2,
        "shared_memory": 0.4,
        "cublas": 0.6,
        "cudnn": 0.8,
        "thrust": 0.5,
        "warp_shuffle": 0.85,
        "triton": 0.95,
        "other": 0.3,
    }
    return base.get(pattern, 0.3)


# ============================================================
# Stub translator (for samples without ROCm reference)
# ============================================================

def stub_translate(cuda_source: str) -> str:
    """Apply simple syntactic swaps (mimics HIPIFY).
    Used as a fallback when no manual ROCm port is available.
    """
    swaps = [
        ("cudaMalloc", "hipMalloc"),
        ("cudaFree", "hipFree"),
        ("cudaMemcpy", "hipMemcpy"),
        ("cudaMemcpyToSymbol", "hipMemcpyToSymbol"),
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
        ("cudaMemset", "hipMemset"),
        ("cudaMallocPitch", "hipMallocPitch"),
        ("cudaMemcpy2D", "hipMemcpy2D"),
        ("cudaHostAlloc", "hipHostMalloc"),
        ("cudaFreeHost", "hipHostFree"),
        ("cudaDeviceReset", "hipDeviceReset"),
        ("#include <cuda_runtime.h>", "#include <hip/hip_runtime.h>"),
        ("#include <cuda.h>", "#include <hip/hip_runtime.h>"),
        ("cudaMemcpyHostToDevice", "hipMemcpyHostToDevice"),
        ("cudaMemcpyDeviceToHost", "hipMemcpyDeviceToHost"),
        ("cudaMemcpyDeviceToDevice", "hipMemcpyDeviceToDevice"),
    ]
    result = cuda_source
    for old, new in swaps:
        result = result.replace(old, new)

    # Kernel launch syntax
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
# Generate pairs from CUDA samples directory
# ============================================================

def generate_pairs_from_samples(
    samples_dir: Path,
    output_path: Path,
    source_tag: str = "sample",
):
    """Generate (cuda, rocm) pairs from a directory of .cu files.

    The ROCm version is generated via stub_translate (syntactic swaps).
    These are LOW QUALITY pairs — useful for teaching syntax but not semantics.
    """
    pairs = []
    for cu_file in sorted(samples_dir.glob("*.cu")):
        cuda_source = cu_file.read_text()
        rocm_source = stub_translate(cuda_source)
        pattern = detect_pattern(cuda_source)
        difficulty = compute_difficulty(cuda_source, pattern)

        pairs.append({
            "cuda": cuda_source,
            "rocm": rocm_source,
            "pattern": pattern,
            "difficulty": difficulty,
            "source": source_tag,
            "filename": cu_file.name,
        })

    return pairs


# ============================================================
# Manual hero examples (highest quality)
# ============================================================

HERO_EXAMPLES = [
    {
        "cuda": """#include <cuda_runtime.h>
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    float *a, *b, *c;
    cudaMalloc(&a, 1024 * sizeof(float));
    vector_add<<<4, 256>>>(a, b, c, 1024);
    cudaDeviceSynchronize();
    cudaFree(a);
}""",
        "rocm": """#include <hip/hip_runtime.h>
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    float *a, *b, *c;
    hipMalloc(&a, 1024 * sizeof(float));
    hipLaunchKernelGGL(vector_add, dim3(4), dim3(256), 0, 0, a, b, c, 1024);
    hipDeviceSynchronize();
    hipFree(a);
}""",
        "pattern": "kernel",
        "difficulty": 0.1,
        "source": "hero",
        "filename": "hero_vector_add",
    },
    {
        "cuda": """#include <cublas_v2.h>
cublasHandle_t handle;
cublasCreate(&handle);
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
cublasDestroy(handle);""",
        "rocm": """#include <hipblas/hipblas.h>
hipblasHandle_t handle;
hipblasCreate(&handle);
hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, N, M, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
hipblasDestroy(handle);""",
        "pattern": "cublas",
        "difficulty": 0.65,
        "source": "hero",
        "filename": "hero_sgemm",
    },
    {
        "cuda": """#include <cudnn_v2.h>
cudnnHandle_t handle;
cudnnCreate(&handle);
cudnnTensorDescriptor_t desc;
cudnnCreateTensorDescriptor(&desc);
cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224);
cudnnDestroyTensorDescriptor(desc);
cudnnDestroy(handle);""",
        "rocm": """#include <miopen/miopen.h>
miopenHandle_t handle;
miopenCreate(&handle);
miopenTensorDescriptor_t desc;
miopenCreateTensorDescriptor(&desc);
miopenSet4dTensorDescriptor(desc, miopenFloat, 1, 3, 224, 224);
miopenDestroyTensorDescriptor(desc);
miopenDestroy(handle);""",
        "pattern": "cudnn",
        "difficulty": 0.85,
        "source": "hero",
        "filename": "hero_cudnn",
    },
]


def generate_hero_pairs() -> list[dict]:
    """Return the hand-crafted hero examples."""
    return HERO_EXAMPLES.copy()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare CUDA-to-ROCm fine-tuning dataset")
    parser.add_argument("--output", default="data/cuda_rocm_pairs.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--samples-dir", default="../samples/cuda",
                        help="Directory of .cu sample files")
    parser.add_argument("--include-hero", action="store_true", default=True,
                        help="Include hand-crafted hero examples (highest quality)")
    parser.add_argument("--augment", type=int, default=1,
                        help="Augmentation factor for hero examples (duplicate with minor variations)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []

    # 1. Hero examples (hand-crafted, highest quality)
    if args.include_hero:
        hero_pairs = generate_hero_pairs()
        # Augment: duplicate hero examples to upweight them
        for _ in range(args.augment):
            all_pairs.extend(hero_pairs)
        logger.info(f"Added {len(hero_pairs) * args.augment} hero examples (with augmentation {args.augment}x)")

    # 2. Sample-generated pairs (lower quality, syntactic only)
    samples_dir = Path(args.samples_dir)
    if samples_dir.exists():
        sample_pairs = generate_pairs_from_samples(
            samples_dir, output_path, source_tag="sample"
        )
        all_pairs.extend(sample_pairs)
        logger.info(f"Added {len(sample_pairs)} sample-generated pairs")
    else:
        logger.warning(f"Samples directory not found: {samples_dir}")

    # 3. Roadmap: Add HIPIFY-generated pairs (requires running hipify tool)
    # 4. Roadmap: Add ROCm sample ports (requires ROCm samples repo)
    # 5. Roadmap: Add synthetic pairs (requires LLM generation)

    # Write JSONL
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # Stats
    pattern_counts = {}
    for p in all_pairs:
        pattern_counts[p["pattern"]] = pattern_counts.get(p["pattern"], 0) + 1

    logger.info("=" * 60)
    logger.info(f"Dataset prepared: {output_path}")
    logger.info(f"  Total examples: {len(all_pairs)}")
    logger.info(f"  By pattern:     {pattern_counts}")
    logger.info(f"  File size:      {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info(f"  1. Run fine-tuning: python scripts/finetune.py --dataset {output_path}")
    logger.info("  2. (Optional) Add HIPIFY pairs and ROCm sample ports to improve coverage")


if __name__ == "__main__":
    main()
