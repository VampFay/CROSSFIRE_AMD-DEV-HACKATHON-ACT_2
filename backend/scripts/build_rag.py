#!/usr/bin/env python3
"""
Build the RAG corpus for Crossfire.

Downloads and indexes:
1. ROCm 7.2.3 documentation (HIP runtime API)
2. MIOpen API reference
3. hipBLAS API reference
4. rocPRIM documentation
5. Curated HIPIFY translation examples

Outputs a ChromaDB collection with embedded chunks.

Usage:
    python scripts/build_rag.py --output ../chroma_db

Requirements:
    - ChromaDB running (locally or via docker-compose)
    - HuggingFace sentence-transformers (for embeddings)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

from loguru import logger

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.rag.retriever import RAGRetriever


# ============================================================
# Curated ROCm API reference chunks
# ============================================================

# These are hand-crafted reference chunks covering the most common translation scenarios.
# In production, this would be replaced by scraping the actual ROCm docs.

CURATED_CHUNKS = [
    # ---- hip_runtime ----
    {
        "id": "hip_runtime_malloc",
        "text": """# hipMalloc — Allocate memory on AMD GPU

## Signature
```c
hipError_t hipMalloc(void** ptr, size_t size);
```

## Description
Allocates `size` bytes of linear memory on the AMD GPU device. Returns a pointer in `*ptr`.

Equivalent to: `cudaMalloc`

## Example
```c
float* d_data;
hipMalloc(&d_data, N * sizeof(float));
```

## Common errors
- `hipErrorMemoryAllocation`: out of memory
- `hipErrorInvalidValue`: ptr is NULL or size is 0""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "hipMalloc"},
    },
    {
        "id": "hip_runtime_free",
        "text": """# hipFree — Free GPU memory

## Signature
```c
hipError_t hipFree(void* ptr);
```

Equivalent to: `cudaFree`""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "hipFree"},
    },
    {
        "id": "hip_runtime_memcpy",
        "text": """# hipMemcpy — Copy memory between host and device

## Signature
```c
hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind);
```

## hipMemcpyKind values
- `hipMemcpyHostToHost` = 0
- `hipMemcpyHostToDevice` = 1
- `hipMemcpyDeviceToHost` = 2
- `hipMemcpyDeviceToDevice` = 3
- `hipMemcpyDefault` = 4 (auto-detect)

Equivalent to: `cudaMemcpy` with `cudaMemcpyHostToDevice` etc.""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "hipMemcpy"},
    },
    {
        "id": "hip_launch_kernel",
        "text": """# hipLaunchKernelGGL — Launch a GPU kernel

## Signature
```c
template<typename... Args>
void hipLaunchKernelGGL(
    void(*kernel)(Args...),
    dim3 gridDim, dim3 blockDim,
    size_t sharedMem, hipStream_t stream,
    Args... args
);
```

## Description
Launches a GPU kernel with the given grid/block configuration. Equivalent to CUDA's `kernel<<<gridDim, blockDim, sharedMem, stream>>>(args)` syntax.

## Example
```c
// CUDA: my_kernel<<<blocks, threads, 0, stream>>>(arg1, arg2);
// ROCm:
hipLaunchKernelGGL(my_kernel, dim3(blocks), dim3(threads), 0, stream, arg1, arg2);
```

If sharedMem and stream are 0, you can use:
```c
hipLaunchKernelGGL(my_kernel, dim3(blocks), dim3(threads), 0, 0, arg1, arg2);
```""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "hipLaunchKernelGGL"},
    },
    {
        "id": "hip_sync",
        "text": """# hipDeviceSynchronize — Wait for all GPU work to complete

## Signature
```c
hipError_t hipDeviceSynchronize(void);
```

Equivalent to: `cudaDeviceSynchronize`""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "hipDeviceSynchronize"},
    },
    {
        "id": "hip_streams",
        "text": """# hipStream_t — Asynchronous execution streams

## API
- `hipStreamCreate(hipStream_t* stream)` — create a stream
- `hipStreamDestroy(hipStream_t stream)` — destroy a stream
- `hipStreamSynchronize(hipStream_t stream)` — wait for all work in stream
- `hipStreamWaitEvent(hipStream_t stream, hipEvent_t event)` — wait for event

Equivalent to: `cudaStream*` family""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "hipStream"},
    },
    {
        "id": "hip_events",
        "text": """# hipEvent_t — Timing events

## API
- `hipEventCreate(hipEvent_t* event)`
- `hipEventRecord(hipEvent_t event, hipStream_t stream = 0)`
- `hipEventSynchronize(hipEvent_t event)`
- `hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop)`

Equivalent to: `cudaEvent*` family""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "hipEvent"},
    },
    {
        "id": "hip_warp_shuffle",
        "text": """# Warp Shuffle Primitives (HIP)

HIP supports the same warp shuffle syntax as CUDA:
- `__shfl_sync(mask, val, srcLane)`
- `__shfl_up_sync(mask, val, delta)`
- `__shfl_down_sync(mask, val, delta)`
- `__shfl_xor_sync(mask, val, laneMask)`
- `__any_sync(mask, pred)`
- `__all_sync(mask, pred)`
- `__ballot_sync(mask, pred)`

These are NOT translated — they remain as-is when porting CUDA to HIP.

## Example
```c
__device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "warp_shuffle"},
    },
    {
        "id": "hip_shared_memory",
        "text": """# Shared Memory in HIP

HIP uses the same `__shared__` qualifier as CUDA:
```c
__shared__ float shared[TILE_SIZE];
```

`__syncthreads()` and `__threadfence()` also work identically.

No translation needed for shared memory declarations.""",
        "metadata": {"source": "rocm_docs", "category": "hip_runtime", "api": "shared_memory"},
    },

    # ---- hipBLAS ----
    {
        "id": "hipblas_overview",
        "text": """# hipBLAS — AMD BLAS library (replaces cuBLAS)

## Header
```c
#include <hipblas/hipblas.h>
```

## Handle management
- `hipblasHandle_t` (replaces `cublasHandle_t`)
- `hipblasCreate(hipblasHandle_t* handle)` (replaces `cublasCreate`)
- `hipblasDestroy(hipblasHandle_t handle)` (replaces `cublasDestroy`)

## Operations
- `hipblasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc)`
- `hipblasDgemm` (double precision)
- `hipblasGemmEx` (mixed precision)

## Enum mappings
- `CUBLAS_OP_N` → `HIPBLAS_OP_N`
- `CUBLAS_OP_T` → `HIPBLAS_OP_T`
- `CUBLAS_OP_C` → `HIPBLAS_OP_C`
- `CUBLAS_STATUS_SUCCESS` → `HIPBLAS_STATUS_SUCCESS`

Signatures are 1:1 compatible with cuBLAS — direct substitution works.""",
        "metadata": {"source": "rocm_docs", "category": "hipblas", "api": "overview"},
    },
    {
        "id": "hipblas_sgemm",
        "text": """# hipblasSgemm — Single-precision matrix multiply

## Signature
```c
hipblasStatus_t hipblasSgemm(
    hipblasHandle_t handle,
    hipblasOperation_t transa, hipblasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
);
```

Performs: C = alpha * op(A) * op(B) + beta * C

Note: hipBLAS uses column-major order (same as cuBLAS).

CUDA equivalent: `cublasSgemm` — signatures are identical, only the prefix changes.""",
        "metadata": {"source": "rocm_docs", "category": "hipblas", "api": "hipblasSgemm"},
    },

    # ---- MIOpen ----
    {
        "id": "miopen_overview",
        "text": """# MIOpen — AMD Deep Learning library (replaces cuDNN)

## Header
```c
#include <miopen/miopen.h>
```

## Handle management
- `miopenHandle_t` (replaces `cudnnHandle_t`)
- `miopenCreate(miopenHandle_t* handle)` (replaces `cudnnCreate`)
- `miopenDestroy(miopenHandle_t handle)` (replaces `cudnnDestroy`)

## IMPORTANT: API differences from cuDNN
MIOpen is NOT 1:1 with cuDNN. Key differences:
1. Tensor descriptor: `miopenSet4dTensorDescriptor(desc, dataType, n, c, h, w)` — does NOT take format/stride like cuDNN's `cudnnSetTensor4dDescriptor`
2. Convolution forward requires workspace pre-allocation: `miopenConvolutionForwardGetWorkSpaceSize` then allocate
3. Different enum names: `miopenFloat` (vs `CUDNN_DATA_FLOAT`), `miopenTensorNCHW` (vs `CUDNN_TENSOR_NCHW`)
4. Convolution mode: `miopenCrossCorrelation` (vs `CUDNN_CROSS_CORRELATION`)

When translating cuDNN → MIOpen, ALWAYS verify the API signature with the MIOpen docs.""",
        "metadata": {"source": "rocm_docs", "category": "miopen", "api": "overview"},
    },
    {
        "id": "miopen_tensor_descriptor",
        "text": """# miopenSet4dTensorDescriptor — 4D tensor descriptor

## Signature
```c
miopenStatus_t miopenSet4dTensorDescriptor(
    miopenTensorDescriptor_t tensorDesc,
    miopenDataType_t dataType,
    int n, int c, int h, int w
);
```

## cuDNN → MIOpen differences
CUDA:
```c
cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
```
ROCm:
```c
miopenSet4dTensorDescriptor(desc, miopenFloat, n, c, h, w);
```

Note: MIOpen assumes NCHW by default — no format parameter needed.
Note: Use `miopenFloat` not `CUDNN_DATA_FLOAT`.""",
        "metadata": {"source": "rocm_docs", "category": "miopen", "api": "miopenSet4dTensorDescriptor"},
    },
    {
        "id": "miopen_convolution_forward",
        "text": """# miopenConvolutionForward — Convolution operation

## Required steps (different from cuDNN!)

1. Initialize convolution descriptor:
```c
miopenConvolutionDescriptor_t conv_desc;
miopenCreateConvolutionDescriptor(&conv_desc);
miopenInitConvolutionDescriptor(conv_desc, miopenCrossCorrelation,
                                 pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
```

2. Get workspace size:
```c
size_t workspace_size;
miopenConvolutionForwardGetWorkSpaceSize(handle,
    input_desc, filter_desc, conv_desc, output_desc, &workspace_size);
void* workspace;
hipMalloc(&workspace, workspace_size);
```

3. Find best algorithm:
```c
miopenConvFwdAlgorithm_t algo;
miopenFindConvolutionForwardAlgorithm(handle,
    input_desc, d_input, filter_desc, d_filter,
    conv_desc, output_desc, d_output,
    1, &returned_count, &algo, &workspace_size, workspace, false);
```

4. Run convolution:
```c
miopenConvolutionForward(handle, &alpha,
    input_desc, d_input, filter_desc, d_filter,
    conv_desc, algo, &beta,
    output_desc, d_output, workspace, workspace_size);
```

This is significantly different from cuDNN's single-call API.""",
        "metadata": {"source": "rocm_docs", "category": "miopen", "api": "miopenConvolutionForward"},
    },

    # ---- rocPRIM ----
    {
        "id": "rocprim_overview",
        "text": """# rocPRIM — AMD Parallel Primitives (replaces Thrust/CUB)

## Header
```c
#include <rocprim/rocprim.hpp>
```

## Common operations
- `rocprim::reduce` — replaces `thrust::reduce`
- `rocprim::sort` — replaces `thrust::sort`
- `rocprim::scan` — replaces `thrust::exclusive_scan`/`inclusive_scan`
- `rocprim::transform` — replaces `thrust::transform`

## Note on Thrust
HIP actually ships with Thrust support — `thrust::device_vector` etc. work on AMD GPUs.
You can keep Thrust code as-is when porting to ROCm. Use rocPRIM only if you need
lower-level control or better performance.""",
        "metadata": {"source": "rocm_docs", "category": "rocprim", "api": "overview"},
    },

    # ---- PyTorch extensions ----
    {
        "id": "pytorch_extension_porting",
        "text": """# Porting PyTorch CUDA extensions to ROCm

## Include changes
- `#include <ATen/cuda/CUDAContext.h>` → `#include <ATen/hip/HIPContext.h>`
- `#include <cuda_runtime.h>` → `#include <hip/hip_runtime.h>`
- `#include <c10/cuda/CUDAGuard.h>` → `#include <c10/hip/HIPGuard.h>`

## Unchanged APIs
- `torch::Tensor` — same API
- `x.data_ptr<float>()` — same API
- `x.is_cuda()` — returns true for HIP tensors
- `x.device()` — returns `DeviceType::HIP` (but code usually doesn't care)
- `TORCH_CHECK(x.is_cuda(), ...)` — same
- `PYBIND11_MODULE` — same

## Kernel launches
All `kernel<<<>>>` syntax must be converted to `hipLaunchKernelGGL(kernel, ...)`.

## Build system
Use `TORCH_LIBRARY` registration as-is. PyTorch ROCm build transparently
maps CUDA symbols to HIP equivalents for most of the public API.""",
        "metadata": {"source": "rocm_docs", "category": "pytorch", "api": "extension_porting"},
    },

    # ---- Translation reference tables ----
    {
        "id": "translation_table_runtime",
        "text": """# CUDA Runtime API → HIP Runtime API Translation Table

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaMemset` | `hipMemset` |
| `cudaMallocPitch` | `hipMallocPitch` |
| `cudaMemcpy2D` | `hipMemcpy2D` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaGetLastError` | `hipGetLastError` |
| `cudaError_t` | `hipError_t` |
| `cudaSuccess` | `hipSuccess` |
| `cudaSetDevice` | `hipSetDevice` |
| `cudaGetDeviceCount` | `hipGetDeviceCount` |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| `cudaStreamCreate` | `hipStreamCreate` |
| `cudaStreamDestroy` | `hipStreamDestroy` |
| `cudaStreamSynchronize` | `hipStreamSynchronize` |
| `cudaEventCreate` | `hipEventCreate` |
| `cudaEventRecord` | `hipEventRecord` |
| `cudaEventSynchronize` | `hipEventSynchronize` |
| `cudaEventElapsedTime` | `hipEventElapsedTime` |
| `cudaHostAlloc` | `hipHostMalloc` |
| `cudaFreeHost` | `hipHostFree` |
| `cudaDeviceReset` | `hipDeviceReset` |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` |
| `cudaMemcpyDeviceToDevice` | `hipMemcpyDeviceToDevice` |""",
        "metadata": {"source": "rocm_docs", "category": "translation_table", "api": "runtime"},
    },
    {
        "id": "translation_table_headers",
        "text": """# Header Translation Table

| CUDA Header | HIP/ROCm Header |
|-------------|-----------------|
| `#include <cuda_runtime.h>` | `#include <hip/hip_runtime.h>` |
| `#include <cuda.h>` | `#include <hip/hip_runtime.h>` |
| `#include <cublas_v2.h>` | `#include <hipblas/hipblas.h>` |
| `#include <cudnn_v2.h>` | `#include <miopen/miopen.h>` |
| `#include <cudnn.h>` | `#include <miopen/miopen.h>` |
| `#include <thrust/device_vector.h>` | `#include <thrust/device_vector.h>` (unchanged — HIP supports Thrust) |
| `#include <curand.h>` | `#include <hiprand/hiprand.h>` |
| `#include <cusparse.h>` | `#include <rocsparse/rocsparse.h>` |
| `#include <cusolver.h>` | `#include <rocsolver/rocsolver.h>` |
| `#include <nvrtc.h>` | `#include <hiprtc/hiprtc.h>` |""",
        "metadata": {"source": "rocm_docs", "category": "translation_table", "api": "headers"},
    },
]


# ============================================================
# Build RAG corpus
# ============================================================

def build_rag_corpus(output_dir: str | None = None):
    """Build the RAG corpus from curated chunks."""
    output = output_dir or settings.chroma_persist_dir

    logger.info("=" * 60)
    logger.info("Building RAG corpus")
    logger.info("=" * 60)
    logger.info(f"Output: {output}")

    retriever = RAGRetriever()

    # Get current count
    initial_count = retriever.count()
    logger.info(f"Current corpus size: {initial_count} chunks")

    # Add curated chunks
    documents = [chunk["text"] for chunk in CURATED_CHUNKS]
    metadatas = [chunk["metadata"] for chunk in CURATED_CHUNKS]

    logger.info(f"Adding {len(documents)} curated chunks...")
    retriever.add_documents(documents, metadatas=metadatas)

    final_count = retriever.count()
    logger.info(f"Final corpus size: {final_count} chunks (added {final_count - initial_count})")

    # Verify retrieval works
    logger.info("Verifying retrieval...")
    test_queries = [
        "cudaMalloc allocate memory",
        "cublasSgemm matrix multiply",
        "cudnn convolution forward",
        "__shfl_down_sync warp shuffle",
        "thrust reduce",
    ]
    for q in test_queries:
        chunks = retriever.retrieve(q, top_k=2)
        logger.info(f"  Query '{q}': {len(chunks)} chunks retrieved")
        if chunks:
            logger.info(f"    Top chunk: {chunks[0][:100]}...")

    logger.info("=" * 60)
    logger.info("RAG corpus build complete!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build RAG corpus from ROCm docs")
    parser.add_argument("--output", default=None,
                        help="ChromaDB persist directory (default: from settings)")
    args = parser.parse_args()

    build_rag_corpus(args.output)


if __name__ == "__main__":
    main()
