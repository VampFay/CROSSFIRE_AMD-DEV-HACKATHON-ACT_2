"""
Tests for the tree-sitter enhanced CUDA analyzer.
Verifies both tree-sitter and regex fallback modes work correctly.
"""
import pytest

from app.analyzers.cuda_analyzer import (
    CudaAnalyzer,
    _analyze_with_tree_sitter,
    _extract_includes,
    _init_tree_sitter,
    analyze_cuda,
)


# ============================================================
# Test fixtures
# ============================================================

SIMPLE_KERNEL = """
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

MULTIPLE_KERNELS = """
__global__ void kernel1(float* x) { x[0] = 1.0f; }
__global__ void kernel2(float* x) { x[0] = 2.0f; }
__global__ void kernel3(float* x) { x[0] = 3.0f; }
"""

SHARED_MEMORY_KERNEL = """
__global__ void tile_kernel(float* out) {
    __shared__ float shared[256];
    shared[threadIdx.x] = threadIdx.x;
    __syncthreads();
    out[threadIdx.x] = shared[threadIdx.x];
}
"""

WARP_SHUFFLE_KERNEL = """
__global__ void reduce_warp(float* data) {
    float val = data[threadIdx.x];
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    if (threadIdx.x == 0) data[0] = val;
}
"""

CUBLAS_CALL = """
#include <cublas_v2.h>
void foo() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64, &alpha, A, 64, B, 64, &beta, C, 64);
}
"""

CUDNN_CALL = """
#include <cudnn_v2.h>
void foo() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    cudnnTensorDescriptor_t desc;
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224);
}
"""

THRUST_CODE = """
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
void foo() {
    thrust::device_vector<float> v(1024);
    float sum = thrust::reduce(v.begin(), v.end(), 0.0f);
}
"""

TRITON_CODE = """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, n):
    pid = tl.program_id(0)
    offsets = pid * 256 + tl.arange(0, 256)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, x * 2.0, mask=mask)
"""

PYTORCH_EXT = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor foo(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Must be CUDA");
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("foo", &foo);
}
"""


# ============================================================
# Tree-sitter initialization tests
# ============================================================

class TestTreeSitterInit:

    def test_init_does_not_crash(self):
        """_init_tree_sitter should not raise even if tree-sitter not installed."""
        _init_tree_sitter()  # should not raise

    def test_analyze_with_tree_sitter_returns_dict(self):
        """_analyze_with_tree_sitter should return a dict (possibly empty)."""
        result = _analyze_with_tree_sitter(SIMPLE_KERNEL)
        assert isinstance(result, dict)


# ============================================================
# Tree-sitter analysis tests (run even if tree-sitter not available —
# the analyzer falls back to regex)
# ============================================================

class TestCudaAnalyzerEnhanced:

    def test_simple_kernel_detected(self):
        result = analyze_cuda(SIMPLE_KERNEL, "test.cu")
        assert result.kernel_count == 1
        assert result.difficulty_score > 0

    def test_multiple_kernels_counted(self):
        result = analyze_cuda(MULTIPLE_KERNELS, "multi.cu")
        assert result.kernel_count == 3

    def test_shared_memory_detected(self):
        result = analyze_cuda(SHARED_MEMORY_KERNEL, "shared.cu")
        assert result.has_shared_memory is True

    def test_warp_shuffle_detected(self):
        result = analyze_cuda(WARP_SHUFFLE_KERNEL, "warp.cu")
        assert result.has_warp_primitives is True

    def test_cublas_detected(self):
        from app.schemas import TranslationPattern
        result = analyze_cuda(CUBLAS_CALL, "cublas.cu")
        assert TranslationPattern.CUBLAS in result.patterns

    def test_cudnn_detected(self):
        from app.schemas import TranslationPattern
        result = analyze_cuda(CUDNN_CALL, "cudnn.cu")
        assert TranslationPattern.CUDNN in result.patterns

    def test_thrust_detected(self):
        from app.schemas import TranslationPattern
        result = analyze_cuda(THRUST_CODE, "thrust.cu")
        assert TranslationPattern.THRUST in result.patterns

    def test_triton_detected(self):
        from app.schemas import TranslationPattern
        result = analyze_cuda(TRITON_CODE, "triton.py")
        assert TranslationPattern.TRITON in result.patterns

    def test_pytorch_ext_detected(self):
        from app.schemas import TranslationPattern
        result = analyze_cuda(PYTORCH_EXT, "ext.cu")
        assert TranslationPattern.PYTORCH_EXTENSION in result.patterns

    def test_difficulty_in_range(self):
        """All difficulty scores must be in [0, 1]."""
        for src in [SIMPLE_KERNEL, MULTIPLE_KERNELS, SHARED_MEMORY_KERNEL,
                    WARP_SHUFFLE_KERNEL, CUBLAS_CALL, CUDNN_CALL, THRUST_CODE,
                    TRITON_CODE, PYTORCH_EXT]:
            result = analyze_cuda(src, "test.cu")
            assert 0.0 <= result.difficulty_score <= 1.0

    def test_notes_mention_analyzer_method(self):
        """Notes should mention which analyzer method was used."""
        result = analyze_cuda(SIMPLE_KERNEL, "test.cu")
        assert "tree-sitter" in result.notes or "regex" in result.notes

    def test_empty_source_does_not_crash(self):
        result = analyze_cuda("", "empty.cu")
        assert result.kernel_count == 0

    def test_include_extraction_excludes_system_headers(self):
        """System headers like stdio.h should NOT be in file_dependencies."""
        src = """
        #include <cuda_runtime.h>
        #include <stdio.h>
        #include <cublas_v2.h>
        #include "my_header.h"
        #include "local_kernel.cuh"
        """
        deps = _extract_includes(src)
        assert "my_header.h" in deps
        assert "local_kernel.cuh" in deps
        assert "stdio.h" not in deps
        assert "cuda_runtime.h" not in deps
        assert "cublas_v2.h" not in deps


# ============================================================
# Regression: ensure all 20 samples still analyze correctly
# ============================================================

class TestSampleRegression:

    def test_all_20_samples_analyze(self):
        """All 20 CUDA samples should analyze without errors."""
        from pathlib import Path
        samples_dir = Path(__file__).parent.parent.parent / "samples" / "cuda"
        sample_files = list(samples_dir.glob("*.cu"))
        assert len(sample_files) == 20

        for cu_file in sample_files:
            source = cu_file.read_text()
            result = analyze_cuda(source, cu_file.name)
            assert result is not None
            assert 0.0 <= result.difficulty_score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
