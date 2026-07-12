"""
Tests for the CUDA static analyzer.
"""
import pytest

from app.analyzers.cuda_analyzer import CudaAnalyzer, analyze_cuda


# ============================================================
# Fixtures
# ============================================================

VECTOR_ADD_CUDA = """
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *c;
    cudaMalloc(&a, N * sizeof(float));
    cudaMalloc(&b, N * sizeof(float));
    cudaMalloc(&c, N * sizeof(float));
    vector_add<<<(N+255)/256, 256>>>(a, b, c, N);
    cudaDeviceSynchronize();
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
"""

MATRIX_MUL_CUDA = """
#include <cuda_runtime.h>

__global__ void matrix_mul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    // ... tiling logic
}
"""

CUBLAS_CUDA = """
#include <cublas_v2.h>
cublasHandle_t handle;
cublasCreate(&handle);
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_A, K, d_B, N, &beta, d_C, N);
cublasDestroy(handle);
"""

CUDNN_CUDA = """
#include <cudnn_v2.h>
cudnnHandle_t handle;
cudnnCreate(&handle);
cudnnTensorDescriptor_t desc;
cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 224, 224);
cudnnDestroy(handle);
"""

WARP_SHUFFLE_CUDA = """
__global__ void reduce(float* data, int n) {
    float val = data[threadIdx.x];
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
}
"""

THRUST_CUDA = """
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
thrust::device_vector<float> v(N);
float sum = thrust::reduce(v.begin(), v.end(), 0.0f);
"""

TRITON_CUDA = """
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, n):
    pid = tl.program_id(0)
    # ...
"""

PYTORCH_EXT_CUDA = """
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
# Tests
# ============================================================

class TestCudaAnalyzer:

    def test_vector_add_difficulty_is_low(self):
        result = analyze_cuda(VECTOR_ADD_CUDA)
        assert result.difficulty_score <= 0.35
        assert result.kernel_count == 1
        assert not result.has_shared_memory
        assert not result.has_warp_primitives

    def test_matrix_mul_has_shared_memory(self):
        result = analyze_cuda(MATRIX_MUL_CUDA)
        assert result.has_shared_memory is True
        assert result.kernel_count == 1

    def test_cublas_detected(self):
        result = analyze_cuda(CUBLAS_CUDA)
        from app.schemas import TranslationPattern
        assert TranslationPattern.CUBLAS in result.patterns
        assert any("cublas" in call for call in result.library_calls)

    def test_cudnn_detected(self):
        result = analyze_cuda(CUDNN_CUDA)
        from app.schemas import TranslationPattern
        assert TranslationPattern.CUDNN in result.patterns

    def test_warp_shuffle_detected(self):
        result = analyze_cuda(WARP_SHUFFLE_CUDA)
        from app.schemas import TranslationPattern
        assert TranslationPattern.WARP_SHUFFLE in result.patterns
        assert result.has_warp_primitives is True

    def test_thrust_detected(self):
        result = analyze_cuda(THRUST_CUDA)
        from app.schemas import TranslationPattern
        assert TranslationPattern.THRUST in result.patterns

    def test_triton_detected(self):
        result = analyze_cuda(TRITON_CUDA)
        from app.schemas import TranslationPattern
        assert TranslationPattern.TRITON in result.patterns

    def test_pytorch_extension_detected(self):
        result = analyze_cuda(PYTORCH_EXT_CUDA)
        from app.schemas import TranslationPattern
        assert TranslationPattern.PYTORCH_EXTENSION in result.patterns

    def test_difficulty_in_range(self):
        """All difficulty scores must be in [0, 1]."""
        for src in [VECTOR_ADD_CUDA, MATRIX_MUL_CUDA, CUBLAS_CUDA, CUDNN_CUDA,
                    WARP_SHUFFLE_CUDA, THRUST_CUDA, TRITON_CUDA, PYTORCH_EXT_CUDA]:
            result = analyze_cuda(src)
            assert 0.0 <= result.difficulty_score <= 1.0, \
                f"{result.notes}: {result.difficulty_score}"

    def test_warp_shuffle_has_highest_difficulty(self):
        """Warp shuffle should have higher difficulty than vector_add."""
        easy = analyze_cuda(VECTOR_ADD_CUDA)
        hard = analyze_cuda(WARP_SHUFFLE_CUDA)
        assert hard.difficulty_score > easy.difficulty_score

    def test_empty_source(self):
        """Empty source should not crash."""
        result = analyze_cuda("")
        assert result.kernel_count == 0
        assert len(result.patterns) == 0

    def test_includes_extracted(self):
        """Local includes should be extracted."""
        src = """
        #include <cuda_runtime.h>
        #include "my_header.h"
        #include <cublas_v2.h>
        #include "local_kernel.cuh"
        """
        result = analyze_cuda(src)
        # System headers should NOT be in file_dependencies
        assert "my_header.h" in result.file_dependencies
        assert "local_kernel.cuh" in result.file_dependencies
        # System headers should NOT
        assert "cuda_runtime.h" not in result.file_dependencies
        assert "cublas_v2.h" not in result.file_dependencies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
