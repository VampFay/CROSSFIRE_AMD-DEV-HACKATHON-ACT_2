"""
Tests for the vLLM stub translator (used when vLLM is not running).
"""
import pytest

from app.models.vllm_client import VLLMClient


SIMPLE_CUDA = """
#include <cuda_runtime.h>

__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *c;
    cudaMalloc(&a, 1024 * sizeof(float));
    cudaMalloc(&b, 1024 * sizeof(float));
    cudaMalloc(&c, 1024 * sizeof(float));
    add<<<4, 256>>>(a, b, c, 1024);
    cudaDeviceSynchronize();
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
"""


class TestStubTranslator:

    def test_stub_replaces_cuda_malloc(self):
        client = VLLMClient()
        result = client._stub_translate(SIMPLE_CUDA)
        assert "hipMalloc" in result
        assert "cudaMalloc" not in result

    def test_stub_replaces_cuda_free(self):
        client = VLLMClient()
        result = client._stub_translate(SIMPLE_CUDA)
        assert "hipFree" in result
        assert "cudaFree" not in result

    def test_stub_replaces_device_sync(self):
        client = VLLMClient()
        result = client._stub_translate(SIMPLE_CUDA)
        assert "hipDeviceSynchronize" in result

    def test_stub_replaces_kernel_launch(self):
        client = VLLMClient()
        result = client._stub_translate(SIMPLE_CUDA)
        assert "hipLaunchKernelGGL" in result
        assert "<<<" not in result  # CUDA syntax removed

    def test_stub_replaces_header(self):
        client = VLLMClient()
        result = client._stub_translate(SIMPLE_CUDA)
        assert "<hip/hip_runtime.h>" in result
        assert "<cuda_runtime.h>" not in result

    def test_stub_preserves_kernel_signature(self):
        """The kernel function itself should be unchanged."""
        client = VLLMClient()
        result = client._stub_translate(SIMPLE_CUDA)
        assert "__global__ void add(float* a, float* b, float* c, int n)" in result

    def test_stub_preserves_thread_indexing(self):
        client = VLLMClient()
        result = client._stub_translate(SIMPLE_CUDA)
        assert "blockIdx.x * blockDim.x + threadIdx.x" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
