// 07_pytorch_ext.cu — PyTorch C++ extension with CUDA kernel
// Difficulty: 0.70 (hard) — real-world pattern

// Note: This file cannot be compiled standalone (requires PyTorch headers).
// For the hackathon demo, the agent should produce a translated .hip file
// that compiles with PyTorch ROCm build.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA kernel
__global__ void double_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= 2.0f;
    }
}

// PyTorch-visible function
torch::Tensor double_tensor(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Input must be float32");

    auto out = x.clone();
    int n = out.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    double_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(), n
    );

    return out;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("double_tensor", &double_tensor, "Double all values in a tensor");
}
