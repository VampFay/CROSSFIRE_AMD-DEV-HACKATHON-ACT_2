// 01_vector_add.cu — Simplest possible CUDA kernel
// Difficulty: 0.10 (easy) — pure syntactic translation

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float *a, *b, *c;
    float host_a[N], host_b[N], host_c[N];

    // Initialize inputs
    for (int i = 0; i < N; i++) {
        host_a[i] = (float)i;
        host_b[i] = (float)(i * 2);
    }

    // Allocate device memory
    cudaMalloc(&a, N * sizeof(float));
    cudaMalloc(&b, N * sizeof(float));
    cudaMalloc(&c, N * sizeof(float));

    // Copy to device
    cudaMemcpy(a, host_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    vector_add<<<(N + 255) / 256, 256>>>(a, b, c, N);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(host_c, c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = host_a[i] + host_b[i];
        float err = fabsf(host_c[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);
    printf("Result: %s\n", max_err < 1e-5 ? "PASS" : "FAIL");

    // Output for diff harness (JSON to stdout)
    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-5 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    // Cleanup
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
