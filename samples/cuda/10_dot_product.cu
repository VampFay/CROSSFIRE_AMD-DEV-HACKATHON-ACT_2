// 10_dot_product.cu — Dot product via reduction
// Difficulty: 0.45 (medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 512
#define THREADS 256

__global__ void dot_kernel(const float* a, const float* b, float* partial) {
    __shared__ float shared[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * THREADS + tid;

    // Compute element-wise product
    float val = (i < N) ? a[i] * b[i] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    // Tree reduction
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Block result
    if (tid == 0) {
        partial[blockIdx.x] = shared[0];
    }
}

int main() {
    int blocks = (N + THREADS - 1) / THREADS;
    float *a, *b, *partial;
    float host_a[N], host_b[N], host_partial[blocks];

    for (int i = 0; i < N; i++) {
        host_a[i] = (float)(i + 1) * 0.01f;
        host_b[i] = (float)(N - i) * 0.01f;
    }

    cudaMalloc(&a, N * sizeof(float));
    cudaMalloc(&b, N * sizeof(float));
    cudaMalloc(&partial, blocks * sizeof(float));

    cudaMemcpy(a, host_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, host_b, N * sizeof(float), cudaMemcpyHostToDevice);

    dot_kernel<<<blocks, THREADS>>>(a, b, partial);
    cudaDeviceSynchronize();

    cudaMemcpy(host_partial, partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum partial results on CPU
    float gpu_sum = 0.0f;
    for (int i = 0; i < blocks; i++) gpu_sum += host_partial[i];

    // CPU reference
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) cpu_sum += host_a[i] * host_b[i];

    float err = fabsf(gpu_sum - cpu_sum);
    printf("GPU sum: %f, CPU sum: %f, Error: %e\n", gpu_sum, cpu_sum, err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"gpu_sum\": %f, \"cpu_sum\": %f, \"max_error\": %e, \"status\": \"%s\"}\n",
           gpu_sum, cpu_sum, err, err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(a); cudaFree(b); cudaFree(partial);
    return 0;
}
