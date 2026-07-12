// 18_events.cu — CUDA events for timing
// Difficulty: 0.30 (easy-medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 20)

__global__ void scale_kernel(float* data, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

int main() {
    float *data;
    float* host_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) host_data[i] = (float)i;

    cudaMalloc(&data, N * sizeof(float));
    cudaMemcpy(data, host_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Time the kernel
    cudaEventRecord(start);
    scale_kernel<<<(N + 255) / 256, 256>>>(data, 2.0f, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaMemcpy(host_data, data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = (float)i * 2.0f;
        if (fabsf(host_data[i] - expected) > max_err) max_err = fabsf(host_data[i] - expected);
    }
    printf("Kernel time: %.3f ms, Max error: %e\n", elapsed_ms, max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"kernel_time_ms\": %f, \"max_error\": %e, \"status\": \"%s\"}\n",
           elapsed_ms, max_err, max_err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(data);
    free(host_data);
    return 0;
}
