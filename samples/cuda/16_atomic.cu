// 16_atomic.cu — Atomic operations demo
// Difficulty: 0.30 (easy-medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000
#define THREADS 256

__global__ void atomic_kernel(int* counter, float* sum, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(counter, 1);
        atomicAdd(sum, input[i]);
    }
}

int main() {
    int *counter;
    float *sum, *input;
    int host_counter = 0;
    float host_sum = 0.0f;
    float* host_input = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) host_input[i] = 1.0f;

    cudaMalloc(&counter, sizeof(int));
    cudaMalloc(&sum, sizeof(float));
    cudaMalloc(&input, N * sizeof(float));

    cudaMemset(counter, 0, sizeof(int));
    cudaMemset(sum, 0, sizeof(float));
    cudaMemcpy(input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);

    atomic_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(counter, sum, input, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&host_counter, counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_sum, sum, sizeof(float), cudaMemcpyDeviceToHost);

    float err_count = fabsf((float)host_counter - (float)N);
    float err_sum = fabsf(host_sum - (float)N);

    printf("Counter: %d (expected %d), Sum: %f (expected %f)\n",
           host_counter, N, host_sum, (float)N);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"counter\": %d, \"sum\": %f, \"max_error\": %e, \"status\": \"%s\"}\n",
           host_counter, host_sum, fmaxf(err_count, err_sum),
           err_count < 1e-5 && err_sum < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(counter); cudaFree(sum); cudaFree(input);
    free(host_input);
    return 0;
}
