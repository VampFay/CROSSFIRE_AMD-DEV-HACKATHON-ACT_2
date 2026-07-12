// 14_histogram.cu — Histogram via atomicAdd
// Difficulty: 0.40 (medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define BINS 16

__global__ void histogram_kernel(const int* input, int* histogram, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = input[i] % BINS;
        atomicAdd(&histogram[bin], 1);
    }
}

int main() {
    int *input, *histogram;
    int host_input[N], host_histogram[BINS] = {0};

    for (int i = 0; i < N; i++) host_input[i] = i;

    cudaMalloc(&input, N * sizeof(int));
    cudaMalloc(&histogram, BINS * sizeof(int));
    cudaMemcpy(input, host_input, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(histogram, 0, BINS * sizeof(int));

    histogram_kernel<<<(N + 255) / 256, 256>>>(input, histogram, N);
    cudaDeviceSynchronize();
    cudaMemcpy(host_histogram, histogram, BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU reference
    int ref[BINS] = {0};
    for (int i = 0; i < N; i++) ref[host_input[i] % BINS]++;

    int max_err = 0;
    for (int i = 0; i < BINS; i++) {
        int err = abs(host_histogram[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %d\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"histogram\": [%d", host_histogram[0]);
    for (int i = 1; i < BINS; i++) printf(", %d", host_histogram[i]);
    printf("], \"max_error\": %d, \"status\": \"%s\"}\n", max_err,
           max_err == 0 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(input); cudaFree(histogram);
    return 0;
}
