// 15_sort.cu — Bitonic sort
// Difficulty: 0.55 (medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 512
#define THREADS 256

__global__ void bitonic_step(float* data, int j, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = i ^ j;
    if (i < ixj && ixj < n) {
        bool ascending = ((i & k) == 0);
        if ((ascending && data[i] > data[ixj]) || (!ascending && data[i] < data[ixj])) {
            float tmp = data[i];
            data[i] = data[ixj];
            data[ixj] = tmp;
        }
    }
}

int main() {
    float *data;
    float host_data[N];

    // Initialize with reverse-sorted
    for (int i = 0; i < N; i++) host_data[i] = (float)(N - i);

    cudaMalloc(&data, N * sizeof(float));
    cudaMemcpy(data, host_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Bitonic sort: each kernel call is one swap step
    for (int k = 2; k <= N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonic_step<<<(N + THREADS - 1) / THREADS, THREADS>>>(data, j, k, N);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(host_data, data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify sorted
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = (float)(i + 1);
        float err = fabsf(host_data[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-5 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(data);
    return 0;
}
