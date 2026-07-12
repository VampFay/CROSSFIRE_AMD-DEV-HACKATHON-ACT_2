// 11_reduction.cu — Simple block reduction with shared memory
// Difficulty: 0.40 (medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define THREADS 256

__global__ void reduce_kernel(const float* input, float* output, int n) {
    __shared__ float shared[THREADS];
    int tid = threadIdx.x;
    int i = blockIdx.x * THREADS + tid;

    shared[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared[0];
}

int main() {
    int blocks = (N + THREADS - 1) / THREADS;
    float *input, *output;
    float host_input[N], host_output[blocks];

    for (int i = 0; i < N; i++) host_input[i] = 1.0f;

    cudaMalloc(&input, N * sizeof(float));
    cudaMalloc(&output, blocks * sizeof(float));
    cudaMemcpy(input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);

    reduce_kernel<<<blocks, THREADS>>>(input, output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(host_output, output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; i++) sum += host_output[i];

    float err = fabsf(sum - (float)N);
    printf("Sum: %f, Expected: %f, Error: %e\n", sum, (float)N, err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"sum\": %f, \"expected\": %f, \"max_error\": %e, \"status\": \"%s\"}\n",
           sum, (float)N, err, err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(input); cudaFree(output);
    return 0;
}
