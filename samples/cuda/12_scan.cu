// 12_scan.cu — Parallel prefix sum (Blelloch scan)
// Difficulty: 0.55 (medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 512
#define THREADS 512

__global__ void scan_kernel(float* input, float* output, int n) {
    __shared__ float shared[N];
    int tid = threadIdx.x;

    if (tid < n) shared[tid] = input[tid];
    __syncthreads();

    // Up-sweep (reduction)
    int stride = 1;
    while (stride < n) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < n && (idx - stride) >= 0) {
            shared[idx] += shared[idx - stride];
        }
        stride *= 2;
        __syncthreads();
    }

    // Down-sweep
    stride = n / 2;
    while (stride > 0) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < n && (idx - stride) >= 0) {
            shared[idx + stride] += shared[idx];
        }
        stride /= 2;
        __syncthreads();
    }

    if (tid < n) output[tid] = shared[tid];
}

int main() {
    float *input, *output;
    float host_input[N], host_output[N];

    for (int i = 0; i < N; i++) host_input[i] = 1.0f;

    cudaMalloc(&input, N * sizeof(float));
    cudaMalloc(&output, N * sizeof(float));
    cudaMemcpy(input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);

    scan_kernel<<<1, THREADS>>>(input, output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(host_output, output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: prefix sum of 1,1,1,...,1 is 1,2,3,...,N
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = (float)(i + 1);
        float err = fabsf(host_output[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(input); cudaFree(output);
    return 0;
}
