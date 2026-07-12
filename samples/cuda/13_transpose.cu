// 13_transpose.cu — Matrix transpose with shared memory
// Difficulty: 0.50 (medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 32
#define TILE 16

__global__ void transpose(const float* input, float* output, int n) {
    __shared__ float shared[TILE][TILE + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Load
    if (x < n && y < n) {
        shared[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    __syncthreads();

    // Compute transposed position
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // Store
    if (x < n && y < n) {
        output[y * n + x] = shared[threadIdx.x][threadIdx.y];
    }
}

int main() {
    float *input, *output;
    float host_input[N * N], host_output[N * N];

    for (int i = 0; i < N * N; i++) host_input[i] = (float)i;

    cudaMalloc(&input, N * N * sizeof(float));
    cudaMalloc(&output, N * N * sizeof(float));
    cudaMemcpy(input, host_input, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 threads(TILE, TILE);
    transpose<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(host_output, output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    float max_err = 0.0f;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            float err = fabsf(host_output[r * N + c] - host_input[c * N + r]);
            if (err > max_err) max_err = err;
        }
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-5 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(input); cudaFree(output);
    return 0;
}
