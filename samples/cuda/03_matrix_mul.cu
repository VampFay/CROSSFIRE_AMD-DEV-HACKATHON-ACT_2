// 03_matrix_mul.cu — Tiled matrix multiplication with shared memory
// Difficulty: 0.55 (medium) — shared memory + tiling

#include <cuda_runtime.h>
#include <stdio.h>

#define N 64
#define TILE 16

__global__ void matrix_mul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;

    for (int p = 0; p < (n + TILE - 1) / TILE; ++p) {
        // Load tiles into shared memory
        if (row < n && p * TILE + tx < n) {
            As[ty][tx] = A[row * n + p * TILE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        if (col < n && p * TILE + ty < n) {
            Bs[ty][tx] = B[(p * TILE + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Compute partial sum
        for (int i = 0; i < TILE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    float *A, *B, *C;
    float host_A[N * N], host_B[N * N], host_C[N * N];

    // Initialize
    for (int i = 0; i < N * N; i++) {
        host_A[i] = (float)(i % 7) * 0.1f;
        host_B[i] = (float)(i % 5) * 0.1f;
    }

    cudaMalloc(&A, N * N * sizeof(float));
    cudaMalloc(&B, N * N * sizeof(float));
    cudaMalloc(&C, N * N * sizeof(float));

    cudaMemcpy(A, host_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    dim3 threads(TILE, TILE);
    matrix_mul_tiled<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(host_C, C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify against CPU
    float max_err = 0.0f;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            float expected = 0.0f;
            for (int k = 0; k < N; k++) {
                expected += host_A[r * N + k] * host_B[k * N + c];
            }
            float err = fabsf(host_C[r * N + c] - expected);
            if (err > max_err) max_err = err;
        }
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
