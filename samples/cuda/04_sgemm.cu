// 04_sgemm.cu — cuBLAS matrix multiply
// Difficulty: 0.70 (hard) — library call translation: cuBLAS → hipBLAS

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define M 64
#define N 64
#define K 64

int main() {
    float *d_A, *d_B, *d_C;
    float host_A[M * K], host_B[K * N], host_C[M * N];
    const float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < M * K; i++) host_A[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < K * N; i++) host_B[i] = (float)(i % 5) * 0.1f;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, host_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, host_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS setup
    cublasHandle_t handle;
    cublasCreate(&handle);

    // C = alpha * A * B + beta * C
    // Note: cuBLAS uses column-major order
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N, d_A, K, &beta, d_C, N);

    cudaDeviceSynchronize();
    cudaMemcpy(host_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    // Verify against CPU
    float max_err = 0.0f;
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                // Column-major access
                expected += host_A[r + k * M] * host_B[k + c * K];
            }
            float err = fabsf(host_C[r + c * M] - expected);
            if (err > max_err) max_err = err;
        }
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
