// 02_saxpy.cu — SAXPY: y = a*x + y
// Difficulty: 0.15 (easy)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 65536

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    float *x, *y;
    float host_x[N], host_y[N];
    const float a = 2.0f;

    for (int i = 0; i < N; i++) {
        host_x[i] = (float)i * 0.001f;
        host_y[i] = (float)(N - i) * 0.001f;
    }

    cudaMalloc(&x, N * sizeof(float));
    cudaMalloc(&y, N * sizeof(float));

    cudaMemcpy(x, host_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y, host_y, N * sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<(N + 255) / 256, 256>>>(a, x, y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(host_y, y, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = a * host_x[i] + ((float)(N - i) * 0.001f);
        float err = fabsf(host_y[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-5 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(x);
    cudaFree(y);
    return 0;
}
