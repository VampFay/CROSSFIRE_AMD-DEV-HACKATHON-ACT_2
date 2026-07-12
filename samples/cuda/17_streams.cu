// 17_streams.cu — CUDA streams for concurrent execution
// Difficulty: 0.35 (easy-medium)

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define NUM_STREAMS 4

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    cudaStream_t streams[NUM_STREAMS];
    float *a[NUM_STREAMS], *b[NUM_STREAMS], *c[NUM_STREAMS];
    float host_a[N], host_b[N], host_c[N];

    for (int i = 0; i < N; i++) {
        host_a[i] = (float)i;
        host_b[i] = (float)(i * 2);
    }

    // Create streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
        cudaMalloc(&a[s], N * sizeof(float));
        cudaMalloc(&b[s], N * sizeof(float));
        cudaMalloc(&c[s], N * sizeof(float));
    }

    // Run kernels on different streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaMemcpyAsync(a[s], host_a, N * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(b[s], host_b, N * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        add_kernel<<<(N + 255) / 256, 256, 0, streams[s]>>>(a[s], b[s], c[s], N);
        cudaMemcpyAsync(host_c, c[s], N * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
    }

    // Sync all streams
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamSynchronize(streams[s]);
    }

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = host_a[i] + host_b[i];
        if (fabsf(host_c[i] - expected) > max_err) max_err = fabsf(host_c[i] - expected);
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-5 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    // Cleanup
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaFree(a[s]); cudaFree(b[s]); cudaFree(c[s]);
        cudaStreamDestroy(streams[s]);
    }
    return 0;
}
