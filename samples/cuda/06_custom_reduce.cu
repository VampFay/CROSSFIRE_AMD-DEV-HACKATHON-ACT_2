// 06_custom_reduce.cu — Block reduce using warp shuffle primitives
// Difficulty: 0.90 (hardest) — warp shuffle + shared memory

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level reduction using shuffle
__device__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Block-level reduction
__global__ void block_reduce(const float* input, float* output, int n) {
    __shared__ float shared[WARP_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * BLOCK_SIZE + tid;

    // Load and bound check
    float val = (i < n) ? input[i] : 0.0f;

    // Warp reduce
    val = warp_reduce(val);

    // Write warp results to shared memory
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;

    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    // Final reduce: first warp reduces all warp results
    val = (tid < BLOCK_SIZE / WARP_SIZE) ? shared[tid] : 0.0f;
    if (warp == 0) {
        val = warp_reduce(val);
    }

    // Write block result
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

int main() {
    const int n = BLOCK_SIZE * 4;  // 4 blocks
    float *input, *output;
    float host_input[n], host_output[4];

    // Initialize
    for (int i = 0; i < n; i++) {
        host_input[i] = 1.0f;  // sum per block = 256
    }

    cudaMalloc(&input, n * sizeof(float));
    cudaMalloc(&output, 4 * sizeof(float));

    cudaMemcpy(input, host_input, n * sizeof(float), cudaMemcpyHostToDevice);

    block_reduce<<<4, BLOCK_SIZE>>>(input, output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(host_output, output, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: each block should sum to 256
    float max_err = 0.0f;
    for (int i = 0; i < 4; i++) {
        float err = fabsf(host_output[i] - 256.0f);
        if (err > max_err) max_err = err;
    }
    printf("Block sums: %.1f %.1f %.1f %.1f\n",
           host_output[0], host_output[1], host_output[2], host_output[3]);
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"block_sums\": [%f, %f, %f, %f], \"max_error\": %e, \"status\": \"%s\"}\n",
           host_output[0], host_output[1], host_output[2], host_output[3],
           max_err, max_err < 1e-5 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(input);
    cudaFree(output);
    return 0;
}
