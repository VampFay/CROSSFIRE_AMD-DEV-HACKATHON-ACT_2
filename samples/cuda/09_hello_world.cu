// 09_hello_world.cu — Minimal CUDA hello world
// Difficulty: 0.05 (easiest)

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_kernel() {
    // Thread 0 of block 0 prints
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello from GPU! Thread %d of block %d\n", threadIdx.x, blockIdx.x);
    }
}

int main() {
    printf("Hello from CPU!\n");

    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"status\": \"pass\"}\n");
    printf("===OUTPUT_END===\n");

    return 0;
}
