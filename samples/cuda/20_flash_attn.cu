// 20_flash_attn.cu — Simplified flash attention block (preview)
// Difficulty: 0.95 (hardest) — complex kernel with tiling, shared memory, online softmax
//
// Note: This is a SIMPLIFIED single-block version for demo purposes.
// Real flash-attention (Dao et al.) is far more complex (multi-block, warp-specialized).
// The agent should mark this as "preview" mode if it cannot fully translate.

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define SEQ 64
#define DIM 32
#define BLOCK 32

__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* O, int seq, int dim
) {
    __shared__ float s_Q[BLOCK][BLOCK];
    __shared__ float s_K[BLOCK][BLOCK];
    __shared__ float s_V[BLOCK][BLOCK];

    int tx = threadIdx.x;
    int q_row = blockIdx.x * BLOCK + tx;

    if (q_row >= seq) return;

    // Load Q row into shared memory
    for (int d = 0; d < dim; d++) {
        if (tx < dim) {
            s_Q[tx][d] = (d < dim) ? Q[q_row * dim + d] : 0.0f;
        }
    }
    __syncthreads();

    float max_score = -INFINITY;
    float scores[BLOCK];

    // Compute attention scores
    for (int k_row = 0; k_row < seq; k_row++) {
        // Load K row
        for (int d = 0; d < dim; d++) {
            s_K[tx][d] = (tx < dim) ? K[k_row * dim + d] : 0.0f;
        }
        __syncthreads();

        // Dot product
        float score = 0.0f;
        for (int d = 0; d < dim; d++) {
            score += s_Q[tx][d] * s_K[tx][d];
        }
        score /= sqrtf((float)dim);
        scores[k_row % BLOCK] = score;
        if (score > max_score) max_score = score;
        __syncthreads();
    }

    // Online softmax
    float sum_exp = 0.0f;
    for (int k_row = 0; k_row < seq; k_row++) {
        scores[k_row % BLOCK] = expf(scores[k_row % BLOCK] - max_score);
        sum_exp += scores[k_row % BLOCK];
    }

    // Compute weighted sum of V
    float output[BLOCK] = {0.0f};
    for (int k_row = 0; k_row < seq; k_row++) {
        float weight = scores[k_row % BLOCK] / sum_exp;
        for (int d = 0; d < dim; d++) {
            output[d] += weight * V[k_row * dim + d];
        }
    }

    // Write output
    for (int d = 0; d < dim; d++) {
        O[q_row * dim + d] = output[d];
    }
}

int main() {
    int q_size = SEQ * DIM * sizeof(float);
    float *Q, *K, *V, *O;
    float* host_Q = (float*)malloc(q_size);
    float* host_K = (float*)malloc(q_size);
    float* host_V = (float*)malloc(q_size);
    float* host_O = (float*)malloc(q_size);

    // Initialize with simple values
    for (int i = 0; i < SEQ * DIM; i++) {
        host_Q[i] = 0.1f;
        host_K[i] = 0.1f;
        host_V[i] = 1.0f;
    }

    cudaMalloc(&Q, q_size);
    cudaMalloc(&K, q_size);
    cudaMalloc(&V, q_size);
    cudaMalloc(&O, q_size);

    cudaMemcpy(Q, host_Q, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K, host_K, q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V, host_V, q_size, cudaMemcpyHostToDevice);

    flash_attention_kernel<<<(SEQ + BLOCK - 1) / BLOCK, BLOCK>>>(Q, K, V, O, SEQ, DIM);
    cudaDeviceSynchronize();

    cudaMemcpy(host_O, O, q_size, cudaMemcpyDeviceToHost);

    // CPU reference
    float* ref_O = (float*)malloc(q_size);
    for (int q = 0; q < SEQ; q++) {
        // Compute scores
        float scores[SEQ];
        float max_s = -INFINITY;
        for (int k = 0; k < SEQ; k++) {
            float s = 0.0f;
            for (int d = 0; d < DIM; d++) {
                s += host_Q[q * DIM + d] * host_K[k * DIM + d];
            }
            s /= sqrtf((float)DIM);
            scores[k] = s;
            if (s > max_s) max_s = s;
        }
        // Softmax
        float sum = 0.0f;
        for (int k = 0; k < SEQ; k++) {
            scores[k] = expf(scores[k] - max_s);
            sum += scores[k];
        }
        // Output
        for (int d = 0; d < DIM; d++) {
            ref_O[q * DIM + d] = 0.0f;
            for (int k = 0; k < SEQ; k++) {
                ref_O[q * DIM + d] += (scores[k] / sum) * host_V[k * DIM + d];
            }
        }
    }

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < SEQ * DIM; i++) {
        float err = fabsf(host_O[i] - ref_O[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(O);
    free(host_Q); free(host_K); free(host_V); free(host_O); free(ref_O);
    return 0;
}
