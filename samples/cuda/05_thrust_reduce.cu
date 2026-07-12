// 05_thrust_reduce.cu — Thrust-based reduction
// Difficulty: 0.65 (medium-hard) — Thrust → rocPRIM

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <stdio.h>

#define N 1024

int main() {
    thrust::device_vector<float> d_vec(N);

    // Fill with 0..N-1
    thrust::sequence(d_vec.begin(), d_vec.end(), 0.0f, 1.0f);

    // Compute sum
    float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    // Expected: 0 + 1 + ... + 1023 = 1023*1024/2 = 523776
    float expected = (float)(N * (N - 1)) / 2.0f;
    float err = fabsf(sum - expected);

    printf("Sum: %f, Expected: %f, Error: %e\n", sum, expected, err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"sum\": %f, \"expected\": %f, \"max_error\": %e, \"status\": \"%s\"}\n",
           sum, expected, err, err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    return 0;
}
