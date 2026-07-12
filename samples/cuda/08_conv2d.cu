// 08_conv2d.cu — cuDNN convolution
// Difficulty: 0.85 (hard) — cuDNN → MIOpen semantic mapping

#include <cuda_runtime.h>
#include <cudnn_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define BATCH 1
#define CHANNELS 1
#define HEIGHT 8
#define WIDTH 8
#define KERN_SIZE 3
#define OUT_CHANNELS 1

#define CHECK_CUDNN(x) do { \
    cudnnStatus_t s = (x); \
    if (s != CUDNN_STATUS_SUCCESS) { \
        printf("cuDNN error: %s at line %d\n", cudnnGetErrorString(s), __LINE__); \
        exit(1); \
    } \
} while(0)

int main() {
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    // Input tensor descriptor
    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            BATCH, CHANNELS, HEIGHT, WIDTH));

    // Filter (kernel) descriptor
    cudnnFilterDescriptor_t filter_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                            CUDNN_TENSOR_NCHW,
                                            OUT_CHANNELS, CHANNELS, KERN_SIZE, KERN_SIZE));

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 1, 1,  // padding
                                                 1, 1,  // stride
                                                 1, 1,  // dilation
                                                 CUDNN_CROSS_CORRELATION,
                                                 CUDNN_DATA_FLOAT));

    // Output tensor descriptor
    int out_n, out_c, out_h, out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc,
                                                       filter_desc,
                                                       &out_n, &out_c, &out_h, &out_w));

    cudnnTensorDescriptor_t output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_FLOAT,
                                            out_n, out_c, out_h, out_w));

    // Allocate memory
    size_t input_size = BATCH * CHANNELS * HEIGHT * WIDTH * sizeof(float);
    size_t filter_size = OUT_CHANNELS * CHANNELS * KERN_SIZE * KERN_SIZE * sizeof(float);
    size_t output_size = out_n * out_c * out_h * out_w * sizeof(float);

    float *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_output, output_size);

    // Init input and filter
    float* host_input = (float*)malloc(input_size);
    float* host_filter = (float*)malloc(filter_size);
    for (int i = 0; i < BATCH * CHANNELS * HEIGHT * WIDTH; i++) host_input[i] = 1.0f;
    for (int i = 0; i < OUT_CHANNELS * CHANNELS * KERN_SIZE * KERN_SIZE; i++) host_filter[i] = 1.0f;

    cudaMemcpy(d_input, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, host_filter, filter_size, cudaMemcpyHostToDevice);

    // Get algorithm
    cudnnConvolutionFwdAlgoPerf_t perf;
    int returned_count;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle,
        input_desc, filter_desc, conv_desc, output_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 1, &returned_count, &perf));

    // Allocate workspace
    void* d_workspace;
    cudaMalloc(&d_workspace, perf.memory);

    // Run convolution
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(handle,
        &alpha, input_desc, d_input,
        filter_desc, d_filter,
        conv_desc, perf.algo,
        d_workspace, perf.memory,
        &beta, output_desc, d_output));

    cudaDeviceSynchronize();

    // Get output sum (single value to verify)
    float* host_output = (float*)malloc(output_size);
    cudaMemcpy(host_output, d_output, output_size, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < out_n * out_c * out_h * out_w; i++) sum += host_output[i];
    printf("Output sum: %f (expected: %f)\n", sum, 9.0f * out_n * out_c * out_h * out_w);

    float err = fabsf(sum - 9.0f * out_n * out_c * out_h * out_w);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"output_sum\": %f, \"expected\": %f, \"max_error\": %e, \"status\": \"%s\"}\n",
           sum, 9.0f * out_n * out_c * out_h * out_w, err,
           err < 1e-3 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    // Cleanup
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_output); cudaFree(d_workspace);
    free(host_input); free(host_filter); free(host_output);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    return 0;
}
