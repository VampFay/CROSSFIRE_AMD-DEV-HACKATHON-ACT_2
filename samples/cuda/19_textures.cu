// 19_textures.cu — CUDA texture memory (preview mode)
// Difficulty: 0.65 (medium-hard) — textures → texture objects (HIP support varies)

#include <cuda_runtime.h>
#include <stdio.h>

#define WIDTH 16
#define HEIGHT 16

__global__ void texture_kernel(cudaTextureObject_t tex, float* output, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        // Sample texture at integer coordinates
        float val = tex2D<float>(tex, (float)x, (float)y);
        output[y * w + x] = val * 2.0f;
    }
}

int main() {
    // Allocate host data
    size_t pitch = WIDTH * sizeof(float);
    float* host_input = (float*)malloc(HEIGHT * WIDTH * sizeof(float));
    float* host_output = (float*)malloc(HEIGHT * WIDTH * sizeof(float));
    for (int i = 0; i < WIDTH * HEIGHT; i++) host_input[i] = (float)i;

    // Allocate device memory with pitch
    float* d_input;
    size_t d_pitch;
    cudaMallocPitch(&d_input, &d_pitch, WIDTH * sizeof(float), HEIGHT);
    cudaMemcpy2D(d_input, d_pitch, host_input, pitch,
                 WIDTH * sizeof(float), HEIGHT, cudaMemcpyHostToDevice);

    // Create texture object
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = d_input;
    res_desc.res.pitch2D.width = WIDTH;
    res_desc.res.pitch2D.height = HEIGHT;
    res_desc.res.pitch2D.pitchInBytes = d_pitch;
    res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &res_desc, &tex_desc, NULL);

    // Allocate output
    float* d_output;
    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(float));

    // Launch kernel
    dim3 blocks((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    dim3 threads(16, 16);
    texture_kernel<<<blocks, threads>>>(tex, d_output, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(host_output, d_output, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify
    float max_err = 0.0f;
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        float expected = (float)i * 2.0f;
        if (fabsf(host_output[i] - expected) > max_err) max_err = fabsf(host_output[i] - expected);
    }
    printf("Max error: %e\n", max_err);

    printf("===OUTPUT_BEGIN===\n");
    printf("{\"max_error\": %e, \"status\": \"%s\"}\n", max_err,
           max_err < 1e-5 ? "pass" : "fail");
    printf("===OUTPUT_END===\n");

    cudaDestroyTextureObject(tex);
    cudaFree(d_input); cudaFree(d_output);
    free(host_input); free(host_output);
    return 0;
}
