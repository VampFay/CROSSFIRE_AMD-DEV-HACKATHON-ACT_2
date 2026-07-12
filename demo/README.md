# Demo Video

## File

`crossfire_demo.mp4` — 40 seconds, 1920x1080

Shows two CUDA files translated to ROCm and validated on AMD MI300X:

1. `01_vector_add.cu` — passes, zero error
2. `02_saxpy.cu` — passes, zero error

## What you're seeing

- CUDA code on the left, translated HIP code on the right
- Each translation is compiled with hipcc and run on the GPU
- Output is compared against expected values (max error = 0.0)
- The "Test Verified" badge means it compiled, ran, and passed validation

## Stack

- Model: Gemma 4 12B via vLLM
- GPU: AMD MI300X (gfx942)
- ROCm: 7.2.3
- Translation: hipify-clang + Gemma 4 12B

## Live demo

http://129.212.185.42:8000/ui/
