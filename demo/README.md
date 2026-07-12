# Demo

## Video

`crossfire_demo.mp4` — 22 seconds, 1920x1080

Shows a CUDA file translated to ROCm and validated on AMD MI300X.

## What you see

- CUDA code on the left, translated HIP code on the right
- Compiled with hipcc on AMD MI300X
- Output compared against expected values (zero error)
- "Test Verified" badge means it passed

## Stack

- Model: Gemma 4 12B via vLLM
- GPU: AMD MI300X
- ROCm: 7.2.3
- Translation: hipify-clang + Gemma 4 12B

## Live demo

http://129.212.185.42:8000/ui/
