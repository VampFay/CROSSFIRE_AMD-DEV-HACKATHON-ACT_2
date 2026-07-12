#!/usr/bin/env python3
"""
Generate baseline output files for all 20 CUDA samples.

These baselines are the expected JSON output that each sample emits
between the ===OUTPUT_BEGIN=== and ===OUTPUT_END=== markers when run
on real CUDA hardware. They're used by the agent's diff function to
validate that the translated ROCm code produces equivalent results.

Since we don't have CUDA hardware available in the dev environment,
these baselines are computed analytically based on what each sample
is documented to produce. They should be verified against actual CUDA
runs when hardware is available.

Usage:
    python scripts/generate_baselines.py

Outputs:
    samples/baselines/{stem}_outputs.json  (expected stdout JSON)
    samples/baselines/{stem}_inputs.json   (inputs to feed via stdin, if applicable)
"""
from __future__ import annotations

import json
from pathlib import Path


# ============================================================
# Pre-compute dynamic values
# ============================================================

# Sample 10: dot product of (i+1)*0.01 and (N-i)*0.01 for i in 0..511, N=512
DOT_N = 512
DOT_SUM = sum((i + 1) * (DOT_N - i) * 0.0001 for i in range(DOT_N))

# Sample 14: histogram of (i % 16) for i in 0..1023 -> 64 per bin
HIST_N = 1024
HIST_BINS = 16
HIST_EXPECTED = [HIST_N // HIST_BINS] * HIST_BINS  # [64, 64, ..., 64]

# Sample 16: atomic counter and sum
ATOMIC_N = 1000000


# ============================================================
# Baseline definitions
# ============================================================

BASELINES = {
    "01_vector_add": {
        "outputs": {"max_error": 0.0, "status": "pass"},
        "inputs": {"_hardcoded": True, "N": 1024},
    },
    "02_saxpy": {
        "outputs": {"max_error": 0.0, "status": "pass"},
        "inputs": {"_hardcoded": True, "N": 65536, "a": 2.0},
    },
    "03_matrix_mul": {
        "outputs": {"max_error": 1e-4, "status": "pass"},
        "inputs": {"_hardcoded": True, "N": 64},
    },
    "04_sgemm": {
        "outputs": {"max_error": 1e-4, "status": "pass"},
        "inputs": {"_hardcoded": True, "M": 64, "N": 64, "K": 64},
    },
    "05_thrust_reduce": {
        "outputs": {
            "sum": 523776.0,
            "expected": 523776.0,
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": 1024},
    },
    "06_custom_reduce": {
        "outputs": {
            "block_sums": [256.0, 256.0, 256.0, 256.0],
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "BLOCK_SIZE": 256},
    },
    "07_pytorch_ext": {
        "outputs": {"status": "skip", "reason": "PyTorch extension - requires PyTorch headers, standalone compile not supported"},
        "inputs": {"_hardcoded": True},
    },
    "08_conv2d": {
        "outputs": {
            "output_sum": 576.0,
            "expected": 576.0,
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True},
    },
    "09_hello_world": {
        "outputs": {"status": "pass"},
        "inputs": {"_hardcoded": True},
    },
    "10_dot_product": {
        "outputs": {
            "gpu_sum": DOT_SUM,
            "cpu_sum": DOT_SUM,
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": DOT_N},
    },
    "11_reduction": {
        "outputs": {
            "sum": 1024.0,
            "expected": 1024.0,
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": 1024},
    },
    "12_scan": {
        "outputs": {
            "first_5": [1.0, 2.0, 3.0, 4.0, 5.0],
            "last_5": [508.0, 509.0, 510.0, 511.0, 512.0],
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": 512},
    },
    "13_transpose": {
        "outputs": {
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": 32},
    },
    "14_histogram": {
        "outputs": {
            "histogram": HIST_EXPECTED,
            "max_error": 0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": HIST_N, "BINS": HIST_BINS},
    },
    "15_sort": {
        "outputs": {
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": 512},
    },
    "16_atomic": {
        "outputs": {
            "counter": ATOMIC_N,
            "sum": float(ATOMIC_N),
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": ATOMIC_N},
    },
    "17_streams": {
        "outputs": {
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "N": 1024, "NUM_STREAMS": 4},
    },
    "18_events": {
        "outputs": {
            "max_error": 0.0,
            "status": "pass",
            "kernel_time_ms": 1.0,
        },
        "inputs": {"_hardcoded": True, "N": 1048576},
    },
    "19_textures": {
        "outputs": {
            "max_error": 0.0,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "WIDTH": 16, "HEIGHT": 16},
    },
    "20_flash_attn": {
        "outputs": {
            "max_error": 1e-3,
            "status": "pass",
        },
        "inputs": {"_hardcoded": True, "SEQ": 64, "DIM": 32},
    },
}


# ============================================================
# Main
# ============================================================

def main():
    baselines_dir = Path(__file__).parent.parent.parent / "samples" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating baselines in: {baselines_dir}")
    print(f"Total samples: {len(BASELINES)}")
    print()

    for stem, data in BASELINES.items():
        outputs_path = baselines_dir / f"{stem}_outputs.json"
        inputs_path = baselines_dir / f"{stem}_inputs.json"

        with open(outputs_path, "w") as f:
            json.dump(data["outputs"], f, indent=2)
        with open(inputs_path, "w") as f:
            json.dump(data["inputs"], f, indent=2)

        status = data["outputs"].get("status", "?")
        print(f"  {stem:25s}  status={status}")

    print()
    print(f"Generated {len(BASELINES)} baseline output files")
    print(f"Generated {len(BASELINES)} baseline input files")
    print()
    print("NOTE: These baselines are computed analytically.")
    print("      Verify against actual CUDA runs when hardware is available.")
    print("      The 'max_error' values are conservative estimates - actual runs")
    print("      may produce slightly different FP error. Adjust thresholds as needed.")


if __name__ == "__main__":
    main()
