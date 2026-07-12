"""
GPU metrics collector — parses rocm-smi output to surface AMD GPU info
in the UI, proving translations run on real AMD hardware.

Runs `rocm-smi --showproductname --showmeminfo vram --showgpuutil` inside
the sandbox container and parses the output.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class GPUMetrics:
    """GPU metrics captured during a translation run."""
    gpu_name: str = ""
    gpu_arch: str = ""  # e.g., "gfx942" for MI300X
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    gpu_utilization_pct: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    captured_at: str = ""
    raw_output: str = ""  # for debugging

    def to_dict(self) -> dict:
        return {
            "gpu_name": self.gpu_name,
            "gpu_arch": self.gpu_arch,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "temperature_c": self.temperature_c,
            "power_draw_w": self.power_draw_w,
            "captured_at": self.captured_at,
        }


async def collect_gpu_metrics(container: str = "crossfire-sandbox") -> Optional[GPUMetrics]:
    """Collect GPU metrics — tries direct rocm-smi first, falls back to Docker.

    Returns None if neither is available.
    """
    # Try direct mode first (we're inside the ROCm container on the droplet)
    metrics = await _collect_direct()
    if metrics is not None:
        return metrics

    # Fall back to Docker mode (dev environment)
    metrics = await _collect_docker(container)
    return metrics


async def _collect_direct() -> Optional[GPUMetrics]:
    """Collect GPU metrics by calling rocm-smi directly (inside ROCm container)."""
    metrics = GPUMetrics()
    try:
        proc = await asyncio.create_subprocess_exec(
            "rocm-smi", "--showproductname", "--showmeminfo", "vram",
            "--showgpuutil", "--showtemp", "--showpower",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        output = stdout.decode(errors="replace")
        metrics.raw_output = output[:2000]

        if proc.returncode != 0:
            return None

        _parse_rocm_smi(output, metrics)
        from datetime import datetime
        metrics.captured_at = datetime.utcnow().isoformat()

        logger.info(
            f"GPU metrics (direct): {metrics.gpu_name} ({metrics.gpu_arch}), "
            f"VRAM {metrics.vram_used_mb}/{metrics.vram_total_mb} MB, "
            f"util={metrics.gpu_utilization_pct}%"
        )
        return metrics

    except FileNotFoundError:
        return None
    except Exception as e:
        logger.debug(f"Direct GPU metrics failed: {e}")
        return None


async def _collect_docker(container: str) -> Optional[GPUMetrics]:
    """Collect GPU metrics from the ROCm sandbox container via docker exec."""
    metrics = GPUMetrics()

    try:
        cmd = [
            "docker", "exec", container,
            "rocm-smi", "--showproductname", "--showmeminfo", "vram",
            "--showgpuutil", "--showtemp", "--showpower",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        output = stdout.decode(errors="replace")
        metrics.raw_output = output[:2000]

        if proc.returncode != 0:
            logger.warning(f"rocm-smi failed: {stderr.decode()[:200]}")
            return None

        _parse_rocm_smi(output, metrics)
        from datetime import datetime
        metrics.captured_at = datetime.utcnow().isoformat()

        logger.info(
            f"GPU metrics (docker): {metrics.gpu_name} ({metrics.gpu_arch}), "
            f"VRAM {metrics.vram_used_mb}/{metrics.vram_total_mb} MB, "
            f"util={metrics.gpu_utilization_pct}%"
        )
        return metrics

    except asyncio.TimeoutError:
        logger.warning("rocm-smi timed out")
        return None
    except FileNotFoundError:
        logger.debug("Docker not available for GPU metrics")
        return None
    except Exception as e:
        logger.warning(f"GPU metrics collection failed: {e}")
        return None


def _parse_rocm_smi(output: str, metrics: GPUMetrics) -> None:
    """Parse rocm-smi output into the GPUMetrics struct."""

    # GPU name: e.g., "GPU[0]           : AMD Instinct MI300X"
    name_match = re.search(r"GPU\[\d+\]\s*:\s*(.+?)$", output, re.MULTILINE)
    if name_match:
        metrics.gpu_name = name_match.group(1).strip()

    # Architecture / series: e.g., "GPU[0]              : gfx942"
    arch_match = re.search(r"GPU\[\d+\]\s*:\s*gfx(\d+[a-z0-9]*)", output, re.MULTILINE)
    if arch_match:
        metrics.gpu_arch = f"gfx{arch_match.group(1)}"
    else:
        # Try alternative format
        arch_match = re.search(r"gfx(\d+[a-z0-9]*)", output)
        if arch_match:
            metrics.gpu_arch = f"gfx{arch_match.group(1)}"

    # VRAM: e.g., "GPU[0]               : VRAM Total Memory (B): 201388810240"
    vram_total_match = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", output)
    if vram_total_match:
        metrics.vram_total_mb = int(vram_total_match.group(1)) // (1024 * 1024)

    vram_used_match = re.search(r"VRAM Total Used Memory \(B\):\s*(\d+)", output)
    if vram_used_match:
        metrics.vram_used_mb = int(vram_used_match.group(1)) // (1024 * 1024)

    metrics.vram_free_mb = metrics.vram_total_mb - metrics.vram_used_mb

    # GPU utilization: e.g., "GPU[0]            : GPU-UTIL (%)"
    util_match = re.search(r"GPU-UTIL \(%\):\s*(\d+(?:\.\d+)?)", output)
    if util_match:
        metrics.gpu_utilization_pct = float(util_match.group(1))

    # Temperature: e.g., "Temperature (Sensor edge) (C): 42"
    temp_match = re.search(r"Temperature \(Sensor edge\) \(C\):\s*(\d+(?:\.\d+)?)", output)
    if temp_match:
        metrics.temperature_c = float(temp_match.group(1))

    # Power: e.g., "Average Graphics Package Power (W): 75.0"
    power_match = re.search(r"Power \(W\):\s*(\d+(?:\.\d+)?)", output)
    if not power_match:
        power_match = re.search(r"Average Graphics Package Power \(W\):\s*(\d+(?:\.\d+)?)", output)
    if power_match:
        metrics.power_draw_w = float(power_match.group(1))


def is_amd_gpu_available(container: str = "crossfire-sandbox") -> bool:
    """Quick synchronous check if AMD GPU is available in the sandbox.

    Used by the /health endpoint for fast status reporting.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "exec", container, "rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=3,
        )
        return result.returncode == 0 and ("MI300" in result.stdout or "MI250" in result.stdout)
    except Exception:
        return False
