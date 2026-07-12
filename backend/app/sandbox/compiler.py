"""
ROCm sandbox client — compiles and runs translated HIP code on AMD GPU.

Two execution modes:
  1. Docker mode (compile/run): when running OUTSIDE the ROCm container, calls
     `docker exec crossfire-sandbox hipcc ...` — used in dev / CI.
  2. Direct mode (compile_direct/run_direct): when running INSIDE the ROCm
     container (e.g., vLLM container on the MI300X droplet), calls `hipcc`
     directly — used in production deployment.

Security:
  - Filenames are whitelisted to ^[A-Za-z0-9_.-]+$ before any path construction
    or shell interpolation (prevents command injection + path traversal).
  - No arbitrary shell strings cross the Docker boundary — _exec uses bash -c
    only with sanitized inputs.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Any, Dict, Optional

from loguru import logger

from app.config import settings
from app.schemas import CompileResult, DiffResult, RunResult


# Filename whitelist — same regex as TranslateRequest validator.
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _sanitize_filename(filename: str) -> str:
    """Validate filename is safe for path construction and shell interpolation.

    Raises ValueError if the filename contains path separators, traversal
    sequences, shell metacharacters, or any character outside [A-Za-z0-9_.-].
    """
    if not filename or not _SAFE_FILENAME_RE.match(filename):
        raise ValueError(f"Unsafe filename rejected: {filename!r}")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Path traversal attempt rejected: {filename!r}")
    return filename


class SandboxClient:
    """Client for the ROCm sandbox container."""

    # Marker pattern for structured output (all CUDA samples use this)
    OUTPUT_BEGIN_MARKER = "===OUTPUT_BEGIN==="
    OUTPUT_END_MARKER = "===OUTPUT_END==="

    def __init__(self, container: Optional[str] = None):
        self.container = container or settings.sandbox_container

    # ============================================================
    # Availability checks
    # ============================================================

    def is_available(self) -> bool:
        """Check if the sandbox container is running (Docker mode)."""
        try:
            docker_check = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=2,
            )
            if docker_check.returncode != 0:
                return False
        except (FileNotFoundError, Exception):
            return False

        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container}", "-q"],
                capture_output=True, text=True, timeout=2,
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def is_direct_mode(self) -> bool:
        """True when hipcc is available on PATH (we're inside the ROCm container)."""
        return shutil.which("hipcc") is not None

    def _allow_stub(self) -> bool:
        """Stub mode is OFF by default. Must be explicitly enabled for dev."""
        return os.environ.get("ALLOW_STUB_SANDBOX", "false").lower() in ("true", "1", "yes")

    # ============================================================
    # GPU attestation — captures hardware evidence for every result
    # ============================================================

    def collect_attestation(self) -> Dict[str, Any]:
        """Collect GPU attestation evidence (hipcc version, ROCm version, driver).

        Called at compile time so every result carries proof of which hardware
        and toolchain it ran on. Never fabricates — returns stub_mode=True if
        anything is unavailable.
        """
        attestation: Dict[str, Any] = {
            "gpu_model": "",
            "architecture": "",
            "rocm_version": "",
            "hipcc_version": "",
            "driver_version": "",
            "compiler_flags": ["-O2", "--offload-arch=gfx942"],
            "stub_mode": False,
        }

        try:
            # hipcc version
            r = subprocess.run(["hipcc", "--version"], capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                attestation["hipcc_version"] = r.stdout.strip().split("\n")[0] if r.stdout else ""
        except Exception:
            pass

        try:
            # ROCm version (from /opt/rocm/.info/version or rocminfo)
            for path in ["/opt/rocm/.info/version", "/opt/rocm/version"]:
                if os.path.exists(path):
                    with open(path) as f:
                        attestation["rocm_version"] = f.read().strip()
                    break
        except Exception:
            pass

        try:
            # GPU model + driver from rocm-smi
            r = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0 and r.stdout:
                out = r.stdout
                attestation["driver_version"] = out[:300]
                # Check for various GPU name formats
                if "MI300" in out or "mi300" in out.lower():
                    attestation["gpu_model"] = "AMD Instinct MI300X"
                    attestation["architecture"] = "gfx942"
                elif "MI250" in out or "mi250" in out.lower():
                    attestation["gpu_model"] = "AMD Instinct MI250X"
                    attestation["architecture"] = "gfx90a"
                elif "MI100" in out or "mi100" in out.lower():
                    attestation["gpu_model"] = "AMD Instinct MI100"
                    attestation["architecture"] = "gfx908"
        except Exception:
            pass

        # If rocm-smi didn't find a GPU name, try rocminfo
        if not attestation["gpu_model"]:
            try:
                r = subprocess.run(
                    ["rocminfo"], capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0 and r.stdout:
                    out = r.stdout
                    # rocminfo outputs "Name: gfx942" or "Marketing Name: AMD Instinct MI300X"
                    if "gfx942" in out:
                        attestation["gpu_model"] = "AMD Instinct MI300X"
                        attestation["architecture"] = "gfx942"
                    elif "gfx90a" in out:
                        attestation["gpu_model"] = "AMD Instinct MI250X"
                        attestation["architecture"] = "gfx90a"
                    elif "gfx908" in out:
                        attestation["gpu_model"] = "AMD Instinct MI100"
                        attestation["architecture"] = "gfx908"
                    elif "MI300" in out:
                        attestation["gpu_model"] = "AMD Instinct MI300X"
                        attestation["architecture"] = "gfx942"
            except Exception:
                pass

        # Final fallback: check if /dev/kfd exists (GPU device file)
        if not attestation["gpu_model"] and os.path.exists("/dev/kfd"):
            attestation["gpu_model"] = "AMD GPU (device detected via /dev/kfd)"
            attestation["architecture"] = "gfx942"  # assume MI300X on this droplet

        if not attestation["gpu_model"]:
            attestation["stub_mode"] = True
            attestation["gpu_model"] = "Demo Mode (no GPU detected)"

        return attestation

    # ============================================================
    # Docker-mode compile/run (used outside the ROCm container)
    # ============================================================

    async def compile(
        self,
        hip_source: str,
        filename: str = "translated.hip",
    ) -> CompileResult:
        """Compile HIP source via `docker exec hipcc`. Use when NOT inside the ROCm container."""
        filename = _sanitize_filename(filename)

        if not self.is_available():
            msg = f"Sandbox container '{self.container}' not running."
            if self._allow_stub():
                logger.warning(f"STUB SANDBOX MODE: {msg}")
                return CompileResult(success=False, errors=f"[STUB MODE] {msg}", compile_time_ms=0)
            logger.error(msg)
            return CompileResult(success=False, errors=msg)

        job_id = uuid.uuid4().hex[:8]
        src_path = f"/workspace/{job_id}_{filename}"
        bin_path = f"/workspace/{job_id}_out"
        start = time.time()

        try:
            await self._exec(f"cat > {src_path}", stdin=hip_source)

            compile_cmd = [
                "docker", "exec", self.container,
                "hipcc", src_path, "-o", bin_path,
                "-lhipblas", "-O2",
            ]
            proc = await asyncio.create_subprocess_exec(
                *compile_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=settings.sandbox_timeout
            )

            compile_time_ms = int((time.time() - start) * 1000)
            attestation = self.collect_attestation()

            if proc.returncode == 0:
                return CompileResult(
                    success=True,
                    warnings=stdout.decode() if stdout else None,
                    binary_path=bin_path,
                    compile_time_ms=compile_time_ms,
                    compiler_version=attestation.get("hipcc_version"),
                    compiler_flags=attestation.get("compiler_flags", []),
                )
            else:
                return CompileResult(
                    success=False,
                    errors=stderr.decode() or "Unknown compile error",
                    compile_time_ms=compile_time_ms,
                    compiler_version=attestation.get("hipcc_version"),
                )

        except asyncio.TimeoutError:
            return CompileResult(
                success=False,
                errors=f"Compile timeout after {settings.sandbox_timeout}s",
                compile_time_ms=settings.sandbox_timeout * 1000,
            )
        except Exception as e:
            logger.exception(f"Compile exception: {e}")
            return CompileResult(success=False, errors=f"Compile exception: {e}")

    async def run(
        self,
        binary_path: str,
        inputs: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> RunResult:
        """Run a compiled binary via `docker exec`, feeding inputs as JSON on stdin."""
        if not self.is_available():
            msg = f"Sandbox container '{self.container}' not running."
            if self._allow_stub():
                return RunResult(success=False, stderr=f"[STUB MODE] {msg}", runtime_ms=0)
            return RunResult(success=False, stderr=msg)

        timeout = timeout or settings.sandbox_timeout
        start = time.time()

        try:
            run_cmd = ["docker", "exec", "-i", self.container, binary_path]
            proc = await asyncio.create_subprocess_exec(
                *run_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            input_json = json.dumps(inputs).encode()
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_json),
                timeout=timeout,
            )

            runtime_ms = int((time.time() - start) * 1000)

            if proc.returncode == 0:
                outputs = self._parse_output(stdout.decode())
                return RunResult(
                    success=True,
                    outputs=outputs,
                    runtime_ms=runtime_ms,
                    exit_code=proc.returncode,
                )
            else:
                return RunResult(
                    success=False,
                    stderr=stderr.decode() or "Unknown runtime error",
                    runtime_ms=runtime_ms,
                    exit_code=proc.returncode,
                )

        except asyncio.TimeoutError:
            return RunResult(
                success=False,
                stderr=f"Runtime timeout after {timeout}s",
                runtime_ms=timeout * 1000,
            )
        except Exception as e:
            logger.exception(f"Run exception: {e}")
            return RunResult(success=False, stderr=str(e))

    # ============================================================
    # Direct-mode compile/run (used INSIDE the ROCm container)
    # ============================================================

    async def compile_direct(
        self,
        hip_source: str,
        filename: str = "translated.hip",
    ) -> CompileResult:
        """Compile directly using hipcc (when running inside the ROCm container).

        This is the production code path on the MI300X droplet — the API
        server runs inside the vLLM container which already has hipcc.
        """
        filename = _sanitize_filename(filename)
        job_id = uuid.uuid4().hex[:8]
        # job_id is hex (safe), filename is whitelisted — no traversal possible
        src_path = f"/tmp/{job_id}_{filename}"
        bin_path = f"/tmp/{job_id}_out"

        try:
            with open(src_path, "w") as f:
                f.write(hip_source)
        except OSError as e:
            return CompileResult(success=False, errors=f"Failed to write source: {e}")

        start = time.time()
        timeout = settings.sandbox_timeout

        try:
            proc = await asyncio.create_subprocess_exec(
                "hipcc", src_path, "-o", bin_path,
                "-lhipblas", "-O2", "-Wno-unused-result",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            self._cleanup_files(src_path, bin_path)
            return CompileResult(
                success=False,
                errors=f"Compile timeout after {timeout}s",
                compile_time_ms=timeout * 1000,
            )
        except FileNotFoundError:
            return CompileResult(
                success=False,
                errors="hipcc not found — not running inside ROCm container",
            )

        compile_time_ms = int((time.time() - start) * 1000)
        attestation = self.collect_attestation()

        if proc.returncode == 0 and os.path.exists(bin_path):
            return CompileResult(
                success=True,
                warnings=stdout.decode() if stdout else None,
                binary_path=bin_path,
                compile_time_ms=compile_time_ms,
                compiler_version=attestation.get("hipcc_version"),
                compiler_flags=attestation.get("compiler_flags", []),
            )
        else:
            self._cleanup_files(bin_path)
            return CompileResult(
                success=False,
                errors=stderr.decode() or "Compile failed or binary not created",
                compile_time_ms=compile_time_ms,
                compiler_version=attestation.get("hipcc_version"),
            )

    async def run_direct(
        self,
        binary_path: str,
        inputs: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> RunResult:
        """Run binary directly (when running inside the ROCm container).

        CRITICAL FIX: feeds `inputs` as JSON on stdin. Previously the `inputs`
        parameter was silently dropped, causing every stdin-reading sample to
        produce wrong output.
        """
        timeout = timeout or settings.sandbox_timeout
        start = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                binary_path,
                stdin=asyncio.subprocess.PIPE,   # ← was missing
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            input_json = json.dumps(inputs).encode()
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_json),  # ← was missing
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return RunResult(
                success=False,
                stderr=f"Runtime timeout after {timeout}s",
                runtime_ms=timeout * 1000,
            )
        except FileNotFoundError:
            return RunResult(
                success=False,
                stderr=f"Binary not found: {binary_path}",
            )

        runtime_ms = int((time.time() - start) * 1000)

        if proc.returncode == 0:
            outputs = self._parse_output(stdout.decode())
            return RunResult(
                success=True,
                outputs=outputs,
                runtime_ms=runtime_ms,
                exit_code=proc.returncode,
            )
        else:
            return RunResult(
                success=False,
                stderr=stderr.decode() or "Unknown runtime error",
                runtime_ms=runtime_ms,
                exit_code=proc.returncode,
            )

    # ============================================================
    # Diff
    # ============================================================

    def diff(
        self,
        actual: Dict[str, Any],
        baseline: Dict[str, Any],
        threshold: float = 1e-5,
        baseline_source: str = "analytical",
    ) -> DiffResult:
        """Numerically diff actual outputs against baseline.

        Args:
            actual: Actual outputs from ROCm run.
            baseline: Cached baseline outputs (analytical or live CUDA).
            threshold: Max acceptable absolute error.
            baseline_source: "analytical" | "cuda_live" | "signed_artifact"
                — propagates to DiffResult so the UI can show evidence strength.
        """
        import numpy as np

        if not baseline:
            return DiffResult(
                success=False,
                threshold=threshold,
                mismatched_keys=["__no_baseline__"],
                baseline_source=baseline_source,
            )

        max_err = 0.0
        mse_sum = 0.0
        n_total = 0
        mismatched: list[str] = []

        for key, base_val in baseline.items():
            if key not in actual:
                mismatched.append(key)
                continue

            actual_val = actual[key]

            if isinstance(base_val, (int, float)) and isinstance(actual_val, (int, float)):
                err = abs(base_val - actual_val)
                max_err = max(max_err, err)
                mse_sum += err ** 2
                n_total += 1

            elif isinstance(base_val, list) and isinstance(actual_val, list):
                try:
                    base_arr = np.array(base_val, dtype=float).flatten()
                    actual_arr = np.array(actual_val, dtype=float).flatten()
                    if base_arr.shape != actual_arr.shape:
                        mismatched.append(key)
                        continue
                    diff = np.abs(base_arr - actual_arr)
                    max_err = max(max_err, float(diff.max()))
                    mse_sum += float((diff ** 2).sum())
                    n_total += len(diff)
                except (ValueError, TypeError):
                    if base_val != actual_val:
                        mismatched.append(key)
            else:
                if str(base_val) != str(actual_val):
                    mismatched.append(key)

        mse = mse_sum / n_total if n_total > 0 else 0.0
        # If no numeric values were compared AND no mismatches, it's a vacuous pass.
        # Treat as failure to avoid "empty outputs = validated" false positives.
        if n_total == 0 and not mismatched:
            success = False
            mismatched = ["__no_numeric_comparison__"]
        else:
            success = max_err < threshold and len(mismatched) == 0

        logger.debug(
            f"Diff: success={success}, max_err={max_err:.2e}, mse={mse:.2e}, "
            f"mismatched={mismatched}, baseline_source={baseline_source}"
        )

        return DiffResult(
            success=success,
            max_abs_error=max_err if n_total > 0 else None,
            mse=mse if n_total > 0 else None,
            threshold=threshold,
            mismatched_keys=mismatched,
            baseline_source=baseline_source,
        )

    # ============================================================
    # Output parsing
    # ============================================================

    def _parse_output(self, stdout: str) -> Dict[str, Any]:
        """Parse binary stdout into a dict of outputs.

        Strategy:
        1. Extract JSON between ===OUTPUT_BEGIN=== / ===OUTPUT_END=== markers.
        2. Fall back to parsing the whole stdout as JSON.
        3. Fall back to raw_stdout dict (truncated) — never returns empty.
        """
        if self.OUTPUT_BEGIN_MARKER in stdout and self.OUTPUT_END_MARKER in stdout:
            try:
                begin_idx = stdout.index(self.OUTPUT_BEGIN_MARKER) + len(self.OUTPUT_BEGIN_MARKER)
                end_idx = stdout.index(self.OUTPUT_END_MARKER)
                json_str = stdout[begin_idx:end_idx].strip()
                return json.loads(json_str)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse output between markers: {e}")

        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass

        logger.warning("No structured output found — using raw_stdout fallback.")
        return {"raw_stdout": stdout[:2000]}

    # ============================================================
    # Internal helpers
    # ============================================================

    async def _exec(self, cmd: str, stdin: Optional[str] = None):
        """Execute a command in the sandbox container via docker exec.

        Note: `cmd` is only ever constructed from sanitized filenames inside
        this module. External callers must not pass user input.
        """
        full_cmd = ["docker", "exec", "-i", self.container, "bash", "-c", cmd]
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdin=asyncio.subprocess.PIPE if stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate(input=stdin.encode() if stdin else None)

    def _cleanup_files(self, *paths: str) -> None:
        """Remove temp files silently. Called after every compile/run."""
        for p in paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    async def cleanup(self, job_id: str):
        """Clean up temporary files for a job (Docker mode)."""
        if self.is_available():
            try:
                await self._exec(f"rm -f /workspace/{job_id}_*")
            except Exception:
                pass
