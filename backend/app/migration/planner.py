"""
Repository-level CUDA→ROCm migration planner.

Analyzes a repository structure, detects CUDA features, classifies
files by migration strategy, and produces a MigrationPlan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from loguru import logger

from app.schemas import MigrationPlan, MigrationStrategy


# ============================================================
# Capability matrix — which CUDA features CROSSFIRE supports
# ============================================================

SUPPORTED_LIBRARIES = {
    "cuda_runtime": "hip_runtime",
    "cublas": "hipblas",
    "cufft": "hipfft",
    "curand": "hiprand",
    "cub": "rocprim",
    "thrust": "thrust",  # thrust is available on ROCm
}

BLOCKED_LIBRARIES = {
    "cudnn": "miopen (API differences — requires semantic repair)",
    "nccl": "rccl (API compatible but requires rebuild)",
    "nvrtc": "hiprtc (partial support)",
    "cutlass": "no direct ROCm equivalent (blocked)",
}

SUPPORTED_PATTERNS = {
    "kernel_launch",
    "shared_memory",
    "warp_shuffle",
    "streams",
    "events",
    "atomics",
    "cuda_runtime_api",
}

BLOCKED_PATTERNS = {
    "inline_ptx": "PTX assembly is NVIDIA-specific — no ROCm equivalent",
    "cuda_graphics_interop": "Graphics interop requires OpenGL/Vulkan rewrite",
    "cuobjdump": "NVIDIA toolchain only",
    "nvcc_intrinsics": "NVIDIA-specific intrinsics (e.g., __nv_*)",
}


@dataclass
class FileInventory:
    """Inventory of a single source file."""
    path: Path
    language: str  # "cuda", "cpp", "header", "cmake", "make", "python"
    lines: int = 0
    cuda_libraries: Set[str] = field(default_factory=set)
    cuda_patterns: Set[str] = field(default_factory=set)
    includes: List[str] = field(default_factory=list)
    risk_items: List[Dict] = field(default_factory=list)
    supported: bool = True
    block_reason: Optional[str] = None


def detect_language(path: Path) -> str:
    """Detect file language from extension."""
    ext = path.suffix.lower()
    if ext in (".cu", ".cuh"):
        return "cuda"
    if ext in (".cpp", ".cc", ".cxx", ".c"):
        return "cpp"
    if ext in (".hpp", ".hh", ".hxx", ".h"):
        return "header"
    if ext == ".py":
        return "python"
    if path.name.lower() == "cmakelists.txt":
        return "cmake"
    if path.name.lower() in ("makefile", "gnumakefile"):
        return "make"
    return "unknown"


def analyze_cuda_file(path: Path, content: str) -> FileInventory:
    """Analyze a CUDA file for libraries, patterns, and risks."""
    inv = FileInventory(path=path, language="cuda", lines=len(content.splitlines()))

    # Detect CUDA libraries from #include directives
    lib_patterns = {
        "cuda_runtime": r"#include\s*[<\"].*cuda_runtime",
        "cublas": r"#include\s*[<\"].*cublas",
        "cufft": r"#include\s*[<\"].*cufft",
        "curand": r"#include\s*[<\"].*curand",
        "cudnn": r"#include\s*[<\"].*cudnn",
        "nccl": r"#include\s*[<\"].*nccl",
        "cub": r"#include\s*[<\"].*cub",
        "thrust": r"#include\s*[<\"].*thrust",
        "cutlass": r"#include\s*[<\"].*cutlass",
    }
    for lib, pattern in lib_patterns.items():
        if re.search(pattern, content):
            inv.cuda_libraries.add(lib)

    # Detect CUDA patterns
    if re.search(r"__global__\s+", content):
        inv.cuda_patterns.add("kernel_launch")
    if re.search(r"__shared__\s+", content):
        inv.cuda_patterns.add("shared_memory")
    if re.search(r"__shfl|warpSync|__activemask", content):
        inv.cuda_patterns.add("warp_shuffle")
    if re.search(r"cudaStream", content):
        inv.cuda_patterns.add("streams")
    if re.search(r"cudaEvent", content):
        inv.cuda_patterns.add("events")
    if re.search(r"atomicAdd|atomicSub|atomicExch|atomicCAS", content):
        inv.cuda_patterns.add("atomics")

    # Detect blocked patterns
    if re.search(r"asm\s*\(\s*\".*?ptx", content, re.IGNORECASE):
        inv.cuda_patterns.add("inline_ptx")
        inv.risk_items.append({
            "type": "inline_ptx",
            "file": str(path),
            "line": _find_line_number(content, r"asm\s*\(\s*\""),
            "severity": "high",
        })
    if re.search(r"cudaGraphics", content):
        inv.cuda_patterns.add("cuda_graphics_interop")
        inv.risk_items.append({
            "type": "cuda_graphics_interop",
            "file": str(path),
            "line": _find_line_number(content, r"cudaGraphics"),
            "severity": "high",
        })

    # Detect warp-size assumptions (32 on NVIDIA, 64 on AMD)
    if re.search(r"\b32\b.*warp|warp.*\b32\b", content, re.IGNORECASE):
        inv.risk_items.append({
            "type": "warp_size_assumption",
            "file": str(path),
            "line": _find_line_number(content, r"warp.*32|32.*warp"),
            "severity": "medium",
        })

    # Extract #include directives
    for m in re.finditer(r"#include\s*[<\"]([^>\"]+)[>\"]", content):
        inv.includes.append(m.group(1))

    # Check capability matrix
    for lib in inv.cuda_libraries:
        if lib in BLOCKED_LIBRARIES:
            inv.supported = False
            inv.block_reason = BLOCKED_LIBRARIES[lib]

    for pattern in inv.cuda_patterns:
        if pattern in BLOCKED_PATTERNS:
            inv.supported = False
            inv.block_reason = BLOCKED_PATTERNS[pattern]

    return inv


def _find_line_number(content: str, pattern: str) -> int:
    """Find the first line number matching a pattern (1-indexed)."""
    for i, line in enumerate(content.splitlines(), 1):
        if re.search(pattern, line):
            return i
    return 0


# ============================================================
# Repository inventory
# ============================================================

def inventory_repository(repo_path: Path) -> List[FileInventory]:
    """Walk a repository and produce a FileInventory for each relevant file."""
    inventory: List[FileInventory] = []

    relevant_exts = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c",
                     ".hpp", ".hh", ".hxx", ".h", ".py"}
    relevant_names = {"CMakeLists.txt", "Makefile", "GNUmakefile"}

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        # Skip hidden, build, and vendor directories
        parts = path.parts
        if any(p.startswith(".") or p in {"build", "vendor", "node_modules", "__pycache__"}
               for p in parts):
            continue

        if path.suffix.lower() in relevant_exts or path.name in relevant_names:
            language = detect_language(path)
            try:
                content = path.read_text(errors="replace")
            except Exception:
                continue

            if language == "cuda":
                inv = analyze_cuda_file(path, content)
            else:
                inv = FileInventory(path=path, language=language, lines=len(content.splitlines()))

            inventory.append(inv)

    return inventory


# ============================================================
# Planner
# ============================================================

def plan_migration(repo_path: Path) -> MigrationPlan:
    """
    Produce a MigrationPlan for a repository — BEFORE any code is modified.

    The plan classifies each file as:
      - mechanical (hipify-clang will handle it)
      - semantic_repair (hipify + LLM repair)
      - blocked (unsupported construct)
    """
    inventory = inventory_repository(repo_path)

    cuda_files = [f for f in inventory if f.language == "cuda"]
    libraries: Set[str] = set()
    for f in cuda_files:
        libraries.update(f.cuda_libraries)

    risk_items: List[Dict] = []
    blocked_count = 0
    mechanical_count = 0
    semantic_repair_count = 0

    for f in cuda_files:
        risk_items.extend(f.risk_items)
        if not f.supported:
            blocked_count += 1
        elif f.risk_items:
            semantic_repair_count += 1
        else:
            mechanical_count += 1

    supported = blocked_count == 0
    block_reason = None
    if not supported:
        blocked_files = [f for f in cuda_files if not f.supported]
        block_reason = "; ".join(
            f"{f.path.name}: {f.block_reason}" for f in blocked_files[:3]
        )

    return MigrationPlan(
        total_files=len(inventory),
        cuda_translation_units=len(cuda_files),
        libraries=sorted(libraries),
        risk_items=risk_items[:20],  # cap for display
        strategy_counts={
            "mechanical": mechanical_count,
            "semantic_repair": semantic_repair_count,
            "blocked": blocked_count,
        },
        supported=supported,
        block_reason=block_reason,
    )


# ============================================================
# Patch manager
# ============================================================

def generate_unified_diff(original: str, modified: str, filename: str = "file") -> str:
    """Generate a unified diff between original and modified source."""
    import difflib
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines, modified_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}",
    )
    return "".join(diff)


def generate_patch_bundle(changes: List[Dict[str, str]]) -> str:
    """
    Generate a git-style patch bundle from a list of changes.

    Args:
        changes: List of {"filename": str, "original": str, "modified": str}

    Returns:
        Unified diff text suitable for `git apply`.
    """
    patches = []
    for change in changes:
        diff = generate_unified_diff(
            change.get("original", ""),
            change.get("modified", ""),
            change.get("filename", "file"),
        )
        if diff.strip():
            patches.append(diff)
    return "\n".join(patches)
