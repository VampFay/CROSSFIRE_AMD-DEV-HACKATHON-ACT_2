"""
CUDA static analyzer with tree-sitter AST parsing + regex fallback.

Uses tree-sitter for robust C/C++ parsing when available, falls back to
regex patterns for environments where tree-sitter is not installed.
This dual approach gives us:
- Accurate AST-based detection in production (Docker container)
- Graceful degradation in dev environments without tree-sitter
"""
from __future__ import annotations

import re
from typing import List, Set

from loguru import logger

from app.schemas import AnalysisResult, TranslationPattern


# ============================================================
# Tree-sitter setup (lazy — only loads if available)
# ============================================================

_ts_parser = None
_ts_available = False

def _init_tree_sitter():
    """Initialize tree-sitter parser. Sets _ts_available flag."""
    global _ts_parser, _ts_available
    if _ts_available:
        return

    try:
        import tree_sitter
        import tree_sitter_cpp as ts_cpp

        language = tree_sitter.Language(ts_cpp.language())
        _ts_parser = tree_sitter.Parser(language)
        _ts_available = True
        logger.debug("Tree-sitter C/C++ parser initialized")
    except ImportError:
        logger.debug("tree-sitter not available, using regex-only analysis")
        _ts_available = False
    except Exception as e:
        logger.warning(f"Tree-sitter init failed: {e}, using regex-only")
        _ts_available = False


# ============================================================
# Regex patterns (used for both regex-only mode and AST augmentation)
# ============================================================

KERNEL_PATTERN = re.compile(r"__global__\s+\w+\s+(\w+)\s*\(", re.MULTILINE)
SHARED_MEMORY_PATTERN = re.compile(r"__shared__\s+", re.MULTILINE)
WARP_SHUFFLE_PATTERN = re.compile(r"__shfl_(?:down|up|xor|idx)_sync\s*\(", re.MULTILINE)
WARP_REDUCE_PATTERN = re.compile(r"__(?:any|all|ballot)_sync\s*\(", re.MULTILINE)

CUBLAS_PATTERN = re.compile(
    r"\bcublas(?:Handle_t|Create|Destroy|Sgemm|Dgemm|Gemv|Gemm|Axpy|Copy|Scal|SetMatrix|GetMatrix|SetVector|GetVector)\b",
    re.MULTILINE,
)
CUDNN_PATTERN = re.compile(
    r"\bcudnn(?:Handle_t|Create|Destroy|SetTensor4dDescriptor|SetConvolution2dDescriptor|ActivationDescriptor|ConvolutionForward|PoolingForward|BatchNormalizationForwardTraining|BatchNormalizationForwardInference)\b",
    re.MULTILINE,
)
THRUST_PATTERN = re.compile(
    r"\bthrust::(?:device_vector|host_vector|reduce|sort|copy|fill|sequence|gather|scatter|transform|unique)\b",
    re.MULTILINE,
)
TRITON_PATTERN = re.compile(r"@\s*triton\.jit\b", re.MULTILINE)
PYTORCH_EXT_PATTERN = re.compile(
    r"\b(?:PYBIND11_MODULE|TORCH_LIBRARY|torch::Tensor|torch::cuda|ATen::cuda)\b",
    re.MULTILINE,
)
STREAMS_PATTERN = re.compile(r"\bcudaStream(?:Create|Destroy|Synchronize|WaitEvent)\b", re.MULTILINE)
EVENTS_PATTERN = re.compile(r"\bcudaEvent(?:Create|Destroy|Record|Synchronize|ElapsedTime)\b", re.MULTILINE)

CUDAMALLOC_PATTERN = re.compile(r"\bcudaMalloc\s*\(", re.MULTILINE)
CUDAFREE_PATTERN = re.compile(r"\bcudaFree\s*\(", re.MULTILINE)
CUDAMEMCPY_PATTERN = re.compile(r"\bcudaMemcpy\s*\(", re.MULTILINE)
KERNEL_LAUNCH_PATTERN = re.compile(r"(\w+)\s*<<<\s*[^>]+>\s*>>\s*\(", re.MULTILINE)
SYNC_PATTERN = re.compile(r"\bcudaDeviceSynchronize\s*\(", re.MULTILINE)

INCLUDE_PATTERN = re.compile(r'#\s*include\s+["<]([^>"]+)[>"]', re.MULTILINE)


# ============================================================
# Tree-sitter AST analysis
# ============================================================

def _analyze_with_tree_sitter(source: str) -> dict:
    """Analyze CUDA source using tree-sitter AST.

    Returns dict with:
        - kernel_count: number of __global__ functions
        - has_shared_memory: bool
        - has_warp_primitives: bool
        - function_names: list of all function names
        - call_expressions: list of all function calls
    """
    _init_tree_sitter()

    if not _ts_available:
        return {}

    result = {
        "kernel_count": 0,
        "has_shared_memory": False,
        "has_warp_primitives": False,
        "function_names": [],
        "call_expressions": [],
    }

    try:
        tree = _ts_parser.parse(source.encode("utf-8"))
        root = tree.root_node

        def visit(node):
            # Function definitions
            if node.type == "function_definition":
                for child in node.children:
                    if child.type == "function_declarator":
                        for decl in child.children:
                            if decl.type == "identifier":
                                name = source[decl.start_byte:decl.end_byte]
                                result["function_names"].append(name)

                # Check for __global__ qualifier
                func_text = source[node.start_byte:node.end_byte]
                if "__global__" in func_text:
                    result["kernel_count"] += 1

                # Check for shared memory and warp primitives inside function body
                if "__shared__" in func_text:
                    result["has_shared_memory"] = True
                if "__shfl_" in func_text or "__any_sync" in func_text or "__all_sync" in func_text:
                    result["has_warp_primitives"] = True

            # Call expressions
            elif node.type == "call_expression":
                for child in node.children:
                    if child.type == "identifier":
                        name = source[child.start_byte:child.end_byte]
                        result["call_expressions"].append(name)

            # Recurse
            for child in node.children:
                visit(child)

        visit(root)

    except Exception as e:
        logger.warning(f"Tree-sitter parse failed: {e}, falling back to regex")
        return {}

    return result


# ============================================================
# Library call detector
# ============================================================

def _detect_library_calls(source: str, ts_calls: list[str] = None) -> List[str]:
    """Detect all CUDA library calls in the source."""
    calls: Set[str] = set()

    # From tree-sitter (more accurate)
    if ts_calls:
        for call in ts_calls:
            if call.startswith("cublas") or call.startswith("cudnn") or call.startswith("thrust"):
                calls.add(call)

    # From regex (catches what tree-sitter might miss)
    for m in CUBLAS_PATTERN.finditer(source):
        calls.add(m.group(0))
    for m in CUDNN_PATTERN.finditer(source):
        calls.add(m.group(0))
    for m in THRUST_PATTERN.finditer(source):
        calls.add(m.group(0))

    return sorted(calls)


# ============================================================
# Difficulty scorer
# ============================================================

def _compute_difficulty(
    patterns: List[TranslationPattern],
    kernel_count: int,
    has_shared_memory: bool,
    has_warp_primitives: bool,
    library_calls: List[str],
) -> float:
    """Compute a difficulty score in [0, 1]."""
    score = 0.1

    if kernel_count > 0:
        score += 0.2
    if kernel_count > 1:
        score += 0.05 * (kernel_count - 1)

    if has_shared_memory:
        score += 0.15

    if has_warp_primitives:
        score += 0.25

    pattern_set = set(patterns)
    if TranslationPattern.CUDNN in pattern_set:
        score += 0.3
    if TranslationPattern.TRITON in pattern_set:
        score += 0.3
    if TranslationPattern.CUBLAS in pattern_set:
        score += 0.1
    if TranslationPattern.THRUST in pattern_set:
        score += 0.1
    if TranslationPattern.PYTORCH_EXTENSION in pattern_set:
        score += 0.1

    return min(score, 1.0)


# ============================================================
# Include extraction
# ============================================================

def _extract_includes(source: str) -> List[str]:
    """Extract local #include dependencies (not system headers)."""
    deps: List[str] = []
    for m in INCLUDE_PATTERN.finditer(source):
        header = m.group(1)
        # Only include local headers (in quotes) or .cuh/.h files
        if header.endswith(".cuh") or header.endswith(".h") or header.endswith(".hpp"):
            # Exclude system/library headers
            system_prefixes = ("cuda", "cublas", "cudnn", "thrust", "stdio", "stdlib",
                             "math", "string", "vector", "algorithm", "iostream",
                             "stdint", "stddef", "assert", "ctype", "time")
            if not any(header.startswith(p) for p in system_prefixes):
                deps.append(header)
    return deps


# ============================================================
# Main analyzer
# ============================================================

class CudaAnalyzer:
    """Static analyzer for CUDA source code.

    Uses tree-sitter for AST-based analysis when available, falls back
    to regex patterns otherwise. The two approaches are complementary:
    tree-sitter gives accurate function/call detection, regex catches
    preprocessor directives and string patterns that tree-sitter misses.
    """

    def analyze(self, source: str, filename: str = "input.cu") -> AnalysisResult:
        """Analyze CUDA source and return structured analysis.

        Args:
            source: CUDA source code as string.
            filename: Original filename (for context).

        Returns:
            AnalysisResult with patterns, difficulty score, and metadata.
        """
        logger.debug(f"Analyzing {filename}: {len(source)} chars (tree-sitter: {_ts_available or 'pending'})")

        # Run tree-sitter analysis (populates kernel_count, has_shared_memory, etc.)
        ts_result = _analyze_with_tree_sitter(source)
        ts_calls = ts_result.get("call_expressions", [])

        # Detect patterns (use tree-sitter data where available, regex as backup)
        patterns: List[TranslationPattern] = []

        kernel_count = ts_result.get("kernel_count") if ts_result else 0
        if kernel_count == 0:
            # Regex fallback
            kernel_count = len(list(KERNEL_PATTERN.finditer(source)))
        if kernel_count > 0:
            patterns.append(TranslationPattern.KERNEL)

        has_shared_memory = ts_result.get("has_shared_memory") if ts_result else False
        if not has_shared_memory and SHARED_MEMORY_PATTERN.search(source):
            has_shared_memory = True
        if has_shared_memory:
            patterns.append(TranslationPattern.SHARED_MEMORY)

        has_warp_primitives = ts_result.get("has_warp_primitives") if ts_result else False
        if not has_warp_primitives and (WARP_SHUFFLE_PATTERN.search(source) or WARP_REDUCE_PATTERN.search(source)):
            has_warp_primitives = True
        if has_warp_primitives:
            patterns.append(TranslationPattern.WARP_SHUFFLE)

        if CUBLAS_PATTERN.search(source):
            patterns.append(TranslationPattern.CUBLAS)

        if CUDNN_PATTERN.search(source):
            patterns.append(TranslationPattern.CUDNN)

        if THRUST_PATTERN.search(source):
            patterns.append(TranslationPattern.THRUST)

        if TRITON_PATTERN.search(source):
            patterns.append(TranslationPattern.TRITON)

        if PYTORCH_EXT_PATTERN.search(source):
            patterns.append(TranslationPattern.PYTORCH_EXTENSION)

        if STREAMS_PATTERN.search(source):
            patterns.append(TranslationPattern.STREAMS)

        if EVENTS_PATTERN.search(source):
            patterns.append(TranslationPattern.EVENTS)

        # Detect library calls
        library_calls = _detect_library_calls(source, ts_calls)

        # Compute difficulty
        difficulty = _compute_difficulty(
            patterns=patterns,
            kernel_count=kernel_count,
            has_shared_memory=has_shared_memory,
            has_warp_primitives=has_warp_primitives,
            library_calls=library_calls,
        )

        # File dependencies
        deps = _extract_includes(source)

        # Build notes
        notes_parts = []
        if kernel_count > 0:
            notes_parts.append(f"{kernel_count} kernel(s)")
        if has_shared_memory:
            notes_parts.append("uses shared memory")
        if has_warp_primitives:
            notes_parts.append("uses warp primitives")
        if library_calls:
            notes_parts.append(f"library calls: {', '.join(library_calls[:3])}")
        analyzer_method = "tree-sitter+regex" if ts_result else "regex-only"
        notes_parts.append(f"analyzed via {analyzer_method}")
        notes = "; ".join(notes_parts) if notes_parts else "simple syntactic translation"

        result = AnalysisResult(
            patterns=patterns,
            difficulty_score=difficulty,
            kernel_count=kernel_count,
            has_shared_memory=has_shared_memory,
            has_warp_primitives=has_warp_primitives,
            library_calls=library_calls,
            file_dependencies=deps,
            notes=notes,
        )

        logger.debug(f"Analysis result: {result.notes}, difficulty={difficulty:.2f}")
        return result


# ============================================================
# Convenience function
# ============================================================

def analyze_cuda(source: str, filename: str = "input.cu") -> AnalysisResult:
    """Analyze CUDA source (module-level convenience function)."""
    return CudaAnalyzer().analyze(source, filename)
