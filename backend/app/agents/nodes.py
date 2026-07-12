"""
LangGraph node implementations.

Each node is an async function that takes AgentState and returns a partial
state update. The graph in graph.py wires these together.

FIXES vs. previous version:
  - stub_mode propagated from model client to state
  - GPU attestation captured at compile time (not just runtime metrics)
  - Baseline I/O moved to asyncio.to_thread (no event-loop blocking)
  - debug_node broadcasts the correct (next) iteration number
  - Per-iteration strategy tracked (mechanical vs. agent_repaired vs. llm_only)
"""
from __future__ import annotations

import time
from typing import Any, Dict

from loguru import logger

from app.agents.state import AgentState
from app.analyzers.cuda_analyzer import CudaAnalyzer
from app.config import settings
from app.agents.routing import ModelRouter
from app.rag.retriever import RAGRetriever
from app.sandbox.compiler import SandboxClient
from app.schemas import MigrationStrategy, ModelChoice
from app.ws.handler import broadcast


# ============================================================
# Node 1: ANALYZE
# ============================================================

async def analyze_node(state: AgentState) -> Dict[str, Any]:
    """Analyze the CUDA source code statically."""
    job_id = state["job_id"]
    logger.info(f"[{job_id}] ANALYZE: {state['filename']}")

    await broadcast(job_id, "status", {"state": "analyzing", "iteration": 0})

    analyzer = CudaAnalyzer()
    analysis = analyzer.analyze(state["cuda_source"], state["filename"])

    logger.info(
        f"[{job_id}] Analysis: difficulty={analysis.difficulty_score:.2f}, "
        f"patterns={[p.value for p in analysis.patterns]}, "
        f"kernels={analysis.kernel_count}"
    )

    return {"analysis": analysis}


# ============================================================
# Node 2: TRANSLATE
# ============================================================

async def translate_node(state: AgentState) -> Dict[str, Any]:
    """Translate CUDA to ROCm using hybrid routing.

    If hipify-clang is available and the previous iteration didn't fail,
    we use the HIPIFY output as the starting point and only call the LLM
    if compilation fails (see compile_node → debug_node flow).
    """
    job_id = state["job_id"]
    iteration = state.get("iteration", 0)
    logger.info(f"[{job_id}] TRANSLATE: iteration={iteration + 1}")

    await broadcast(job_id, "status", {
        "state": "translating",
        "iteration": iteration + 1,
    })

    # ---- Try HIPIFY first (deterministic mechanical pass) ----
    hipified_code = None
    strategy = MigrationStrategy.LLM_ONLY
    if iteration == 0:  # only first pass; later iterations use LLM repair
        try:
            from app.migration.hipify_adapter import hipify_file
            hipified_code = await hipify_file(state["cuda_source"], state["filename"])
            if hipified_code is not None:
                strategy = MigrationStrategy.MECHANICAL
                logger.info(f"[{job_id}] HIPIFY succeeded — using mechanical translation")
            else:
                logger.info(f"[{job_id}] HIPIFY returned None — using LLM")
        except ImportError:
            logger.debug(f"[{job_id}] HIPIFY adapter not available — using LLM only")
        except Exception as e:
            logger.warning(f"[{job_id}] HIPIFY failed: {e} — falling back to LLM")

    # If hipify succeeded and we have no prior error feedback, skip the LLM
    if hipified_code is not None and not state.get("error_feedback"):
        return {
            "translated_code": hipified_code,
            "model_used": ModelChoice.LOCAL,
            "iteration": iteration + 1,
            "total_tokens": state["total_tokens"],
            "total_cost_usd": state["total_cost_usd"],
            "total_latency_ms": state["total_latency_ms"],
            "last_iter_tokens": 0,
            "last_iter_cost": 0.0,
            "last_iter_latency": 0,
            "migration_strategy": strategy,
            "stub_mode": False,
            "compile_result": None,
            "run_result": None,
            "diff_result": None,
        }

    # ---- LLM translation (first pass or repair after hipify failed) ----
    rag_context = state.get("rag_context", "")
    if not rag_context:
        retriever = RAGRetriever()
        try:
            chunks = retriever.retrieve(state["cuda_source"], top_k=settings.rag_top_k)
            rag_context = "\n\n---\n\n".join(chunks)
        except Exception as e:
            logger.warning(f"[{job_id}] RAG retrieval failed: {e}")
            rag_context = ""

    # If we have a hipified base and error feedback, ask the LLM to REPAIR it
    cuda_source_for_llm = state["cuda_source"]
    extra_feedback = ""
    if hipified_code is not None and state.get("error_feedback"):
        cuda_source_for_llm = hipified_code
        extra_feedback = (
            "\n\nNote: The input below has already been mechanically hipified. "
            "Fix the compilation/runtime errors while preserving the hipified API names."
        )
        strategy = MigrationStrategy.AGENT_REPAIRED

    router = ModelRouter()
    router.reset_counters()
    analysis = state["analysis"]
    force_remote = state.get("force_remote", False)
    model_choice = router.route(analysis, force_remote=force_remote)

    logger.info(f"[{job_id}] Routing to: {model_choice.value}")

    start = time.time()
    model = router.get_model(model_choice)
    translated = await model.translate(
        cuda_source=cuda_source_for_llm,
        rag_context=rag_context,
        error_feedback=(state.get("error_feedback") or "") + extra_feedback,
    )
    latency_ms = int((time.time() - start) * 1000)

    new_tokens = state["total_tokens"] + model.tokens_used
    new_cost = state["total_cost_usd"] + model.cost_usd
    new_latency = state["total_latency_ms"] + latency_ms
    stub_mode = getattr(model, "used_stub", False)

    logger.info(
        f"[{job_id}] Translation done: {len(translated)} chars, "
        f"{model.tokens_used} tokens, ${model.cost_usd:.4f}, {latency_ms}ms, "
        f"stub={stub_mode}, strategy={strategy.value}"
    )

    return {
        "translated_code": translated,
        "rag_context": rag_context,
        "model_used": model_choice,
        "iteration": iteration + 1,
        "total_tokens": new_tokens,
        "total_cost_usd": new_cost,
        "total_latency_ms": new_latency,
        "last_iter_tokens": model.tokens_used,
        "last_iter_cost": model.cost_usd,
        "last_iter_latency": latency_ms,
        "migration_strategy": strategy,
        "stub_mode": stub_mode,
        "compile_result": None,
        "run_result": None,
        "diff_result": None,
    }


# ============================================================
# Node 3: COMPILE
# ============================================================

async def compile_node(state: AgentState) -> Dict[str, Any]:
    """Compile the translated code with hipcc."""
    job_id = state["job_id"]
    logger.info(f"[{job_id}] COMPILE")

    await broadcast(job_id, "status", {
        "state": "compiling",
        "iteration": state["iteration"],
    })

    sandbox = SandboxClient()
    result = await sandbox.compile_direct(
        state["translated_code"],
        filename=state["filename"].replace(".cu", ".hip"),
    )

    logger.info(f"[{job_id}] Compile: success={result.success}")

    # Capture GPU attestation at compile time (proves which toolchain ran)
    gpu_attestation = None
    try:
        attestation = sandbox.collect_attestation()
        gpu_attestation = attestation
    except Exception as e:
        logger.warning(f"[{job_id}] GPU attestation collection failed: {e}")

    return {
        "compile_result": result.model_dump(),
        "gpu_attestation": gpu_attestation,
    }


# ============================================================
# Node 4: RUN
# ============================================================

async def run_node(state: AgentState) -> Dict[str, Any]:
    """Run the compiled binary on AMD GPU with sample inputs."""
    job_id = state["job_id"]
    logger.info(f"[{job_id}] RUN")

    await broadcast(job_id, "status", {
        "state": "running",
        "iteration": state["iteration"],
    })

    sandbox = SandboxClient()
    compile_result = state["compile_result"]

    if not compile_result or not compile_result.get("success"):
        return {"run_result": {"success": False, "stderr": "Compile failed"}}

    binary_path = compile_result["binary_path"]
    inputs = await _load_baseline_inputs(state["filename"])

    result = await sandbox.run_direct(binary_path, inputs, timeout=settings.sandbox_timeout)

    logger.info(f"[{job_id}] Run: success={result.success}")

    # Collect GPU metrics (real rocm-smi data)
    gpu_metrics_dict = None
    try:
        from app.sandbox.gpu_metrics import collect_gpu_metrics
        gpu_metrics = await collect_gpu_metrics()
        if gpu_metrics is not None:
            gpu_metrics_dict = gpu_metrics.to_dict()
            logger.info(
                f"[{job_id}] GPU: {gpu_metrics.gpu_name} ({gpu_metrics.gpu_arch}), "
                f"VRAM {gpu_metrics.vram_used_mb}/{gpu_metrics.vram_total_mb} MB"
            )
        else:
            gpu_metrics_dict = {
                "gpu_name": "Demo Mode",
                "gpu_arch": "",
                "vram_total_mb": 0,
                "vram_used_mb": 0,
                "vram_free_mb": 0,
                "gpu_utilization_pct": 0.0,
                "temperature_c": 0.0,
                "power_draw_w": 0.0,
                "captured_at": "",
                "stub_mode": True,
            }
    except Exception as e:
        logger.warning(f"[{job_id}] GPU metrics collection failed: {e}")
        gpu_metrics_dict = {
            "gpu_name": "Demo Mode",
            "gpu_arch": "",
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "vram_free_mb": 0,
            "gpu_utilization_pct": 0.0,
            "temperature_c": 0.0,
            "power_draw_w": 0.0,
            "captured_at": "",
            "stub_mode": True,
        }

    # Cleanup binary to avoid /tmp filling up
    sandbox._cleanup_files(binary_path)

    return {
        "run_result": result.model_dump(),
        "gpu_metrics": gpu_metrics_dict,
    }


# ============================================================
# Node 5: DIFF
# ============================================================

async def diff_node(state: AgentState) -> Dict[str, Any]:
    """Numerically diff run outputs against CUDA baseline."""
    job_id = state["job_id"]
    logger.info(f"[{job_id}] DIFF")

    await broadcast(job_id, "status", {
        "state": "diffing",
        "iteration": state["iteration"],
    })

    sandbox = SandboxClient()
    run_result = state["run_result"]
    baseline_outputs = await _load_baseline_outputs(state["filename"])

    if not run_result or not run_result.get("success"):
        return {"diff_result": {"success": False, "threshold": settings.agent_diff_threshold, "baseline_source": "analytical"}}

    diff = sandbox.diff(
        run_result.get("outputs", {}),
        baseline_outputs,
        threshold=settings.agent_diff_threshold,
        baseline_source="analytical",  # our baselines are analytically computed
    )

    logger.info(f"[{job_id}] Diff: success={diff.success}, max_err={diff.max_abs_error}")

    history = state.get("history", [])
    compile_result = state.get("compile_result", {}) or {}
    history.append({
        "iteration": state["iteration"],
        "model_used": (
            state["model_used"].value
            if hasattr(state.get("model_used", ModelChoice.LOCAL), "value")
            else state.get("model_used", "local")
        ),
        "compile_success": compile_result.get("success", False),
        "run_success": run_result.get("success", False),
        "diff_success": diff.success,
        "tokens_used": state.get("last_iter_tokens", 0),
        "cost_usd": state.get("last_iter_cost", 0.0),
        "latency_ms": state.get("last_iter_latency", 0),
        "error_feedback": state.get("error_feedback"),
        "strategy": state.get("migration_strategy", MigrationStrategy.LLM_ONLY).value
            if hasattr(state.get("migration_strategy", MigrationStrategy.LLM_ONLY), "value")
            else state.get("migration_strategy", "llm_only"),
    })

    return {
        "diff_result": diff.model_dump(),
        "history": history,
    }


# ============================================================
# Node 6: DEBUG (formats error feedback)
# ============================================================

async def debug_node(state: AgentState) -> Dict[str, Any]:
    """Format compile/run/diff errors as feedback for the next iteration."""
    job_id = state["job_id"]
    logger.info(f"[{job_id}] DEBUG")

    # Broadcast the NEXT iteration number (the one about to start)
    await broadcast(job_id, "status", {
        "state": "debugging",
        "iteration": state["iteration"] + 1,  # ← was stale (off-by-one)
    })

    feedback_parts = []

    compile_result = state.get("compile_result", {})
    if compile_result and not compile_result.get("success"):
        errors = compile_result.get("errors", "")
        feedback_parts.append(f"COMPILE ERRORS:\n{errors}")

    run_result = state.get("run_result", {})
    if run_result and not run_result.get("success"):
        stderr = run_result.get("stderr", "")
        feedback_parts.append(f"RUNTIME ERRORS:\n{stderr}")

    diff_result = state.get("diff_result", {})
    if diff_result and not diff_result.get("success"):
        max_err = diff_result.get("max_abs_error", "?")
        mse = diff_result.get("mse", "?")
        feedback_parts.append(
            f"OUTPUT MISMATCH: max_abs_error={max_err}, mse={mse}, "
            f"threshold={diff_result.get('threshold', 1e-5)}"
        )

    feedback = "\n\n".join(feedback_parts) if feedback_parts else "Unknown error"
    logger.debug(f"[{job_id}] Feedback: {feedback[:200]}...")

    return {"error_feedback": feedback}


# ============================================================
# Helper functions (async — no event-loop blocking)
# ============================================================

async def _load_baseline_inputs(filename: str) -> Dict[str, Any]:
    """Load cached sample inputs for a CUDA filename (async, non-blocking)."""
    import json
    from pathlib import Path

    def _read():
        baselines_dir = Path(__file__).parent.parent.parent.parent / "samples" / "baselines"
        inputs_file = baselines_dir / f"{Path(filename).stem}_inputs.json"
        if inputs_file.exists():
            with open(inputs_file) as f:
                return json.load(f)
        return {"N": 1024, "seed": 42}

    import asyncio
    return await asyncio.to_thread(_read)


async def _load_baseline_outputs(filename: str) -> Dict[str, Any]:
    """Load cached CUDA baseline outputs for diffing (async, non-blocking)."""
    import json
    from pathlib import Path

    def _read():
        baselines_dir = Path(__file__).parent.parent.parent.parent / "samples" / "baselines"
        outputs_file = baselines_dir / f"{Path(filename).stem}_outputs.json"
        if outputs_file.exists():
            with open(outputs_file) as f:
                return json.load(f)
        logger.warning(f"No baseline outputs for {filename}, validation will fail")
        return {}

    import asyncio
    return await asyncio.to_thread(_read)
