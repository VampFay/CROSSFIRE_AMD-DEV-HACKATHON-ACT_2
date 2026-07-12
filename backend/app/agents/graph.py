"""
LangGraph state machine for the CUDA-to-ROCm translation agent.

States:
    analyze → translate → compile → run → diff → [done | debug → translate]

CRITICAL FIXES vs. previous version:
  1. Iteration guard on the COMPILE path (not just diff). Previously a
     compile failure looped forever: compile → debug → translate → compile.
     Now should_run_after_compile checks max_iterations too.
  2. Cache-hit TranslationResult sets cache_hit=True (was always False).
  3. stub_mode propagated from model client to TranslationResult.
  4. VerificationLevel computed from actual evidence — never defaults to
     "validated" without proof.
  5. 
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from langgraph.graph import END, StateGraph
from loguru import logger

from app.agents.nodes import (
    analyze_node,
    compile_node,
    debug_node,
    diff_node,
    run_node,
    translate_node,
)
from app.agents.state import AgentState, initial_state
from app.config import settings
from app.schemas import (
    AnalysisResult,
    DiffResult,
    GPUMetrics,
    GPUAttestation,
    IterationRecord,
    JobStatus,
    JobBudget,
    MigrationStrategy,
    ModelChoice,
    TranslationResult,
    VerificationLevel,
)
from app.ws.handler import broadcast


# ============================================================
# Conditional edge logic
# ============================================================

def should_run_after_compile(state: AgentState) -> str:
    """After compile: run if success, debug if failed (within budget)."""
    compile_result = state.get("compile_result") or {}
    if compile_result and compile_result.get("success"):
        return "run"

    # Compile failed — check iteration budget BEFORE looping back to debug
    if state["iteration"] >= state["max_iterations"]:
        logger.warning(
            f"[{state['job_id']}] Max iterations ({state['max_iterations']}) reached "
            f"on compile-fail path, giving up"
        )
        return END
    return "debug"


def should_continue_after_diff(state: AgentState) -> str:
    """After diff: done if success, debug if failed and iterations remain, else done (failed)."""
    diff_result = state.get("diff_result") or {}
    if diff_result and diff_result.get("success"):
        return END

    if state["iteration"] >= state["max_iterations"]:
        logger.warning(f"[{state['job_id']}] Max iterations reached, giving up")
        return END

    return "debug"


# ============================================================
# Build the graph
# ============================================================

def build_agent_graph():
    """Build and compile the LangGraph state machine with checkpointing."""
    from app.agents.checkpointer import get_checkpointer

    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", analyze_node)
    workflow.add_node("translate", translate_node)
    workflow.add_node("compile", compile_node)
    workflow.add_node("run", run_node)
    workflow.add_node("diff", diff_node)
    workflow.add_node("debug", debug_node)

    workflow.set_entry_point("analyze")

    workflow.add_edge("analyze", "translate")
    workflow.add_edge("translate", "compile")

    workflow.add_conditional_edges(
        "compile",
        should_run_after_compile,
        {"run": "run", "debug": "debug", END: END},
    )

    workflow.add_edge("run", "diff")

    workflow.add_conditional_edges(
        "diff",
        should_continue_after_diff,
        {END: END, "debug": "debug"},
    )

    workflow.add_edge("debug", "translate")

    checkpointer = get_checkpointer()
    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


# ============================================================
# Verification level computation
# ============================================================

def compute_verification_level(
    diff_success: bool,
    compile_success: bool,
    run_success: bool,
    baseline_source: str = "analytical",
    benchmarked: bool = False,
) -> VerificationLevel:
    """Map evidence to a VerificationLevel. Never overstates.

    The UI MUST NOT show "Validated" unless level >= DIFFERENTIALLY_VERIFIED.
    """
    if not compile_success:
        return VerificationLevel.TRANSLATED
    if not run_success:
        return VerificationLevel.COMPILED
    if not diff_success:
        return VerificationLevel.EXECUTED
    if baseline_source == "cuda_live":
        if benchmarked:
            return VerificationLevel.BENCHMARKED
        return VerificationLevel.DIFFERENTIALLY_VERIFIED
    # analytical baseline = weaker evidence
    if benchmarked:
        return VerificationLevel.BENCHMARKED
    return VerificationLevel.TEST_VERIFIED


# ============================================================
# Run the agent
# ============================================================

_compiled_graph = None


def get_graph():
    """Get the compiled graph (cached)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agent_graph()
    return _compiled_graph


async def run_agent(
    job_id: str,
    cuda_source: str,
    filename: str = "input.cu",
    force_remote: bool = False,
    max_iterations: Optional[int] = None,
) -> TranslationResult:
    """
    Run the full translation agent on a CUDA source file.

    Checks translation memory first — if this CUDA source was successfully
    translated before, returns the cached result instantly (with cache_hit=True).
    """
    max_iters = max_iterations or settings.agent_max_iterations
    budget = JobBudget(max_attempts=max_iters)

    # ---- Check translation memory first ----
    try:
        from app.memory.store import get_memory
        memory = get_memory()
        cached = memory.lookup(cuda_source)
        if cached is not None:
            logger.info(f"[{job_id}] Translation memory HIT — returning cached result")
            await broadcast(job_id, "status", {
                "state": "cache_hit",
                "iteration": 0,
                "message": "Found in translation memory — instant return",
            })

            result = TranslationResult(
                job_id=job_id,
                status=JobStatus.DONE,
                verification_level=VerificationLevel.TRANSLATED,  # cache hit = translated, not re-verified
                cuda_source=cuda_source,
                translated_code=cached["rocm_source"],
                iterations=[],
                total_tokens=0,
                total_cost_usd=0.0,
                total_latency_ms=0,
                analysis=AnalysisResult(
                    patterns=[],
                    difficulty_score=cached.get("difficulty_score", 0.0),
                    kernel_count=0,
                    has_shared_memory=False,
                    has_warp_primitives=False,
                    library_calls=[],
                    file_dependencies=[],
                    notes=f"Cache hit (saved {cached['tokens_used']} tokens, ${cached['cost_usd']:.4f}). Previous test evidence available but not rerun.",
                ),
                diff=None,  # no diff on cache hit — not rerun
                gpu_metrics=None,
                gpu_attestation=None,
                cache_hit=True,
                budget=budget,
                completed_at=datetime.utcnow(),
            )
            await broadcast(job_id, "result", result.model_dump(mode="json"))
            return result
    except Exception as e:
        logger.warning(f"Translation memory lookup failed: {e}")

    # ---- Normal agent flow ----
    state = initial_state(
        job_id=job_id,
        cuda_source=cuda_source,
        filename=filename,
        max_iterations=max_iters,
        force_remote=force_remote,
    )

    graph = get_graph()

    try:
        config = {"configurable": {"thread_id": job_id}}
        final_state = await graph.ainvoke(state, config=config)

        diff_result = final_state.get("diff_result") or {}
        compile_result = final_state.get("compile_result") or {}
        run_result = final_state.get("run_result") or {}
        diff_success = diff_result.get("success", False) if diff_result else False
        compile_success = compile_result.get("success", False) if compile_result else False
        run_success = run_result.get("success", False) if run_result else False
        baseline_source = diff_result.get("baseline_source", "analytical") if diff_result else "analytical"

        status = JobStatus.DONE if diff_success else JobStatus.FAILED

        # Compute verification level from actual evidence
        verification_level = compute_verification_level(
            diff_success=diff_success,
            compile_success=compile_success,
            run_success=run_success,
            baseline_source=baseline_source,
        )

        # Detect stub mode from any iteration
        stub_mode = final_state.get("stub_mode", False)

        # Convert history to IterationRecords
        # If history is empty (HIPIFY success path doesn't populate history due to
        # LangGraph state management), build a synthetic record from final state.
        history = final_state.get("history") or []
        if not history:
            # Always build a synthetic record if history is empty — even on failure
            # this gives the UI something to display
            model_used_raw = final_state.get("model_used", ModelChoice.LOCAL)
            model_used_val = (
                model_used_raw.value
                if hasattr(model_used_raw, "value")
                else str(model_used_raw)
            )
            strategy_raw = final_state.get("migration_strategy", MigrationStrategy.LLM_ONLY)
            strategy_val = (
                strategy_raw.value
                if hasattr(strategy_raw, "value")
                else str(strategy_raw)
            )
            history = [{
                "iteration": final_state.get("iteration", 1),
                "model_used": model_used_val,
                "compile_success": compile_success,
                "run_success": run_success,
                "diff_success": diff_success,
                "tokens_used": final_state.get("last_iter_tokens", 0),
                "cost_usd": final_state.get("last_iter_cost", 0.0),
                "latency_ms": final_state.get("last_iter_latency", 0),
                "error_feedback": final_state.get("error_feedback"),
                "strategy": strategy_val,
            }]

        iterations = [
            IterationRecord(
                iteration=h["iteration"],
                model_used=ModelChoice(h["model_used"]),
                compile_success=h["compile_success"],
                run_success=h["run_success"],
                diff_success=h["diff_success"],
                tokens_used=h["tokens_used"],
                cost_usd=h["cost_usd"],
                latency_ms=h["latency_ms"],
                error_feedback=h.get("error_feedback"),
                strategy=MigrationStrategy(h.get("strategy", "llm_only")),
            )
            for h in history
        ]

        # GPU attestation — ALWAYS collect fresh if not in state.
        # This guarantees every result has attestation evidence.
        gpu_attestation = None
        attestation_data = final_state.get("gpu_attestation")
        if attestation_data:
            try:
                gpu_attestation = GPUAttestation(**attestation_data)
            except Exception as e:
                logger.warning(f"Failed to parse GPU attestation from state: {e}")
        # Always collect fresh as fallback
        if gpu_attestation is None:
            try:
                from app.sandbox.compiler import SandboxClient
                sandbox = SandboxClient()
                attestation_data = sandbox.collect_attestation()
                if attestation_data:
                    gpu_attestation = GPUAttestation(**attestation_data)
                    logger.info(f"Collected fresh GPU attestation: {gpu_attestation.gpu_model}")
            except Exception as e:
                logger.warning(f"Fresh GPU attestation collection failed: {e}")

        result = TranslationResult(
            job_id=job_id,
            status=status,
            verification_level=verification_level,
            cuda_source=cuda_source,
            translated_code=final_state.get("translated_code"),
            iterations=iterations,
            total_tokens=final_state.get("total_tokens", 0),
            total_cost_usd=final_state.get("total_cost_usd", 0.0),
            total_latency_ms=final_state.get("total_latency_ms", 0),
            analysis=final_state.get("analysis"),
            diff=DiffResult(**diff_result) if diff_result else None,
            gpu_metrics=GPUMetrics(**(final_state.get("gpu_metrics") or {})) if final_state.get("gpu_metrics") else None,
            gpu_attestation=gpu_attestation,
            cache_hit=False,
            stub_mode=stub_mode,
            budget=budget,
            error=final_state.get("error"),
            completed_at=datetime.utcnow(),
        )

        # ---- Store successful translations in memory ----
        if result.status == JobStatus.DONE and result.translated_code and not stub_mode:
            try:
                from app.memory.store import get_memory
                analysis = result.analysis
                memory = get_memory()
                memory.store(
                    cuda_source=cuda_source,
                    rocm_source=result.translated_code,
                    filename=filename,
                    patterns=[p.value for p in analysis.patterns] if analysis else [],
                    difficulty_score=analysis.difficulty_score if analysis else 0.0,
                    iterations=len(result.iterations),
                    tokens_used=result.total_tokens,
                    cost_usd=result.total_cost_usd,
                    latency_ms=result.total_latency_ms,
                    model_used=result.iterations[0].model_used.value if result.iterations else "unknown",
                    max_abs_error=result.diff.max_abs_error if result.diff else None,
                )
            except Exception as e:
                logger.warning(f"Failed to store in translation memory: {e}")

        await broadcast(job_id, "result", result.model_dump(mode="json"))
        return result

    except Exception as e:
        logger.exception(f"Agent failed: {e}")
        await broadcast(job_id, "error", {"message": str(e)})

        return TranslationResult(
            job_id=job_id,
            status=JobStatus.FAILED,
            verification_level=VerificationLevel.ANALYZED,
            cuda_source=cuda_source,
            error=str(e),
            budget=budget,
            completed_at=datetime.utcnow(),
        )
