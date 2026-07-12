"""
LangGraph state definition with type-safe required fields.

Uses TypedDict inheritance to enforce required keys while keeping
optional keys flexible. Nodes access required keys with [] (safe)
and optional keys with .get() (returns None if missing).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from app.schemas import AnalysisResult, MigrationStrategy, ModelChoice


class RequiredAgentState(TypedDict):
    """Required keys — must be provided when creating initial state."""
    cuda_source: str
    filename: str
    job_id: str
    max_iterations: int


class AgentState(RequiredAgentState, total=False):
    """Full agent state. Required keys inherited, optional keys below.

    Optional keys are populated as the agent progresses through the graph.
    Nodes should use state.get("key") for optional keys.
    """
    # Input options
    force_remote: bool

    # Analysis
    analysis: AnalysisResult
    rag_context: str

    # Translation
    translated_code: Optional[str]
    iteration: int
    model_used: ModelChoice
    error_feedback: Optional[str]

    # Per-iteration metrics (consumed by diff_node)
    last_iter_tokens: int
    last_iter_cost: float
    last_iter_latency: int

    # Validation
    compile_result: Optional[Dict[str, Any]]
    run_result: Optional[Dict[str, Any]]
    diff_result: Optional[Dict[str, Any]]
    gpu_metrics: Optional[Dict[str, Any]]
    gpu_attestation: Optional[Dict[str, Any]]

    # Bookkeeping
    history: List[Dict[str, Any]]
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: int
    stub_mode: bool
    migration_strategy: MigrationStrategy

    # Output
    status: str
    error: Optional[str]


def initial_state(
    job_id: str,
    cuda_source: str,
    filename: str = "input.cu",
    max_iterations: int = 5,
    force_remote: bool = False,
) -> AgentState:
    """Create initial state for a new translation job.

    All required keys are set. Optional keys default to safe values.
    """
    return AgentState(
        # Required
        cuda_source=cuda_source,
        filename=filename,
        job_id=job_id,
        max_iterations=max_iterations,
        # Optional — explicitly set safe defaults
        force_remote=force_remote,
        translated_code=None,
        iteration=0,
        model_used=ModelChoice.LOCAL,
        error_feedback=None,
        compile_result=None,
        run_result=None,
        diff_result=None,
        gpu_metrics=None,
        gpu_attestation=None,
        history=[],
        total_tokens=0,
        total_cost_usd=0.0,
        total_latency_ms=0,
        last_iter_tokens=0,
        last_iter_cost=0.0,
        last_iter_latency=0,
        stub_mode=False,
        migration_strategy=MigrationStrategy.LLM_ONLY,
        status="in_progress",
        error=None,
    )
