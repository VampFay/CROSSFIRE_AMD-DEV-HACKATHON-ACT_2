"""
Pydantic schemas for API request/response models.

Core design principle (Phase 0 of the rebuild):
Every result carries a VerificationLevel that maps to actual evidence.
No stub or cached result can masquerade as a GPU-validated result.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ============================================================
# Enums
# ============================================================

class JobStatus(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    TRANSLATING = "translating"
    COMPILING = "compiling"
    RUNNING = "running"
    DIFFING = "diffing"
    DEBUGGING = "debugging"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"  # unsupported input, cannot proceed


class VerificationLevel(str, Enum):
    """
    What evidence backs this result? Monotonically increasing strength.

    The UI MUST NOT show "Validated" unless the level is at least
    DIFFERENTIALLY_VERIFIED. The API always returns the exact level.
    """
    ANALYZED = "analyzed"                        # static analysis only
    TRANSLATED = "translated"                    # code produced, not compiled
    COMPILED = "compiled"                        # hipcc success, not run
    EXECUTED = "executed"                        # ran on AMD GPU, no baseline
    TEST_VERIFIED = "test_verified"              # ran + matched analytical baseline
    DIFFERENTIALLY_VERIFIED = "differentially_verified"  # ran + matched live CUDA reference
    BENCHMARKED = "benchmarked"                  # differentially verified + perf measured
    BLOCKED = "blocked"                          # unsupported input


class ModelChoice(str, Enum):
    LOCAL = "local"      # Gemma 4 12B via vLLM on MI300X
    REMOTE = "remote"    # Gemma 27B via Fireworks


class TranslationPattern(str, Enum):
    """CUDA patterns identified by the static analyzer."""
    KERNEL = "kernel"
    CUBLAS = "cuBLAS"
    CUDNN = "cuDNN"
    THRUST = "Thrust"
    TRITON = "Triton"
    WARP_SHUFFLE = "warp_shuffle"
    SHARED_MEMORY = "shared_memory"
    PYTORCH_EXTENSION = "pytorch_extension"
    STREAMS = "streams"
    EVENTS = "events"


class MigrationStrategy(str, Enum):
    """How a file was migrated."""
    MECHANICAL = "mechanical"        # hipify-clang succeeded, no LLM needed
    AGENT_REPAIRED = "agent_repaired"  # hipify + LLM repair loop
    LLM_ONLY = "llm_only"            # no hipify (e.g., .cu without CUDA headers)
    BLOCKED = "blocked"              # unsupported construct


# ============================================================
# Request Models
# ============================================================

# Filename whitelist — prevents command injection and path traversal.
# Matches the safest subset: letters, digits, underscore, hyphen, dot.
_SAFE_FILENAME_RE = r"^[A-Za-z0-9_\-\.]+$"


class TranslateRequest(BaseModel):
    """Request to translate a single CUDA file."""
    cuda_source: str = Field(..., description="CUDA source code", min_length=1)
    filename: str = Field(default="input.cu", description="Original filename")
    force_remote: bool = Field(default=False, description="Force remote model usage")
    max_iterations: Optional[int] = Field(default=None, ge=1, le=10)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        import re
        if not re.match(_SAFE_FILENAME_RE, v):
            raise ValueError(
                "filename must contain only letters, digits, underscores, hyphens, and dots"
            )
        # Reject path traversal attempts explicitly
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("filename must not contain path separators or traversal sequences")
        return v


class TranslateRepoRequest(BaseModel):
    """Request to translate an entire GitHub repo."""
    repo_url: str = Field(..., description="GitHub repo URL or user/repo")
    file_pattern: str = Field(default="**/*.cu", description="Glob pattern for CUDA files")


# ============================================================
# Job Budget (Phase 4 — bounded agent loop)
# ============================================================

class JobBudget(BaseModel):
    """
    Hard limits enforced before every node. The agent loop CANNOT exceed these.

    Default values are calibrated for single-file translation on MI300X.
    Repository migrations should override with higher limits.
    """
    max_attempts: int = Field(default=5, ge=1, le=20,
                              description="Max translation attempts (iterations)")
    max_model_calls: int = Field(default=12, ge=1, le=50,
                                 description="Max LLM API calls across all iterations")
    max_wall_time_seconds: int = Field(default=1800, ge=30, le=7200,
                                       description="Max wall-clock time per job")
    max_generated_tokens: int = Field(default=50_000, ge=1000, le=500_000,
                                      description="Max total tokens generated by LLM")
    max_compile_seconds: int = Field(default=300, ge=10, le=1800,
                                     description="Max single compile timeout")
    max_run_seconds: int = Field(default=120, ge=5, le=600,
                                 description="Max single binary execution timeout")
    max_cost_usd: Decimal = Field(default=Decimal("5.00"), ge=Decimal("0"),
                                  description="Max USD spend on remote model calls")

    def assert_available(self, action: str, usage: Dict[str, Any] | None = None) -> None:
        """
        Raise BudgetExceeded if the requested action would exceed limits.
        Call BEFORE every node.
        """
        from app.exceptions import BudgetExceeded
        if usage is None:
            usage = {}
        if usage.get("attempts", 0) >= self.max_attempts:
            raise BudgetExceeded(f"max_attempts ({self.max_attempts}) reached")
        if usage.get("model_calls", 0) >= self.max_model_calls:
            raise BudgetExceeded(f"max_model_calls ({self.max_model_calls}) reached")
        if usage.get("tokens", 0) >= self.max_generated_tokens:
            raise BudgetExceeded(f"max_generated_tokens ({self.max_generated_tokens}) reached")
        if Decimal(str(usage.get("cost_usd", "0"))) >= self.max_cost_usd:
            raise BudgetExceeded(f"max_cost_usd ({self.max_cost_usd}) reached")
        if usage.get("wall_time_seconds", 0) >= self.max_wall_time_seconds:
            raise BudgetExceeded(f"max_wall_time_seconds ({self.max_wall_time_seconds}) reached")


# ============================================================
# Response Models
# ============================================================

class AnalysisResult(BaseModel):
    """Output of the static analyzer."""
    patterns: List[TranslationPattern] = Field(default_factory=list)
    difficulty_score: float = Field(..., ge=0.0, le=1.0)
    kernel_count: int = 0
    has_shared_memory: bool = False
    has_warp_primitives: bool = False
    library_calls: List[str] = Field(default_factory=list)
    file_dependencies: List[str] = Field(default_factory=list)
    notes: str = ""


class CompileResult(BaseModel):
    success: bool
    errors: Optional[str] = None
    warnings: Optional[str] = None
    binary_path: Optional[str] = None
    compile_time_ms: Optional[int] = None
    compiler_version: Optional[str] = None  # hipcc version string
    compiler_flags: List[str] = Field(default_factory=list)


class RunResult(BaseModel):
    success: bool
    outputs: Optional[Dict[str, Any]] = None
    stderr: Optional[str] = None
    runtime_ms: Optional[int] = None
    gpu_memory_used_mb: Optional[int] = None
    exit_code: Optional[int] = None


class DiffResult(BaseModel):
    success: bool
    max_abs_error: Optional[float] = None
    mse: Optional[float] = None
    threshold: float = 1e-5
    mismatched_keys: List[str] = Field(default_factory=list)
    baseline_source: str = "analytical"  # "analytical" | "cuda_live" | "signed_artifact"


class IterationRecord(BaseModel):
    """One iteration of the agent loop."""
    iteration: int
    model_used: ModelChoice
    compile_success: bool
    run_success: bool
    diff_success: bool
    tokens_used: int
    cost_usd: float
    latency_ms: int
    error_feedback: Optional[str] = None
    strategy: MigrationStrategy = MigrationStrategy.LLM_ONLY


class GPUAttestation(BaseModel):
    """
    Cryptographic-style evidence that the run happened on real AMD hardware.
    Captured at compile + run time, stored with every result.
    """
    gpu_model: str = ""
    architecture: str = ""
    rocm_version: str = ""
    hipcc_version: str = ""
    driver_version: str = ""
    compiler_flags: List[str] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=datetime.utcnow)
    stub_mode: bool = False  # True = no real GPU, demo mode only


class GPUMetrics(BaseModel):
    """GPU metrics captured during translation run."""
    gpu_name: str = ""
    gpu_arch: str = ""
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    gpu_utilization_pct: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    captured_at: str = ""
    stub_mode: bool = False


class BenchmarkResult(BaseModel):
    """Real benchmark numbers — no Math.random."""
    cuda_median_ms: Optional[float] = None
    cuda_p90_ms: Optional[float] = None
    cuda_p95_ms: Optional[float] = None
    rocm_median_ms: Optional[float] = None
    rocm_p90_ms: Optional[float] = None
    rocm_p95_ms: Optional[float] = None
    relative_performance: Optional[float] = None  # cuda_median / rocm_median
    repetitions: int = 0
    warmup_runs: int = 0
    notes: str = ""


class MigrationPlan(BaseModel):
    """Plan returned before any code is modified."""
    total_files: int = 0
    cuda_translation_units: int = 0
    libraries: List[str] = Field(default_factory=list)
    risk_items: List[Dict[str, Any]] = Field(default_factory=list)
    strategy_counts: Dict[str, int] = Field(default_factory=dict)
    supported: bool = True
    block_reason: Optional[str] = None


class TranslationResult(BaseModel):
    """Final result of a translation job."""
    job_id: str
    status: JobStatus
    verification_level: VerificationLevel = VerificationLevel.ANALYZED
    cuda_source: str
    translated_code: Optional[str] = None
    iterations: List[IterationRecord] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    analysis: Optional[AnalysisResult] = None
    diff: Optional[DiffResult] = None
    gpu_metrics: Optional[GPUMetrics] = None
    gpu_attestation: Optional[GPUAttestation] = None
    benchmark: Optional[BenchmarkResult] = None
    migration_plan: Optional[MigrationPlan] = None
    migration_strategy: MigrationStrategy = MigrationStrategy.LLM_ONLY
    cache_hit: bool = False
    stub_mode: bool = False  # True if any part used stub fallback
    budget: Optional[JobBudget] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class JobStatusResponse(BaseModel):
    """Polled job status."""
    job_id: str
    status: JobStatus
    verification_level: VerificationLevel = VerificationLevel.ANALYZED
    progress: float = Field(..., ge=0.0, le=1.0, description="0-1 progress indicator")
    current_state: str
    iteration: int = 0
    message: Optional[str] = None
    stub_mode: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str
    amd_gpu_available: bool
    vllm_available: bool
    fireworks_configured: bool
    chroma_available: bool
    sandbox_available: bool
    hipify_available: bool
    uptime_seconds: float


# ============================================================
# WebSocket Messages
# ============================================================

class WSMessage(BaseModel):
    """WebSocket message envelope."""
    type: str  # "status", "log", "result", "error"
    job_id: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
