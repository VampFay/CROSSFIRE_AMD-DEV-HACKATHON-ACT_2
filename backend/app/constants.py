"""
Application constants — centralized magic numbers with documentation.

All tunable values live here. Environment-specific overrides go in config.py.
"""
from __future__ import annotations


# ============================================================
# Rate Limiting
# ============================================================
RATE_LIMIT_REQUESTS_PER_MINUTE = 10
RATE_LIMIT_BURST = 20

# ============================================================
# Sandbox
# ============================================================
SANDBOX_CHECK_TIMEOUT_SECONDS = 2
SANDBOX_AVAILABILITY_CACHE_SECONDS = 5
SANDBOX_COMPILE_TIMEOUT_SECONDS = 60
SANDBOX_RUN_TIMEOUT_SECONDS = 60
SANDBOX_MEMORY_LIMIT_GB = 8

# ============================================================
# Agent
# ============================================================
DEFAULT_MAX_ITERATIONS = 5
DIFF_THRESHOLD = 1e-5
ROUTING_THRESHOLD = 0.4

# ============================================================
# Job Store
# ============================================================
JOB_TTL_SECONDS = 86400  # 24 hours
JOB_MAX_LIST_LIMIT = 50

# ============================================================
# Fine-Tuning
# ============================================================
MAX_SEQ_LEN = 4096
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_LEARNING_RATE = 2e-4
LORA_EPOCHS = 3
LORA_BATCH_SIZE = 4
LORA_GRAD_ACCUM = 4
EVAL_SPLIT_RATIO = 0.1

# ============================================================
# RAG
# ============================================================
RAG_TOP_K = 5
RAG_MAX_INIT_RETRIES = 3

# ============================================================
# Model Clients
# ============================================================
MODEL_RETRY_MAX_ATTEMPTS = 3
MODEL_RETRY_MIN_WAIT = 2
MODEL_RETRY_MAX_WAIT = 10
MODEL_MAX_TOKENS = 4096
MODEL_TEMPERATURE = 0.2
HTTP_MAX_CONNECTIONS = 100
HTTP_MAX_KEEPALIVE = 20

# ============================================================
# GPU Metrics
# ============================================================
GPU_METRICS_TIMEOUT_SECONDS = 5

# ============================================================
# Translation Memory
# ============================================================
MEMORY_DB_PATH = "data/translation_memory.db"

# ============================================================
# Checkpointing
# ============================================================
CHECKPOINT_DB_PATH = "data/agent_checkpoints.db"

# ============================================================
# Docker Resource Limits (for docker-compose.yml reference)
# ============================================================
DOCKER_LIMITS = {
    "sandbox": {"memory": "16G", "cpus": "4"},
    "vllm": {"memory": "64G", "cpus": "8"},
    "api": {"memory": "2G", "cpus": "2"},
    "frontend": {"memory": "512M", "cpus": "1"},
    "redis": {"memory": "512M", "cpus": "0.5"},
    "chromadb": {"memory": "1G", "cpus": "1"},
    "worker": {"memory": "4G", "cpus": "2"},
}

# ============================================================
# Graceful Shutdown
# ============================================================
SHUTDOWN_GRACE_PERIOD_SECONDS = 30
