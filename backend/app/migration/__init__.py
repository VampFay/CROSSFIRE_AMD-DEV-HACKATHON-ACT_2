"""Migration engine — repository-level CUDA→ROCm migration."""
from app.migration.hipify_adapter import hipify_file, hipify_available, regex_hipify
from app.migration.planner import (
    plan_migration,
    inventory_repository,
    generate_patch_bundle,
    generate_unified_diff,
    SUPPORTED_LIBRARIES,
    BLOCKED_LIBRARIES,
    SUPPORTED_PATTERNS,
    BLOCKED_PATTERNS,
)

__all__ = [
    "hipify_file",
    "hipify_available",
    "regex_hipify",
    "plan_migration",
    "inventory_repository",
    "generate_patch_bundle",
    "generate_unified_diff",
    "SUPPORTED_LIBRARIES",
    "BLOCKED_LIBRARIES",
    "SUPPORTED_PATTERNS",
    "BLOCKED_PATTERNS",
]
