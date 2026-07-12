"""
Application configuration.
Loads from environment variables with sensible defaults.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- App ----
    app_name: str = "Crossfire"
    app_version: str = "0.3.0"
    debug: bool = False

    # ---- Fireworks AI ----
    fireworks_api_key: str = Field(default="", description="Fireworks AI API key")
    fireworks_model: str = "accounts/fireworks/models/gemma2-27b-it"
    fireworks_max_tokens: int = 4096
    fireworks_temperature: float = 0.2

    # ---- vLLM (Local Model) ----
    vllm_url: str = "http://localhost:8001"
    vllm_model: str = "unsloth/gemma-4-12b-it"
    vllm_max_tokens: int = 4096
    vllm_temperature: float = 0.2

    # ---- AMD GPU ----
    amd_gpu_enabled: bool = True
    pytorch_rocm_arch: str = "gfx942"
    hsa_override_gfx_version: str = ""  # MI300X is gfx942 — no override needed

    # ---- RAG ----
    chroma_persist_dir: str = "./chroma_db"
    chroma_host: str = ""
    chroma_port: int = 8000
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rag_top_k: int = 5

    # ---- Sandbox ----
    sandbox_container: str = "crossfire-sandbox"
    sandbox_timeout: int = 60
    sandbox_memory_gb: int = 8

    # ---- Agent ----
    agent_max_iterations: int = 5
    agent_routing_threshold: float = 0.4
    agent_diff_threshold: float = 1e-5

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    # ---- Redis ----
    redis_url: str = "redis://localhost:6379/0"

    # ---- Demo Auth ----
    demo_basic_auth_user: str = ""
    demo_basic_auth_pass: str = ""

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def chroma_available(self) -> bool:
        return bool(self.chroma_host)

    @field_validator("agent_routing_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("agent_routing_threshold must be in [0, 1]")
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


# Convenience export
settings = get_settings()
