"""Service configuration, read from environment variables (and the repo's root .env in development).

Two model tiers:
  fast — answers most questions cheaply. Ollama (local, free) in development; a hosted
         open-source model (Groq, Together, OpenRouter, DeepInfra, vLLM …) in production.
  top  — a frontier model (Claude / any OpenAI-compatible API) used only for complex questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parent

# Used only outside production when AI_SERVICE_TOKEN is not set. The Node API uses the same default.
DEV_SERVICE_TOKEN = "finpro-dev-ai-token"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=(REPO_ROOT / ".env", MODULE_DIR / ".env"),
        extra="ignore",
        populate_by_name=True,
    )

    env: str = Field("development", validation_alias=AliasChoices("AI_ENV", "NODE_ENV"))
    log_level: str = "INFO"

    # --- HTTP service ---------------------------------------------------------------
    service_host: str = "127.0.0.1"
    service_port: int = 8090
    service_token: str | None = None

    # --- Fast tier ------------------------------------------------------------------
    # ollama | hosted | none. Default: ollama in development, hosted in production.
    fast_provider: Literal["ollama", "hosted", "none"] | None = None
    ollama_base_url: str = Field("http://127.0.0.1:11434/v1", validation_alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("qwen3:8b", validation_alias="OLLAMA_MODEL")
    # Longest silence allowed from Ollama (loading the model counts). Streaming has no cap on total length.
    ollama_timeout_seconds: float = Field(180, gt=0, le=900)
    ollama_think: bool = False  # Qwen3 step-by-step "thinking": better on hard questions, much slower on a laptop
    ollama_keep_alive: str = "30m"  # how long Ollama keeps the model in memory after the warm-up
    hosted_name: str = "hosted"
    hosted_base_url: str | None = None
    hosted_api_key: str | None = None
    hosted_model: str | None = None

    # --- Top tier -------------------------------------------------------------------
    top_provider: Literal["anthropic", "openai", "none"] = "none"
    top_api_key: str | None = None
    top_model: str = "claude-opus-5-5"
    top_base_url: str = "https://api.openai.com/v1"  # only for top_provider=openai (any OpenAI-compatible API)

    # --- Routing & cost control -----------------------------------------------------
    routing_mode: Literal["auto", "fast", "top"] = "auto"
    complex_threshold: int = Field(4, ge=1, le=20)
    top_daily_limit_per_org: int = Field(200, ge=0)  # 0 = unlimited
    redact_pii: bool = True
    cache_ttl_seconds: int = Field(600, ge=0)
    cache_max_entries: int = Field(500, ge=0)

    # --- Generation -----------------------------------------------------------------
    temperature: float = Field(0.2, ge=0, le=1)
    max_output_tokens: int = Field(1500, ge=64, le=16000)
    top_max_output_tokens: int = Field(3000, ge=64, le=32000)
    timeout_seconds: float = Field(90, gt=0, le=600)
    max_context_chars: int = Field(120_000, ge=4_000)

    @property
    def is_prod(self) -> bool:
        return self.env == "production"

    @property
    def token(self) -> str:
        return self.service_token or DEV_SERVICE_TOKEN

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.fast_provider is None:
            self.fast_provider = "hosted" if self.is_prod else "ollama"
        problems: list[str] = []
        if self.is_prod and (not self.service_token or len(self.service_token) < 32):
            problems.append("AI_SERVICE_TOKEN (32+ characters) is required in production")
        if self.fast_provider == "hosted" and not (self.hosted_base_url and self.hosted_api_key and self.hosted_model):
            problems.append("AI_HOSTED_BASE_URL, AI_HOSTED_API_KEY and AI_HOSTED_MODEL are required when the fast tier is 'hosted'")
        if self.top_provider != "none" and not self.top_api_key:
            problems.append(f"AI_TOP_API_KEY is required when AI_TOP_PROVIDER={self.top_provider}")
        if self.fast_provider == "none" and self.top_provider == "none":
            problems.append("configure at least one model: AI_FAST_PROVIDER or AI_TOP_PROVIDER")
        if problems:
            raise ValueError("Invalid AI service configuration: " + "; ".join(problems))
        return self
