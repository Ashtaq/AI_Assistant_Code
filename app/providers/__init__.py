"""Build the fast and top tier providers from settings."""

from __future__ import annotations

import logging
import time

import httpx

from ..config import Settings
from .anthropic import AnthropicProvider
from .base import Completion, Provider, ProviderError
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

__all__ = ["Completion", "Provider", "ProviderError", "build_providers", "warm_up_ollama"]


def build_providers(settings: Settings, client: httpx.AsyncClient) -> tuple[Provider | None, Provider | None]:
    fast: Provider | None = None
    if settings.fast_provider == "ollama":
        fast = OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            client=client,
            timeout=settings.ollama_timeout_seconds,
            think=settings.ollama_think,
            keep_alive=settings.ollama_keep_alive,
        )
    elif settings.fast_provider == "hosted":
        fast = OpenAICompatProvider(
            name=settings.hosted_name,
            base_url=settings.hosted_base_url or "",
            model=settings.hosted_model or "",
            api_key=settings.hosted_api_key,
            client=client,
            timeout=settings.timeout_seconds,
        )

    top: Provider | None = None
    if settings.top_provider == "anthropic":
        top = AnthropicProvider(api_key=settings.top_api_key or "", model=settings.top_model, client=client, timeout=settings.timeout_seconds)
    elif settings.top_provider == "openai":
        top = OpenAICompatProvider(
            name="openai",
            base_url=settings.top_base_url,
            model=settings.top_model,
            api_key=settings.top_api_key,
            client=client,
            timeout=settings.timeout_seconds,
        )
    return fast, top


async def warm_up_ollama(settings: Settings, client: httpx.AsyncClient) -> None:
    """Load the Ollama model into memory at startup so the first user question is not the slow one."""
    log = logging.getLogger("finpro.ai")
    base = settings.ollama_base_url.rstrip("/").removesuffix("/v1")
    started = time.perf_counter()
    try:
        res = await client.post(f"{base}/api/generate", json={"model": settings.ollama_model, "keep_alive": settings.ollama_keep_alive}, timeout=300)
    except httpx.HTTPError as e:
        log.warning("Ollama warm-up failed", extra={"error": f"{type(e).__name__} — is Ollama running? (`ollama serve`)", "model": settings.ollama_model})
        return
    if res.status_code == 404:
        log.warning("Ollama model not found", extra={"model": settings.ollama_model, "fix": f"ollama pull {settings.ollama_model}"})
    elif res.status_code >= 400:
        log.warning("Ollama warm-up failed", extra={"status": res.status_code, "model": settings.ollama_model})
    else:
        log.info("Ollama model loaded", extra={"model": settings.ollama_model, "seconds": round(time.perf_counter() - started, 1)})
