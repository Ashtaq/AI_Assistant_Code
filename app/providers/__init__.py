"""Build the fast and top tier providers from settings."""

from __future__ import annotations

import httpx

from ..config import Settings
from .anthropic import AnthropicProvider
from .base import Completion, Provider, ProviderError
from .openai_compat import OpenAICompatProvider

__all__ = ["Completion", "Provider", "ProviderError", "build_providers"]


def build_providers(settings: Settings, client: httpx.AsyncClient) -> tuple[Provider | None, Provider | None]:
    fast: Provider | None = None
    if settings.fast_provider == "ollama":
        fast = OpenAICompatProvider(
            name="ollama", base_url=settings.ollama_base_url, model=settings.ollama_model, local=True, client=client, timeout=settings.timeout_seconds
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
