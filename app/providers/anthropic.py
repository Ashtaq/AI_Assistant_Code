"""Anthropic Messages API (Claude). The system prompt is marked cacheable to cut repeat-input cost."""

from __future__ import annotations

import httpx

from .base import Completion, ProviderError, with_retry

API_URL = "https://api.anthropic.com/v1/messages"


def _alternate(messages: list[dict]) -> list[dict]:
    """Claude requires strictly alternating turns that start with the user."""
    out: list[dict] = []
    for m in messages:
        if not out and m["role"] != "user":
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


class AnthropicProvider:
    local = False

    def __init__(self, *, api_key: str, model: str, client: httpx.AsyncClient, timeout: float = 90, retry_backoff: float = 1.0):
        self.name = "anthropic"
        self.model = model
        self._headers = {"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
        self._client = client
        self._timeout = timeout
        self._backoff = retry_backoff

    async def complete(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": _alternate(messages),
        }
        res = await with_retry(
            lambda: self._client.post(API_URL, json=body, headers=self._headers, timeout=self._timeout),
            label=self.name,
            backoff=self._backoff,
        )
        data = res.json()
        blocks = data.get("content") or []
        if not isinstance(blocks, list):
            raise ProviderError("anthropic: unexpected response shape")
        text = "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text").strip()
        usage = data.get("usage") or {}
        return Completion(text=text, model=data.get("model") or self.model, input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
