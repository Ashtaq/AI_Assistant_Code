"""Anthropic Messages API (Claude). The system prompt is marked cacheable to cut repeat-input cost."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from .base import Completion, ProviderError, StreamChunk, http_error, with_retry

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

    def _body(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": _alternate(messages),
        }

    async def complete(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        body = self._body(system, messages, max_tokens, temperature)
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

    async def stream(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[StreamChunk]:
        body = {**self._body(system, messages, max_tokens, temperature), "stream": True}
        timeout = httpx.Timeout(self._timeout, connect=10)
        started = False
        for attempt in range(2):
            try:
                async with self._client.stream("POST", API_URL, json=body, headers=self._headers, timeout=timeout) as res:
                    if res.status_code >= 400:
                        err = http_error(self.name, res.status_code, await res.aread())
                        if err.retryable and attempt == 0:
                            await asyncio.sleep(self._backoff)
                            continue
                        raise err
                    async for line in res.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            data = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        kind = data.get("type")
                        if kind == "message_start":
                            msg = data.get("message") or {}
                            yield StreamChunk(model=msg.get("model"), input_tokens=(msg.get("usage") or {}).get("input_tokens"))
                        elif kind == "content_block_delta":
                            delta = data.get("delta") or {}
                            if delta.get("type") == "text_delta" and delta.get("text"):
                                started = True
                                yield StreamChunk(text=delta["text"])
                        elif kind == "message_delta":
                            yield StreamChunk(output_tokens=(data.get("usage") or {}).get("output_tokens"))
                        elif kind == "error":
                            raise ProviderError(f"anthropic: {(data.get('error') or {}).get('message', 'stream error')}", retryable=True)
                        elif kind == "message_stop":
                            break
                    return
            except httpx.TimeoutException:
                raise ProviderError("anthropic: request timed out", kind="timeout") from None
            except httpx.TransportError as e:
                if attempt == 0 and not started:
                    await asyncio.sleep(self._backoff)
                    continue
                raise ProviderError(f"anthropic: cannot connect ({type(e).__name__})", retryable=True, kind="connect") from None
