"""Any OpenAI-compatible chat API: Ollama, vLLM, Groq, Together, OpenRouter, DeepInfra, OpenAI."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

import httpx

from .base import Completion, ProviderError, StreamChunk, http_error, with_retry

# Reasoning models (Qwen3, DeepSeek-R1 …) may include their scratchpad in the answer.
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        local: bool = False,
        client: httpx.AsyncClient,
        timeout: float = 90,
        retry_backoff: float = 1.0,
        retry_timeouts: bool = True,
        no_think: bool = False,
    ):
        self.name = name
        self.model = model
        self.local = local
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"content-type": "application/json", **({"authorization": f"Bearer {api_key}"} if api_key else {})}
        self._client = client
        self._timeout = timeout
        self._backoff = retry_backoff
        self._retry_timeouts = retry_timeouts
        # Qwen3 "thinks" at length before answering, which is very slow on a laptop. `/no_think` is its
        # documented switch to answer directly.
        self._no_think = no_think and model.lower().startswith("qwen3")

    def _body(self, system: str, messages: list[dict], max_tokens: int, temperature: float, stream: bool) -> dict:
        if self._no_think and messages and messages[-1]["role"] == "user":
            messages = [*messages[:-1], {"role": "user", "content": messages[-1]["content"] + "\n\n/no_think"}]
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    def _explain(self, e: ProviderError) -> ProviderError:
        """Add a fix-it hint for the common local Ollama problems."""
        if self.name != "ollama":
            return e
        if e.kind == "timeout":
            e.detail = (
                f"ollama: no output for {self._timeout:.0f}s — Ollama is running but '{self.model}' is too slow on this machine "
                "(the first request also loads the model). Try a smaller model: `ollama pull qwen3:4b` and set OLLAMA_MODEL=qwen3:4b"
            )
        elif e.kind == "connect":
            e.detail += f" — is Ollama running? Start it with `ollama serve` and pull the model with `ollama pull {self.model}`"
        elif e.status == 404:
            e.detail += f" — model not found, run `ollama pull {self.model}`"
        return e

    async def complete(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        body = self._body(system, messages, max_tokens, temperature, stream=False)
        try:
            res = await with_retry(
                lambda: self._client.post(self._url, json=body, headers=self._headers, timeout=self._timeout),
                label=self.name,
                backoff=self._backoff,
                retry_timeouts=self._retry_timeouts,
            )
        except ProviderError as e:
            raise self._explain(e) from None
        data = res.json()
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name}: unexpected response shape") from exc
        usage = data.get("usage") or {}
        return Completion(
            text=_THINK.sub("", text).strip(),
            model=data.get("model") or self.model,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def stream(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[StreamChunk]:
        """Stream the answer. The read timeout is the longest allowed silence, not a cap on total length."""
        body = self._body(system, messages, max_tokens, temperature, stream=True)
        timeout = httpx.Timeout(self._timeout, connect=10)
        started = False
        for attempt in range(2):
            try:
                async with self._client.stream("POST", self._url, json=body, headers=self._headers, timeout=timeout) as res:
                    if res.status_code >= 400:
                        err = http_error(self.name, res.status_code, await res.aread())
                        if err.retryable and attempt == 0:
                            await asyncio.sleep(self._backoff)
                            continue
                        raise self._explain(err)
                    async for line in res.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            data = json.loads(payload)
                        except ValueError:
                            continue
                        if data.get("error"):
                            err = data["error"]
                            raise ProviderError(f"{self.name}: {err.get('message', err) if isinstance(err, dict) else err}")
                        choices = data.get("choices") or []
                        text = (choices[0].get("delta") or {}).get("content") if choices else None
                        usage = data.get("usage") or {}
                        if text:
                            started = True
                        yield StreamChunk(
                            text=text, model=data.get("model"), input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens")
                        )
                    return
            except httpx.TimeoutException:
                raise self._explain(ProviderError(f"{self.name}: request timed out", kind="timeout")) from None
            except httpx.TransportError as e:
                if attempt == 0 and not started:
                    await asyncio.sleep(self._backoff)
                    continue
                raise self._explain(ProviderError(f"{self.name}: cannot connect ({type(e).__name__})", retryable=True, kind="connect")) from None
