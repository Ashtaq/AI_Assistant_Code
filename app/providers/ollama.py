"""Ollama's native chat API (/api/chat).

Used instead of Ollama's OpenAI-compatible endpoint because only the native API reliably switches off the
"thinking" phase of reasoning models (`think: false`) and lets us set keep_alive. With thinking on, Qwen3
can reason for minutes on a laptop before writing a single word of the answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

import httpx

from .base import Completion, ProviderError, StreamChunk

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class OllamaProvider:
    local = True

    def __init__(self, *, base_url: str, model: str, client: httpx.AsyncClient, timeout: float = 180, think: bool = False, keep_alive: str = "30m"):
        self.name = "ollama"
        self.model = model
        self._url = base_url.rstrip("/").removesuffix("/v1") + "/api/chat"
        self._client = client
        self._timeout = timeout
        self._think = think
        self._keep_alive = keep_alive
        # Older Ollama versions ignore `think`; Qwen3 also honours this in-prompt switch.
        self._no_think_hint = not think and model.lower().startswith("qwen3")

    def _body(self, system: str, messages: list[dict], max_tokens: int, temperature: float, stream: bool) -> dict:
        if self._no_think_hint and messages and messages[-1]["role"] == "user":
            messages = [*messages[:-1], {"role": "user", "content": messages[-1]["content"] + "\n\n/no_think"}]
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            "think": self._think,
            "keep_alive": self._keep_alive,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

    def _error(self, detail: str, *, status: int | None = None, kind: str = "http") -> ProviderError:
        if kind == "timeout":
            detail = (
                f"ollama: no output for {self._timeout:.0f}s — Ollama is running but '{self.model}' is too slow on this machine "
                "(the first request also loads the model). Try a smaller model, e.g. `ollama pull qwen3:1.7b` and set OLLAMA_MODEL=qwen3:1.7b"
            )
        elif kind == "connect":
            detail += f" — is Ollama running? Start it with `ollama serve` (or open the Ollama app) and run `ollama pull {self.model}`"
        elif status == 404:
            detail += f" — model not found, run `ollama pull {self.model}`"
        return ProviderError(detail, status=status, kind=kind, retryable=kind == "connect")

    @staticmethod
    def _message(body: bytes) -> str:
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            return ""
        return str(data.get("error", ""))[:300] if isinstance(data, dict) else ""

    async def complete(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        try:
            res = await self._client.post(self._url, json=self._body(system, messages, max_tokens, temperature, False), timeout=self._timeout)
        except httpx.TimeoutException:
            raise self._error("ollama: request timed out", kind="timeout") from None
        except httpx.TransportError as e:
            raise self._error(f"ollama: cannot connect ({type(e).__name__})", kind="connect") from None
        if res.status_code >= 400:
            raise self._error(f"ollama: HTTP {res.status_code} {self._message(res.content)}".rstrip(), status=res.status_code)
        data = res.json()
        text = (data.get("message") or {}).get("content") or ""
        return Completion(
            text=_THINK.sub("", text).strip(),
            model=data.get("model") or self.model,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
        )

    async def stream(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[StreamChunk]:
        body = self._body(system, messages, max_tokens, temperature, True)
        try:
            async with self._client.stream("POST", self._url, json=body, timeout=httpx.Timeout(self._timeout, connect=10)) as res:
                if res.status_code >= 400:
                    raise self._error(f"ollama: HTTP {res.status_code} {self._message(await res.aread())}".rstrip(), status=res.status_code)
                async for line in res.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    if data.get("error"):
                        raise ProviderError(f"ollama: {data['error']}")
                    msg = data.get("message") or {}
                    text = msg.get("content") or None
                    thinking = bool(msg.get("thinking"))
                    if data.get("done"):
                        yield StreamChunk(text=text, model=data.get("model"), input_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"))
                        return
                    if text or thinking:
                        yield StreamChunk(text=text, thinking=thinking and not text)
        except httpx.TimeoutException:
            raise self._error("ollama: request timed out", kind="timeout") from None
        except httpx.TransportError as e:
            raise self._error(f"ollama: cannot connect ({type(e).__name__})", kind="connect") from None
