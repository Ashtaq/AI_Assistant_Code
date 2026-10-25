"""Any OpenAI-compatible chat API: Ollama, vLLM, Groq, Together, OpenRouter, DeepInfra, OpenAI."""

from __future__ import annotations

import re

import httpx

from .base import Completion, ProviderError, with_retry

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
    ):
        self.name = name
        self.model = model
        self.local = local
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"content-type": "application/json", **({"authorization": f"Bearer {api_key}"} if api_key else {})}
        self._client = client
        self._timeout = timeout
        self._backoff = retry_backoff

    async def complete(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> Completion:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        try:
            res = await with_retry(
                lambda: self._client.post(self._url, json=body, headers=self._headers, timeout=self._timeout),
                label=self.name,
                backoff=self._backoff,
            )
        except ProviderError as e:
            if self.name == "ollama" and e.status is None:
                e.detail += f" — is Ollama running? Start it with `ollama serve` and pull the model with `ollama pull {self.model}`"
            elif self.name == "ollama" and e.status == 404:
                e.detail += f" — model not found, run `ollama pull {self.model}`"
            raise
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
