"""Common provider interface, retry policy and error type."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class StreamChunk:
    """One piece of a streamed answer. `text` is new content; usage/model arrive when the provider sends them."""

    text: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking: bool = False  # the model is reasoning (no answer text yet)


class ProviderError(Exception):
    """A model call failed. `detail` is safe to log; it never contains prompt content."""

    def __init__(self, detail: str, *, status: int | None = None, retryable: bool = False, kind: str = "http"):
        super().__init__(detail)
        self.detail = detail
        self.status = status
        self.retryable = retryable
        self.kind = kind  # http | timeout | connect


class Provider(Protocol):
    name: str  # e.g. "ollama", "groq", "anthropic"
    model: str
    local: bool  # True when data never leaves our infrastructure (no PII redaction needed)

    async def complete(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> Completion: ...

    def stream(self, system: str, messages: list[dict], max_tokens: int, temperature: float) -> AsyncIterator[StreamChunk]: ...


def http_error(label: str, status: int, body: bytes) -> ProviderError:
    """Build a ProviderError from a failed HTTP response body."""
    message = ""
    try:
        import json

        data = json.loads(body or b"{}")
        err = data.get("error") if isinstance(data, dict) else None
        message = str(err.get("message", "")) if isinstance(err, dict) else str(err or "")
    except ValueError:
        pass
    return ProviderError(f"{label}: HTTP {status} {message[:300]}".rstrip(), status=status, retryable=status in RETRYABLE_STATUS)


async def with_retry(
    call: Callable[[], Awaitable[httpx.Response]], *, label: str, attempts: int = 2, backoff: float = 1.0, retry_timeouts: bool = True
) -> httpx.Response:
    """Run an HTTP call, retrying once on network errors, rate limits and 5xx (and timeouts unless disabled)."""
    last: ProviderError | None = None
    for i in range(attempts):
        try:
            res = await call()
        except httpx.TimeoutException:
            last = ProviderError(f"{label}: request timed out", retryable=retry_timeouts, kind="timeout")
            if not retry_timeouts:
                raise last from None
        except httpx.TransportError as e:
            last = ProviderError(f"{label}: cannot connect ({type(e).__name__})", retryable=True, kind="connect")
        else:
            if res.status_code < 400:
                return res
            retryable = res.status_code in RETRYABLE_STATUS
            last = ProviderError(f"{label}: HTTP {res.status_code} {_error_message(res)}", status=res.status_code, retryable=retryable)
            if not retryable:
                raise last
        if i + 1 < attempts:
            await asyncio.sleep(backoff * (i + 1))
    assert last is not None
    raise last


def _error_message(res: httpx.Response) -> str:
    try:
        data = res.json()
    except ValueError:
        return ""
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message", ""))[:300]
    if isinstance(err, str):
        return err[:300]
    return ""
