"""Helpers for streaming answers token by token.

ThinkFilter drops a model's <think>…</think> scratchpad even when tags are split across chunks.
PlaceholderBuffer restores masked values ([PAN_1] …) even when a placeholder is split across chunks.
sse() turns (event, data) pairs into Server-Sent Events with heartbeat comments while the model is silent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator

from .redact import Redactor

OPEN, CLOSE = "<think>", "</think>"


def _partial_suffix(text: str, tag: str) -> int:
    """Length of the longest suffix of `text` that is a prefix of `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


class ThinkFilter:
    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        self._strip_lead = True  # drop whitespace at the very start and right after a think block

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._inside:
                j = self._buf.find(CLOSE)
                if j < 0:
                    keep = _partial_suffix(self._buf, CLOSE)
                    self._buf = self._buf[len(self._buf) - keep :] if keep else ""
                    break
                self._buf = self._buf[j + len(CLOSE) :]
                self._inside = False
                self._strip_lead = True
            else:
                j = self._buf.find(OPEN)
                if j >= 0:
                    out.append(self._buf[:j])
                    self._buf = self._buf[j + len(OPEN) :]
                    self._inside = True
                    continue
                keep = _partial_suffix(self._buf, OPEN)
                cut = len(self._buf) - keep
                out.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
        return self._emit("".join(out))

    def flush(self) -> str:
        if self._inside:
            self._buf = ""
            return ""
        text, self._buf = self._buf, ""
        return self._emit(text)

    def _emit(self, text: str) -> str:
        if self._strip_lead and text:
            text = text.lstrip()
            if text:
                self._strip_lead = False
        return text


class PlaceholderBuffer:
    """Holds back a trailing '[…' that may be the start of a placeholder until it is complete."""

    MAX = 24

    def __init__(self, redactor: Redactor) -> None:
        self._redactor = redactor
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        i = self._buf.rfind("[")
        if i >= 0 and "]" not in self._buf[i:] and len(self._buf) - i <= self.MAX:
            emit, self._buf = self._buf[:i], self._buf[i:]
        else:
            emit, self._buf = self._buf, ""
        return self._redactor.restore(emit)

    def flush(self) -> str:
        text, self._buf = self._buf, ""
        return self._redactor.restore(text)


async def sse(events: AsyncIterator[tuple[str, dict]], heartbeat: float = 15.0) -> AsyncIterator[str]:
    """Format events as SSE. Sends `: ping` while waiting so proxies and the Node API keep the connection open.

    If the client disconnects, the pending model call is cancelled, which closes the upstream connection
    (Ollama then stops generating).
    """
    pending: asyncio.Future | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(events.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=heartbeat)
            if not done:
                yield ": ping\n\n"
                continue
            try:
                event, data = pending.result()
            except StopAsyncIteration:
                pending = None
                break
            pending = None
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    finally:
        if pending is not None and not pending.done():
            logging.getLogger("finpro.ai").info("client disconnected, stopping the model")
            # Cancel the model call exactly once and let it close its upstream connection undisturbed. Without the
            # shield, Starlette/anyio keep re-cancelling this task, the cancellation propagates into `pending`, and
            # the cleanup that closes the Ollama connection is itself aborted — the model would keep generating.
            pending.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.shield(pending)
        else:
            aclose = getattr(events, "aclose", None)
            if aclose:
                with contextlib.suppress(BaseException):
                    await asyncio.shield(asyncio.ensure_future(aclose()))
