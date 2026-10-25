from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.assistant import Assistant
from app.config import Settings
from app.main import create_app
from app.providers import Completion, ProviderError
from app.providers.base import StreamChunk

TOKEN = "test-token-" + "x" * 32


class FakeProvider:
    def __init__(self, name: str, *, local: bool = False, fail: bool = False, reply: str = "ok", chunk: int = 4, fail_after: int | None = None):
        self.chunk = chunk
        self.fail_after = fail_after
        self.name = name
        self.model = f"{name}-model"
        self.local = local
        self.fail = fail
        self.reply = reply
        self.calls: list[dict] = []

    async def complete(self, system, messages, max_tokens, temperature):
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        if self.fail:
            raise ProviderError(f"{self.name}: HTTP 503", status=503, retryable=True)
        return Completion(text=self.reply, model=self.model, input_tokens=10, output_tokens=5)

    async def stream(self, system, messages, max_tokens, temperature):
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens, "stream": True})
        if self.fail:
            raise ProviderError(f"{self.name}: HTTP 503", status=503, retryable=True)
        yield StreamChunk(model=self.model, input_tokens=10)
        pieces = [self.reply[i : i + self.chunk] for i in range(0, len(self.reply), self.chunk)]
        for n, piece in enumerate(pieces):
            if self.fail_after is not None and n == self.fail_after:
                raise ProviderError(f"{self.name}: connection lost", kind="connect")
            yield StreamChunk(text=piece)
        yield StreamChunk(output_tokens=5)


def make_settings(**over) -> Settings:
    base = {"env": "test", "service_token": TOKEN, "fast_provider": "ollama", "top_provider": "anthropic", "top_api_key": "k"}
    base.update(over)
    return Settings(_env_file=None, **base)


@pytest.fixture
def build():
    def _build(*, fast=None, top=None, **settings):
        s = make_settings(**settings)
        a = Assistant(s, fast, top)
        client = TestClient(create_app(s, a))
        client.headers["authorization"] = f"Bearer {TOKEN}"
        return client, a

    return _build


def body(message: str, **extra) -> dict:
    return {"message": message, "context": {"org_id": "org-1", "user_role": "Finance User"}, **extra}


def read_sse(res) -> list[tuple[str, dict]]:
    import json

    events = []
    for block in res.text.split("\n\n"):
        lines = [line for line in block.split("\n") if line and not line.startswith(":")]
        if not lines:
            continue
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events
