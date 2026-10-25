import json

import httpx
import pytest

from app.agents import get_agent
from app.config import Settings
from app.limits import DailyBudget, TTLCache
from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatProvider
from app.redact import Redactor
from app.router import route


def run(coro):
    import asyncio

    return asyncio.run(coro)


# --- router ---------------------------------------------------------------------------


def decide(message, agent="finance", docs=(), history=0):
    return route(message=message, agent=get_agent(agent), history_len=history, doc_chars=list(docs), requested="auto", routing_mode="auto", threshold=4)


def test_router_simple_lookups_stay_fast():
    for q in ["What is GST?", "hi", "80C kya hai", "Full form of TDS"]:
        assert decide(q, "tax").tier == "fast", q


def test_router_analysis_goes_top():
    assert decide("Compare old vs new regime and calculate my tax liability", "tax").tier == "top"
    assert decide("Do a variance analysis of these two P&Ls", "finance", docs=[5000, 6000]).tier == "top"
    assert decide("Review these transactions for fraud risk", "audit").tier == "top"


def test_router_explains_itself():
    d = decide("Reconcile 26AS with AIS", "tax")
    assert any("analysis terms" in r for r in d.reasons)


# --- redaction ------------------------------------------------------------------------


def test_redactor_round_trip_and_stable_tokens():
    r = Redactor()
    text = "PAN ABCPV1234K, again ABCPV1234K, mail a.b@x.co.in, mobile +91 9876543210, aadhaar 234567890123"
    masked = r.redact(text)
    assert masked.count("[PAN_1]") == 2
    for raw in ["ABCPV1234K", "a.b@x.co.in", "9876543210", "234567890123"]:
        assert raw not in masked
    assert r.restore(masked) == text


def test_redactor_leaves_ordinary_numbers():
    r = Redactor()
    text = "Revenue 1,25,00,000 in FY 2025-26; invoice 1234567890123456"
    assert r.redact(text) == text


# --- limits ---------------------------------------------------------------------------


def test_ttl_cache_expires_and_evicts():
    now = [0.0]
    c = TTLCache(10, 2, clock=lambda: now[0])
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert c.get("a") is None and c.get("c") == 3
    now[0] = 11
    assert c.get("c") is None


def test_daily_budget_resets_each_day():
    day = ["2026-10-25"]
    b = DailyBudget(1, today=lambda: day[0])
    b.spend("o")
    assert not b.available("o")
    day[0] = "2026-10-26"
    assert b.available("o")


# --- providers ------------------------------------------------------------------------


def test_openai_compat_strips_reasoning_and_retries():
    calls = []

    def handler(req: httpx.Request):
        calls.append(json.loads(req.content))
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "choices": [{"message": {"content": "<think>hmm</think>Answer"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            p = OpenAICompatProvider(name="ollama", base_url="http://x/v1", model="qwen3:8b", local=True, client=client, retry_backoff=0)
            return await p.complete("sys", [{"role": "user", "content": "q"}], 100, 0.2)

    c = run(go())
    assert c.text == "Answer" and c.input_tokens == 3
    assert calls[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_compat_explains_missing_ollama():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            p = OpenAICompatProvider(name="ollama", base_url="http://x/v1", model="qwen3:8b", local=True, client=client, retry_backoff=0)
            await p.complete("s", [{"role": "user", "content": "q"}], 10, 0)

    with pytest.raises(ProviderError) as e:
        run(go())
    assert "ollama serve" in e.value.detail


def test_openai_compat_does_not_retry_auth_errors():
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(401, json={"error": "bad key"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            p = OpenAICompatProvider(name="groq", base_url="http://x/v1", model="m", api_key="k", client=client, retry_backoff=0)
            await p.complete("s", [{"role": "user", "content": "q"}], 10, 0)

    with pytest.raises(ProviderError):
        run(go())
    assert len(n) == 1


def test_anthropic_request_shape():
    seen = {}

    def handler(req: httpx.Request):
        seen["headers"] = req.headers
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200, json={"model": "claude-opus-5-5", "content": [{"type": "text", "text": "Hi"}], "usage": {"input_tokens": 5, "output_tokens": 1}}
        )

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            p = AnthropicProvider(api_key="key", model="claude-opus-5-5", client=client)
            msgs = [{"role": "assistant", "content": "orphan"}, {"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
            return await p.complete("sys", msgs, 50, 0.2)

    c = run(go())
    assert c.text == "Hi" and c.output_tokens == 1
    assert seen["headers"]["x-api-key"] == "key"
    assert seen["body"]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert seen["body"]["messages"] == [{"role": "user", "content": "a\n\nb"}]


# --- settings -------------------------------------------------------------------------


def test_settings_defaults_to_ollama_in_development():
    s = Settings(_env_file=None, env="development")
    assert s.fast_provider == "ollama" and s.token


def test_settings_refuse_unsafe_production():
    with pytest.raises(ValueError) as e:
        Settings(_env_file=None, env="production")
    msg = str(e.value)
    assert "AI_SERVICE_TOKEN" in msg and "AI_HOSTED_BASE_URL" in msg


def test_settings_valid_production():
    s = Settings(
        _env_file=None,
        env="production",
        service_token="t" * 40,
        hosted_base_url="https://api.groq.com/openai/v1",
        hosted_api_key="k",
        hosted_model="m",
        top_provider="anthropic",
        top_api_key="k2",
    )
    assert s.fast_provider == "hosted" and s.is_prod
