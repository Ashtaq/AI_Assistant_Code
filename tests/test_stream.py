import asyncio
import random

from conftest import FakeProvider, body, read_sse

from app.redact import Redactor
from app.streaming import PlaceholderBuffer, ThinkFilter, sse


def stream(client, payload):
    res = client.post("/v1/chat/stream", json=payload)
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("text/event-stream")
    return read_sse(res)


def text_of(events):
    return "".join(d["text"] for e, d in events if e == "delta")


def test_streams_deltas_then_done(build):
    fast = FakeProvider("ollama", local=True, reply="GST is a tax on supply of goods and services.")
    client, _ = build(fast=fast, top=None, top_provider="none")
    events = stream(client, body("What is GST?"))
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert kinds.count("delta") > 3
    assert text_of(events) == "GST is a tax on supply of goods and services."
    done = events[-1][1]
    assert done["content"] == text_of(events) and done["provider"] == "ollama"
    assert done["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_stream_strips_thinking_and_restores_placeholders(build):
    fast = FakeProvider("groq", reply="<think>long reasoning</think>\n\nYour PAN [PAN_1] is valid.", chunk=3)
    client, _ = build(fast=fast, top=None, top_provider="none")
    events = stream(client, body("Is ABCPV1234K valid?"))
    assert text_of(events) == "Your PAN ABCPV1234K is valid."
    assert "[PAN_1]" in fast.calls[0]["messages"][-1]["content"]
    assert events[-1][1]["redacted_items"] == 1


def test_stream_falls_back_before_any_text(build):
    fast, top = FakeProvider("groq", fail=True), FakeProvider("anthropic", reply="from top")
    client, _ = build(fast=fast, top=top)
    events = stream(client, body("What is TDS?"))
    metas = [d for e, d in events if e == "meta"]
    assert [m["provider"] for m in metas] == ["groq", "anthropic"]
    assert events[-1][0] == "done" and events[-1][1]["fallback_used"] is True


def test_stream_does_not_switch_model_mid_answer(build):
    fast, top = FakeProvider("groq", reply="abcdefghijkl", fail_after=2), FakeProvider("anthropic")
    client, _ = build(fast=fast, top=top)
    events = stream(client, body("What is TDS?"))
    assert events[-1][0] == "error" and events[-1][1]["partial"] is True
    assert text_of(events) == "abcdefgh"
    assert not top.calls


def test_stream_all_failed(build):
    client, _ = build(fast=FakeProvider("groq", fail=True), top=FakeProvider("anthropic", fail=True))
    events = stream(client, body("hi"))
    assert events[-1][0] == "error" and "failed" in events[-1][1]["error"]


def test_stream_uses_cache(build):
    fast = FakeProvider("groq", reply="cached answer")
    client, _ = build(fast=fast, top=None, top_provider="none")
    stream(client, body("What is 80C?"))
    events = stream(client, body("What is 80C?"))
    assert text_of(events) == "cached answer" and events[-1][1]["cached"] is True
    assert len(fast.calls) == 1


def test_stream_requires_token_and_reports_config_errors_as_json(build):
    client, _ = build(fast=None, top=None, top_provider="none")
    assert client.post("/v1/chat/stream", json=body("x")).status_code == 503
    client.headers.pop("authorization")
    assert client.post("/v1/chat/stream", json=body("x")).status_code == 401


# --- unit -------------------------------------------------------------------------


def chunks(text, rng):
    i = 0
    while i < len(text):
        n = rng.randint(1, 5)
        yield text[i : i + n]
        i += n


def test_think_filter_any_chunking():
    rng = random.Random(7)
    for raw, want in [
        ("<think>\nplan\n</think>\n\nAnswer here.", "Answer here."),
        ("No thinking at all <b>.", "No thinking at all <b>."),
        ("<think></think>Hi", "Hi"),
        ("a < b and <thin", "a < b and <thin"),
    ]:
        for _ in range(30):
            f = ThinkFilter()
            out = "".join(f.feed(c) for c in chunks(raw, rng)) + f.flush()
            assert out == want, (raw, out)


def test_placeholder_buffer_any_chunking():
    rng = random.Random(3)
    r = Redactor()
    masked = r.redact("PAN ABCPV1234K mail a@b.in [note] done")
    for _ in range(30):
        b = PlaceholderBuffer(r)
        out = "".join(b.feed(c) for c in chunks(masked, rng)) + b.flush()
        assert out == "PAN ABCPV1234K mail a@b.in [note] done"


def test_sse_heartbeat_and_cancellation():
    cancelled = asyncio.Event()

    async def slow():
        try:
            yield "meta", {"a": 1}
            await asyncio.sleep(10)
            yield "delta", {"text": "never"}
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def go():
        out = []
        gen = sse(slow(), heartbeat=0.05)
        out.append(await gen.__anext__())
        out.append(await gen.__anext__())
        await gen.aclose()  # client went away
        return out

    out = asyncio.run(go())
    assert out[0].startswith("event: meta") and out[1] == ": ping\n\n"
    assert cancelled.is_set()


def test_disconnect_under_anyio_cancellation_lets_the_model_call_clean_up():
    """Starlette cancels via anyio (repeatedly). The model call's async cleanup must still complete,
    otherwise the upstream connection stays open and the model keeps generating."""
    import anyio

    closed = asyncio.Event()

    async def model():
        try:
            yield "meta", {}
            while True:
                await asyncio.sleep(0.01)
                yield "delta", {"text": "x"}
        finally:
            await asyncio.sleep(0.02)  # e.g. closing the HTTP connection
            closed.set()

    async def go():
        gen = sse(model(), heartbeat=5)

        async def consume():
            async for _ in gen:
                pass

        async with anyio.create_task_group() as tg:
            tg.start_soon(consume)
            await anyio.sleep(0.1)
            tg.cancel_scope.cancel()
        await asyncio.wait_for(closed.wait(), 1)

    asyncio.run(go())
    assert closed.is_set()
