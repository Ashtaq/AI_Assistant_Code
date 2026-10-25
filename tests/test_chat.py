from conftest import FakeProvider, body


def test_simple_question_uses_fast_tier(build):
    fast, top = FakeProvider("ollama", local=True), FakeProvider("anthropic")
    client, _ = build(fast=fast, top=top)
    res = client.post("/v1/chat", json=body("What is GST?", agent_id="compliance"))
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["tier"] == "fast" and data["provider"] == "ollama"
    assert len(fast.calls) == 1 and not top.calls
    assert data["route"]["mode"] == "auto"


def test_complex_question_uses_top_tier(build):
    fast, top = FakeProvider("groq"), FakeProvider("anthropic")
    client, _ = build(fast=fast, top=top)
    res = client.post("/v1/chat", json=body("Build a DCF valuation with WACC and a sensitivity table for this company", agent_id="investment"))
    data = res.json()
    assert data["tier"] == "top", data["route"]
    assert top.calls[0]["max_tokens"] == 3000
    assert not fast.calls


def test_user_can_request_top_and_fast(build):
    fast, top = FakeProvider("groq"), FakeProvider("anthropic")
    client, _ = build(fast=fast, top=top)
    assert client.post("/v1/chat", json=body("hi", complexity="top")).json()["route"]["mode"] == "user"
    assert top.calls
    r = client.post("/v1/chat", json=body("Reconcile 26AS and AIS and compute tax liability", agent_id="tax", complexity="fast")).json()
    assert r["tier"] == "fast"


def test_routing_mode_forces_tier(build):
    fast, top = FakeProvider("groq"), FakeProvider("anthropic")
    client, _ = build(fast=fast, top=top, routing_mode="fast")
    r = client.post("/v1/chat", json=body("x", complexity="top")).json()
    assert r["tier"] == "fast" and r["route"]["mode"] == "forced"


def test_complex_without_top_model_falls_back_with_note(build):
    fast = FakeProvider("ollama", local=True)
    client, _ = build(fast=fast, top=None, top_provider="none")
    r = client.post("/v1/chat", json=body("analyze", complexity="top")).json()
    assert r["tier"] == "fast"
    assert "not configured" in r["limitations"]


def test_daily_top_budget_per_org(build):
    fast, top = FakeProvider("groq"), FakeProvider("anthropic")
    client, _ = build(fast=fast, top=top, top_daily_limit_per_org=1, cache_ttl_seconds=0)
    assert client.post("/v1/chat", json=body("a", complexity="top")).json()["tier"] == "top"
    second = client.post("/v1/chat", json=body("b", complexity="top")).json()
    assert second["tier"] == "fast" and "limit" in second["limitations"]
    other_org = body("c", complexity="top")
    other_org["context"]["org_id"] = "org-2"
    assert client.post("/v1/chat", json=other_org).json()["tier"] == "top"


def test_fast_failure_falls_back_to_top(build):
    fast, top = FakeProvider("groq", fail=True), FakeProvider("anthropic", reply="from top")
    client, _ = build(fast=fast, top=top)
    r = client.post("/v1/chat", json=body("What is TDS?")).json()
    assert r["content"] == "from top" and r["fallback_used"] is True and r["tier"] == "top"
    assert "backup" in r["limitations"]


def test_all_models_failing_returns_502(build):
    client, _ = build(fast=FakeProvider("groq", fail=True), top=FakeProvider("anthropic", fail=True))
    res = client.post("/v1/chat", json=body("hello"))
    assert res.status_code == 502
    assert "failed" in res.json()["error"]


def test_no_models_returns_503(build):
    client, _ = build(fast=None, top=None, top_provider="none", fast_provider="ollama")
    assert client.post("/v1/chat", json=body("hello")).status_code == 503


def test_pii_is_masked_for_hosted_models_and_restored(build):
    fast = FakeProvider("groq", reply="Your PAN [PAN_1] is linked to Aadhaar [AADHAAR_1].")
    client, _ = build(fast=fast, top=None, top_provider="none")
    r = client.post("/v1/chat", json=body("My PAN is ABCPV1234K and Aadhaar 2345 6789 0123, is it linked?")).json()
    sent = fast.calls[0]["messages"][-1]["content"]
    assert "ABCPV1234K" not in sent and "2345 6789 0123" not in sent
    assert "[PAN_1]" in sent and "[AADHAAR_1]" in sent
    assert r["content"] == "Your PAN ABCPV1234K is linked to Aadhaar 2345 6789 0123."
    assert r["redacted_items"] == 2


def test_local_models_receive_raw_text(build):
    fast = FakeProvider("ollama", local=True)
    client, _ = build(fast=fast, top=None, top_provider="none")
    client.post("/v1/chat", json=body("PAN ABCPV1234K"))
    assert "ABCPV1234K" in fast.calls[0]["messages"][-1]["content"]


def test_identical_requests_are_cached_per_org(build):
    fast = FakeProvider("groq")
    client, _ = build(fast=fast, top=None, top_provider="none")
    assert client.post("/v1/chat", json=body("What is 80C?")).json()["cached"] is False
    assert client.post("/v1/chat", json=body("What is 80C?")).json()["cached"] is True
    other = body("What is 80C?")
    other["context"]["org_id"] = "org-2"
    assert client.post("/v1/chat", json=other).json()["cached"] is False
    assert len(fast.calls) == 2


def test_documents_and_history_reach_the_model(build):
    fast = FakeProvider("ollama", local=True)
    client, _ = build(fast=fast, top=None, top_provider="none")
    client.post(
        "/v1/chat",
        json=body(
            "Summarize",
            agent_id="document",
            history=[{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}],
            documents=[{"name": 'p&l".csv', "text": "revenue,100"}, {"name": "scan.pdf", "text": None}],
        ),
    )
    call = fast.calls[0]
    assert [m["role"] for m in call["messages"]] == ["user", "assistant", "user"]
    last = call["messages"][-1]["content"]
    assert '<document name="p&l\'.csv">' in last and "revenue,100" in last
    assert 'name="scan.pdf" note="contents not extracted"' in last
    assert "Document Intelligence" in call["system"]


def test_requires_token(build):
    client, _ = build(fast=FakeProvider("groq"))
    client.headers.pop("authorization")
    assert client.post("/v1/chat", json=body("x")).status_code == 401
    client.headers["authorization"] = "Bearer wrong"
    assert client.post("/v1/chat", json=body("x")).status_code == 401
    assert client.get("/health").status_code == 200


def test_validation_errors_are_readable(build):
    client, _ = build(fast=FakeProvider("groq"))
    res = client.post("/v1/chat", json={"message": "", "context": {}})
    assert res.status_code == 422
    fields = {d["field"] for d in res.json()["details"]}
    assert "message" in fields and "context.org_id" in fields


def test_health_lists_tiers_without_secrets(build):
    client, _ = build(fast=FakeProvider("ollama", local=True), top=FakeProvider("anthropic"))
    data = client.get("/health").json()
    assert data["tiers"]["fast"]["provider"] == "ollama"
    assert "key" not in str(data).lower()
