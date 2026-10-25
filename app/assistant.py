"""Core orchestration: prompt → route → redact → cache → call model (with fallback) → restore.

`chat()` returns the whole answer at once; `stream()` yields it token by token. Both share `prepare()`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .agents import get_agent, system_prompt
from .config import Settings
from .limits import DailyBudget, TTLCache
from .providers import Completion, Provider, ProviderError
from .redact import Redactor
from .router import RouteDecision, Tier, route
from .schemas import ChatRequest, ChatResponse, Document, RouteInfo, Usage
from .streaming import PlaceholderBuffer, ThinkFilter

log = logging.getLogger("finpro.ai")

EMPTY_ANSWER = "The model returned an empty response. Please rephrase the question or try again."


class ServiceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _attr(value: str) -> str:
    return value.replace('"', "'").replace("<", "(").replace(">", ")")


def build_document_block(documents: list[Document], budget: int) -> tuple[str, list[str]]:
    """Render documents for the prompt within a character budget. Returns (block, limitation notes)."""
    if not documents:
        return "No documents attached.", []
    parts: list[str] = []
    truncated: list[str] = []
    for d in documents:
        name = _attr(d.name)
        if d.text is None:
            parts.append(f'<document name="{name}" note="contents not extracted" />')
            continue
        text = d.text
        if len(text) > budget:
            text = text[: max(budget, 0)]
            truncated.append(d.name)
        budget -= len(text)
        parts.append(f'<document name="{name}">\n{text}\n</document>')
    notes = [f"Only part of {', '.join(truncated)} was read (context limit)."] if truncated else []
    return "\n".join(parts), notes


@dataclass
class Prepared:
    org: str
    agent_id: str
    system: str
    messages: list[dict]
    decision: RouteDecision
    plan: list[tuple[Provider, Tier]]
    notes: list[str]
    started: float
    redactor: Redactor = field(default_factory=Redactor)
    _redacted: list[dict] | None = None

    def outgoing(self, provider: Provider, redact: bool) -> list[dict]:
        if not redact or provider.local:
            return self.messages
        if self._redacted is None:
            self._redacted = [{"role": m["role"], "content": self.redactor.redact(m["content"])} for m in self.messages]
        return self._redacted


class Assistant:
    def __init__(self, settings: Settings, fast: Provider | None, top: Provider | None):
        self.settings = settings
        self.fast = fast
        self.top = top
        self.cache: TTLCache[tuple[Completion, Tier]] = TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries)
        self.budget = DailyBudget(settings.top_daily_limit_per_org)

    def tiers(self) -> dict:
        def info(p: Provider | None) -> dict | None:
            return {"provider": p.name, "model": p.model, "local": p.local} if p else None

        return {"fast": info(self.fast), "top": info(self.top)}

    # --- shared ---------------------------------------------------------------------

    def prepare(self, req: ChatRequest) -> Prepared:
        """Build the prompt and decide which models to try. Raises ServiceError before any model call."""
        s = self.settings
        agent = get_agent(req.agent_id)
        system = system_prompt(agent, user_role=req.context.user_role, project_name=req.context.project_name, client_name=req.context.client_name)
        doc_block, notes = build_document_block(req.documents, s.max_context_chars)
        messages = [{"role": m.role, "content": m.content} for m in req.history]
        messages.append({"role": "user", "content": f"{req.message}\n\n<attached_documents>\n{doc_block}\n</attached_documents>"})
        decision = route(
            message=req.message,
            agent=agent,
            history_len=len(req.history),
            doc_chars=[len(d.text or "") for d in req.documents],
            requested=req.complexity,
            routing_mode=s.routing_mode,
            threshold=s.complex_threshold,
        )
        plan = self._plan(decision.tier, req.context.org_id, notes)
        return Prepared(req.context.org_id, agent.id, system, messages, decision, plan, notes, time.perf_counter())

    def _max_tokens(self, tier: Tier) -> int:
        return self.settings.top_max_output_tokens if tier == "top" else self.settings.max_output_tokens

    def _finish(self, p: Prepared, provider: Provider, tier: Tier, completion: Completion, *, cached: bool, attempt: int, redacted: bool) -> ChatResponse:
        if tier == "top" and not cached:
            self.budget.spend(p.org)
        latency = int((time.perf_counter() - p.started) * 1000)
        redacted_items = p.redactor.count if redacted else 0
        log.info(
            "chat",
            extra={
                "org_id": p.org,
                "agent": p.agent_id,
                "tier": tier,
                "provider": provider.name,
                "model": completion.model,
                "route_score": p.decision.score,
                "route_mode": p.decision.mode,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "latency_ms": latency,
                "cached": cached,
                "fallback": attempt > 0,
                "redacted": redacted_items,
            },
        )
        return ChatResponse(
            content=p.redactor.restore(completion.text) or EMPTY_ANSWER,
            limitations=" ".join(dict.fromkeys(p.notes)) or None,
            tier=tier,
            provider=provider.name,
            model=completion.model,
            route=RouteInfo(mode=p.decision.mode, score=p.decision.score, reasons=p.decision.reasons),
            usage=Usage(input_tokens=completion.input_tokens, output_tokens=completion.output_tokens),
            latency_ms=latency,
            cached=cached,
            fallback_used=attempt > 0,
            redacted_items=redacted_items,
        )

    def _failed(self, p: Prepared, provider: Provider, tier: Tier, e: ProviderError, attempt: int) -> None:
        log.warning("model call failed", extra={"org_id": p.org, "provider": provider.name, "tier": tier, "error": e.detail})
        if attempt + 1 < len(p.plan):
            p.notes.append(f"The {tier} model was unavailable, so a backup model answered.")

    # --- whole answer ---------------------------------------------------------------

    async def chat(self, req: ChatRequest) -> ChatResponse:
        p = self.prepare(req)
        s = self.settings
        last_error: ProviderError | None = None
        for i, (provider, tier) in enumerate(p.plan):
            outgoing = p.outgoing(provider, s.redact_pii)
            key = self._cache_key(p.org, provider, p.system, outgoing, self._max_tokens(tier))
            hit = self.cache.get(key)
            if hit:
                return self._finish(p, provider, tier, hit[0], cached=True, attempt=i, redacted=outgoing is not p.messages)
            try:
                completion = await provider.complete(p.system, outgoing, self._max_tokens(tier), s.temperature)
            except ProviderError as e:
                last_error = e
                self._failed(p, provider, tier, e, i)
                continue
            if completion.text:
                self.cache.set(key, (completion, tier))
            return self._finish(p, provider, tier, completion, cached=False, attempt=i, redacted=outgoing is not p.messages)
        raise ServiceError(502, f"All AI models failed for this request ({last_error.detail if last_error else 'no provider'}).")

    # --- streamed answer ------------------------------------------------------------

    async def stream(self, p: Prepared) -> AsyncIterator[tuple[str, dict]]:
        """Yield ("meta" | "delta" | "done" | "error", data). Falls back to the next model only before any text was sent."""
        s = self.settings
        last_error: ProviderError | None = None
        for i, (provider, tier) in enumerate(p.plan):
            outgoing = p.outgoing(provider, s.redact_pii)
            redacted = outgoing is not p.messages
            max_tokens = self._max_tokens(tier)
            yield "meta", {"tier": tier, "provider": provider.name, "model": provider.model, "fallback_used": i > 0}

            key = self._cache_key(p.org, provider, p.system, outgoing, max_tokens)
            hit = self.cache.get(key)
            if hit:
                answer = self._finish(p, provider, tier, hit[0], cached=True, attempt=i, redacted=redacted)
                yield "delta", {"text": answer.content}
                yield "done", answer.model_dump()
                return

            think, placeholders = ThinkFilter(), PlaceholderBuffer(p.redactor)
            parts: list[str] = []
            sent = False
            model, tokens_in, tokens_out = provider.model, None, None
            try:
                async for chunk in provider.stream(p.system, outgoing, max_tokens, s.temperature):
                    model = chunk.model or model
                    tokens_in = chunk.input_tokens if chunk.input_tokens is not None else tokens_in
                    tokens_out = chunk.output_tokens if chunk.output_tokens is not None else tokens_out
                    if chunk.text:
                        clean = think.feed(chunk.text)
                        parts.append(clean)
                        out = placeholders.feed(clean)
                        if out:
                            sent = True
                            yield "delta", {"text": out}
                tail = think.flush()
                parts.append(tail)
                out = placeholders.feed(tail) + placeholders.flush()
                if out:
                    yield "delta", {"text": out}
            except ProviderError as e:
                last_error = e
                self._failed(p, provider, tier, e, i)
                if sent:  # part of the answer is already on screen; do not mix in another model
                    yield "error", {"error": f"The answer was interrupted ({e.detail}).", "partial": True}
                    return
                continue

            completion = Completion(text="".join(parts).strip(), model=model, input_tokens=tokens_in, output_tokens=tokens_out)
            if completion.text:
                self.cache.set(key, (completion, tier))
            yield "done", self._finish(p, provider, tier, completion, cached=False, attempt=i, redacted=redacted).model_dump()
            return
        yield "error", {"error": f"All AI models failed for this request ({last_error.detail if last_error else 'no provider'})."}

    # --- helpers --------------------------------------------------------------------

    def _plan(self, wanted: Tier, org: str, notes: list[str]) -> list[tuple[Provider, Tier]]:
        top_ok = self.top is not None and self.budget.available(org)
        if wanted == "top" and not top_ok:
            notes.append(
                "Advanced analysis is not configured, so the standard model answered."
                if self.top is None
                else "Today's advanced-analysis limit for your organization is used up, so the standard model answered."
            )
            wanted = "fast"
        if wanted == "fast" and self.fast is None:
            if not top_ok:
                raise ServiceError(503, "No AI model is available right now. Please try again later.")
            wanted = "top"

        plan: list[tuple[Provider, Tier]] = []
        if wanted == "top" and self.top:
            plan.append((self.top, "top"))
            if self.fast:
                plan.append((self.fast, "fast"))
        elif self.fast:
            plan.append((self.fast, "fast"))
            if top_ok and self.top:
                plan.append((self.top, "top"))
        return plan

    @staticmethod
    def _cache_key(org: str, provider: Provider, system: str, messages: list[dict], max_tokens: int) -> str:
        raw = json.dumps([org, provider.name, provider.model, system, messages, max_tokens], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()
