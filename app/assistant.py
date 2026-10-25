"""Core orchestration: prompt → route → redact → cache → call model (with fallback) → restore."""

from __future__ import annotations

import hashlib
import json
import logging
import time

from .agents import get_agent, system_prompt
from .config import Settings
from .limits import DailyBudget, TTLCache
from .providers import Completion, Provider, ProviderError
from .redact import Redactor
from .router import Tier, route
from .schemas import ChatRequest, ChatResponse, Document, RouteInfo, Usage

log = logging.getLogger("finpro.ai")


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

    async def chat(self, req: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        s = self.settings
        org = req.context.org_id
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
        plan = self._plan(decision.tier, org, notes)

        redactor = Redactor()
        redacted: list[dict] | None = None
        last_error: ProviderError | None = None
        for i, (provider, tier) in enumerate(plan):
            outgoing = messages
            if s.redact_pii and not provider.local:
                if redacted is None:
                    redacted = [{"role": m["role"], "content": redactor.redact(m["content"])} for m in messages]
                outgoing = redacted
            max_tokens = s.top_max_output_tokens if tier == "top" else s.max_output_tokens

            key = self._cache_key(org, provider, system, outgoing, max_tokens)
            hit = self.cache.get(key)
            cached = hit is not None
            try:
                if hit:
                    completion, _ = hit
                else:
                    completion = await provider.complete(system, outgoing, max_tokens, s.temperature)
            except ProviderError as e:
                last_error = e
                log.warning("model call failed", extra={"org_id": org, "provider": provider.name, "tier": tier, "error": e.detail})
                if i + 1 < len(plan):
                    notes.append(f"The {tier} model was unavailable, so a backup model answered.")
                continue

            if not completion.text:
                completion.text = "The model returned an empty response. Please rephrase the question or try again."
            elif not cached:
                self.cache.set(key, (completion, tier))
            if tier == "top" and not cached:
                self.budget.spend(org)

            latency = int((time.perf_counter() - started) * 1000)
            log.info(
                "chat",
                extra={
                    "org_id": org,
                    "agent": agent.id,
                    "tier": tier,
                    "provider": provider.name,
                    "model": completion.model,
                    "route_score": decision.score,
                    "route_mode": decision.mode,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                    "latency_ms": latency,
                    "cached": cached,
                    "fallback": i > 0,
                    "redacted": redactor.count,
                },
            )
            return ChatResponse(
                content=redactor.restore(completion.text),
                limitations=" ".join(dict.fromkeys(notes)) or None,
                tier=tier,
                provider=provider.name,
                model=completion.model,
                route=RouteInfo(mode=decision.mode, score=decision.score, reasons=decision.reasons),
                usage=Usage(input_tokens=completion.input_tokens, output_tokens=completion.output_tokens),
                latency_ms=latency,
                cached=cached,
                fallback_used=i > 0,
                redacted_items=redactor.count if outgoing is redacted else 0,
            )

        detail = last_error.detail if last_error else "no provider"
        raise ServiceError(502, f"All AI models failed for this request ({detail}).")

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
