"""Decide whether a question needs the top model or the cheap fast model.

Rule-based on purpose: it is free, adds no latency, is deterministic and easy to tune.
Each signal adds to a score; at or above AI_COMPLEX_THRESHOLD the question goes to the top tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .agents import Agent

Tier = Literal["fast", "top"]

# Work that needs multi-step reasoning or careful numbers.
COMPLEX_TERMS = re.compile(
    r"\b("
    r"dcf|valuation|valu(e|ing)\s+the|wacc|irr|npv|sensitivity|scenario|forecast|projection|"
    r"reconcil\w*|variance|compare|comparison|analy[sz](e|is|ing)|trend|"
    r"ratio\s+analysis|cash\s*flow|working\s+capital|deferred\s+tax|ind\s*as|ifrs|transfer\s+pricing|"
    r"computation|compute|calculate|tax\s+liability|old\s+(vs|versus|or)\s+new|regime|capital\s+gains?|"
    r"26as|\bais\b|mismatch|audit|risk|fraud|anomal\w*|outlier|due\s+diligence|"
    r"step[\s-]by[\s-]step|explain\s+why|justify|"
    r"nikalo|calculate\s+karo|compare\s+karo|samjhao\s+detail"
    r")\b",
    re.IGNORECASE,
)
# Quick look-ups and small talk.
SIMPLE_START = re.compile(
    r"^\s*(hi|hello|hey|namaste|thanks|thank you|ok|okay|what\s+is|what's|define|meaning\s+of|full\s+form|kya\s+hai|kya\s+hota)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d[\d,]*(\.\d+)?")


@dataclass
class RouteDecision:
    tier: Tier
    mode: Literal["auto", "user", "forced"]
    score: int = 0
    reasons: list[str] = field(default_factory=list)


def route(
    *,
    message: str,
    agent: Agent,
    history_len: int,
    doc_chars: list[int],
    requested: Literal["auto", "fast", "top"],
    routing_mode: Literal["auto", "fast", "top"],
    threshold: int,
) -> RouteDecision:
    if routing_mode != "auto":
        return RouteDecision(routing_mode, "forced", reasons=[f"AI_ROUTING_MODE={routing_mode}"])
    if requested != "auto":
        return RouteDecision(requested, "user", reasons=["requested by user"])

    score = 0
    reasons: list[str] = []

    def add(points: int, why: str) -> None:
        nonlocal score
        score += points
        reasons.append(f"{'+' if points >= 0 else ''}{points} {why}")

    if agent.weight:
        add(agent.weight, f"{agent.id} agent")

    hits = {m.group(0).lower() for m in COMPLEX_TERMS.finditer(message)}
    if hits:
        add(2 if len(hits) < 3 else 3, "analysis terms: " + ", ".join(sorted(hits)[:5]))

    readable = [n for n in doc_chars if n > 0]
    if readable:
        add(min(len(readable), 2), f"{len(readable)} document(s)")
        if sum(readable) > 12_000:
            add(1, "large documents")

    if len(message) > 1500:
        add(2, "long question")
    elif len(message) > 600:
        add(1, "detailed question")

    if len(NUMBER.findall(message)) >= 8:
        add(1, "many figures")
    if message.count("?") >= 2:
        add(1, "multiple questions")
    if history_len >= 8:
        add(1, "long conversation")

    if SIMPLE_START.search(message) and len(message) < 120 and not readable:
        add(-2, "short look-up")

    return RouteDecision("top" if score >= threshold else "fast", "auto", score, reasons)
