"""Request/response contract between the Finpro Node API and this service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class Document(BaseModel):
    name: str = Field(max_length=255)
    text: str | None = Field(None, max_length=60_000, description="Extracted text; null when the file could not be read")


class Context(BaseModel):
    org_id: str = Field(min_length=1, max_length=64)
    user_id: str | None = Field(None, max_length=64)
    user_role: str | None = Field(None, max_length=64)
    project_name: str | None = Field(None, max_length=200)
    client_name: str | None = Field(None, max_length=200)


class ChatRequest(BaseModel):
    agent_id: str = Field("finance", max_length=32)
    message: str = Field(min_length=1, max_length=8_000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=24)
    documents: list[Document] = Field(default_factory=list, max_length=8)
    context: Context
    complexity: Literal["auto", "fast", "top"] = "auto"


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class RouteInfo(BaseModel):
    mode: Literal["auto", "user", "forced"]
    score: int
    reasons: list[str]


class ChatResponse(BaseModel):
    content: str
    limitations: str | None = None
    tier: Literal["fast", "top"]
    provider: str
    model: str
    route: RouteInfo
    usage: Usage
    latency_ms: int
    cached: bool = False
    fallback_used: bool = False
    redacted_items: int = 0
