"""FastAPI entry point. Internal service: only the Finpro Node API calls it, with a bearer token.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8090
"""

from __future__ import annotations

import hmac
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .agents import AGENTS
from .assistant import Assistant, ServiceError
from .config import Settings
from .providers import build_providers
from .schemas import ChatRequest, ChatResponse

_STD = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"level": record.levelname.lower(), "msg": record.getMessage(), "logger": record.name}
        out.update({k: v for k, v in record.__dict__.items() if k not in _STD})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("finpro")
    root.handlers = [handler]
    root.setLevel(level.upper())
    root.propagate = False


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    settings = settings or Settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = None
        if app.state.assistant is None:
            client = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
            fast, top = build_providers(settings, client)
            app.state.assistant = Assistant(settings, fast, top)
        a: Assistant = app.state.assistant
        logging.getLogger("finpro.ai").info("AI service ready", extra={"env": settings.env, **{f"{k}_tier": v for k, v in a.tiers().items()}})
        yield
        if client:
            await client.aclose()

    app = FastAPI(
        title="Finpro AI Assistant",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None if settings.is_prod else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_prod else "/openapi.json",
    )
    app.state.assistant = assistant
    app.state.settings = settings

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = f"Bearer {settings.token}"
        if not authorization or not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise ServiceError(401, "Unauthorized.")

    @app.exception_handler(ServiceError)
    async def _service_error(_req: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"error": exc.message}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()]
        return JSONResponse({"error": "Invalid request.", "details": details}, status_code=422)

    @app.exception_handler(Exception)
    async def _unexpected(_req: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("finpro.ai").exception("Unhandled error")
        return JSONResponse({"error": "Internal error."}, status_code=500)

    @app.get("/health")
    async def health(request: Request) -> dict:
        a: Assistant = request.app.state.assistant
        return {"status": "ok", "env": settings.env, "routing": settings.routing_mode, "tiers": a.tiers()}

    @app.get("/v1/agents", dependencies=[Depends(require_token)])
    async def agents() -> list[dict]:
        return [{"id": a.id, "name": a.name} for a in AGENTS.values()]

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_token)])
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
        a: Assistant = request.app.state.assistant
        return await a.chat(body)

    return app


def __getattr__(name: str):
    # `uvicorn app.main:app` — build lazily so importing this module (tests) does not read the environment.
    if name == "app":
        return create_app()
    raise AttributeError(name)
