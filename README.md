# AI_Assistant_Code — Finpro AI Assistant service

AI assistant for [Finpro AI](https://github.com/Ashtaq/Finpro-AI). FastAPI service that answers Finpro's AI assistant questions. The Finpro Node API keeps authentication, RBAC, tenant isolation, document access and audit logging; it sends this service the question, conversation history and the already-authorized document text, and stores the answer.

```
Browser → Finpro API (Node) ──Bearer token──▶ AI service (FastAPI, :8090)
                                                ├─ router: fast or top tier?
                                                ├─ PII masking (hosted models only)
                                                ├─ cache + per-org daily budget for the top tier
                                                ├─ fast tier: Ollama (dev) / hosted open-source model (prod)
                                                └─ top tier: Claude or any OpenAI-compatible model
```

## How a question is answered

1. **Prompt** — the agent (`finance`, `tax`, `audit`, `investment`, `compliance`, `excel`, `document`) picks the system prompt (`app/agents.py`). Documents are wrapped in `<document>` tags and treated as untrusted data.
2. **Route** (`app/router.py`) — a free, rule-based score: agent type, analysis terms (DCF, reconcile, compare, compute, 26AS, regime…), number of documents, length, figures, multiple questions, conversation length. Short look-ups score negative. Score ≥ `AI_COMPLEX_THRESHOLD` (default 4) → top tier. The user can force the top model with **Deep analysis** in the UI, and `AI_ROUTING_MODE=fast|top` forces a tier for everyone. Every answer returns the score and reasons.
3. **Budget** — top-tier calls are capped per organization per day (`AI_TOP_DAILY_LIMIT_PER_ORG`). Over the cap, or with no top model configured, the fast model answers and the reply says so.
4. **Mask** (`app/redact.py`) — before calling any non-local model, PAN, Aadhaar, mobile numbers and e-mails become `[PAN_1]`-style placeholders; the real values are restored in the answer. Ollama runs locally, so it gets the raw text.
5. **Cache** — identical requests within `AI_CACHE_TTL_SECONDS` (per organization) are answered from memory.
6. **Call with fallback** — one retry on rate limits/5xx/network errors; if the chosen tier still fails, the other tier answers and the reply notes it. Claude calls mark the system prompt cacheable (prompt caching lowers repeat input cost).

## Run locally (macOS)

```bash
# 1. Ollama — free local model for development
brew install ollama
ollama serve            # or open the Ollama app
ollama pull qwen3:8b    # default OLLAMA_MODEL; any Ollama chat model works

# 2. This service (Python 3.11+)
git clone https://github.com/Ashtaq/AI_Assistant_Code.git
cd AI_Assistant_Code
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
.venv/bin/uvicorn app.main:create_app --factory --reload --reload-dir app --port 8090
```

Check it: http://127.0.0.1:8090/health — interactive API docs at http://127.0.0.1:8090/docs (development only).

On startup the service loads the Ollama model into memory (warm-up), so the first question is not the slow one. Ollama is called through its native `/api/chat` API so Qwen3's long "thinking" step can be switched off (`think: false`, default `AI_OLLAMA_THINK=false`) — on a MacBook it can take minutes. On an 8 GB Mac use `qwen3:4b` (or `qwen3:1.7b` if that is still slow).

Optional in development: set `AI_TOP_PROVIDER=anthropic` and `AI_TOP_API_KEY` in `.env` to try the top tier; without it, complex questions are answered by Ollama with a note.

### Connect Finpro

In the Finpro repo's `.env`:

```env
AI_PROVIDER=service
AI_SERVICE_URL=http://127.0.0.1:8090
# AI_SERVICE_TOKEN=<same value as this service>   # optional in development
```

Start Finpro with `npm run dev:full` and this service as above. In development both sides fall back to the same built-in token, so no token is needed locally.

## Production

Fast tier = a hosted open-source model through any OpenAI-compatible API (Groq, Together, OpenRouter, DeepInfra, or your own vLLM server); top tier = Claude or another frontier model. Every setting is in [`.env.example`](.env.example).

```bash
docker build -t finpro-ai-assistant:1.0.0 .
docker run -d --name ai --network <private-net> --env-file ai.env --read-only --tmpfs /tmp finpro-ai-assistant:1.0.0
```

Keep the container on the private network only (no public port); the Finpro API reaches it at `AI_SERVICE_URL` with the shared `AI_SERVICE_TOKEN`. The service refuses to start in production without a 32+ character `AI_SERVICE_TOKEN` and a complete fast-tier configuration, and `/docs` is disabled.

Before enabling hosted or top models: review each provider's data-processing terms (prefer zero-data-retention plans), record customer consent for sending documents to AI providers and update the privacy policy.

The cache and daily budget are in memory (per container, reset on restart). Move them to Redis before running several replicas.

## API

| Method | Path              | Auth   | Purpose                                         |
| ------ | ----------------- | ------ | ----------------------------------------------- |
| GET    | `/health`         | none   | Status, environment and configured tiers/models |
| GET    | `/v1/agents`      | Bearer | Agent ids and names                             |
| POST   | `/v1/chat`        | Bearer | Answer a question (whole answer at once)        |
| POST   | `/v1/chat/stream` | Bearer | Answer as Server-Sent Events (token by token)   |

`POST /v1/chat` request:

```json
{
  "agent_id": "tax",
  "message": "Compare old vs new regime for me",
  "history": [
    { "role": "user", "content": "…" },
    { "role": "assistant", "content": "…" }
  ],
  "documents": [{ "name": "form16.txt", "text": "…" }],
  "context": { "org_id": "…", "user_id": "…", "user_role": "Individual" },
  "complexity": "auto"
}
```

`POST /v1/chat/stream` takes the same body and sends SSE events: `meta` (`tier`, `provider`, `model`), `delta` (`text`), then `done` (the full response below) or `error` (`error`, `partial`). `: ping` comments keep the connection alive while the model is loading. Closing the connection stops the model, so a **Stop** button in the client really stops generation. If a model fails before sending any text the next tier takes over; once text has been sent it never switches model mid-answer.

Response: `content`, `limitations`, `tier` (`fast`/`top`), `provider`, `model`, `route` (`mode`, `score`, `reasons`), `usage` (tokens), `latency_ms`, `cached`, `fallback_used`, `redacted_items`. Errors: `{ "error": "…" }` with 401, 422, 502 or 503.

## Develop

```bash
.venv/bin/pytest
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Code map: `app/main.py` (FastAPI app), `app/assistant.py` (orchestration), `app/router.py`, `app/redact.py`, `app/limits.py` (cache, budget), `app/agents.py` (prompts), `app/providers/` (Ollama/OpenAI-compatible, Anthropic), `app/config.py` (settings and production checks).
