# syntax=docker/dockerfile:1.7
# Finpro AI Assistant service — internal only; the Finpro API calls it over the private network.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt && useradd --system --uid 10001 finpro
COPY app ./app

ENV AI_SERVICE_HOST=0.0.0.0 \
    AI_SERVICE_PORT=8090
USER finpro
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=4).status == 200 else 1)"
CMD ["python", "-m", "app"]
