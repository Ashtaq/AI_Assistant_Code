"""`python -m app` — run the service with host/port from settings."""

import uvicorn

from .config import Settings
from .main import create_app

settings = Settings()
uvicorn.run(create_app(settings), host=settings.service_host, port=settings.service_port, log_level="warning", access_log=False)
