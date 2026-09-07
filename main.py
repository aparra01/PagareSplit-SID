from __future__ import annotations

import uvicorn

from app.config import get_settings

try:
    from logging_config import configure_logging
    configure_logging()
except ImportError:
    pass


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=max(1, settings.workers),
    )
