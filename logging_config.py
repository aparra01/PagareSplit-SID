"""Logging opt-in SID para PagareSplit."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sid_daily_file_handler import DailyNamedFileHandler

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir_raw = (os.environ.get("PAGARE_SPLIT_LOG_DIR") or "").strip()
    if not log_dir_raw and os.environ.get("SID_DAILY_LOG") != "1":
        _CONFIGURED = True
        return

    log_dir = Path(log_dir_raw or "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = DailyNamedFileHandler(log_dir, backup_count=30)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.getLogger("sid.startup").info("Logging diario SID activo en %s", log_dir.resolve())

    _CONFIGURED = True
