"""Handler de logging: un archivo {YYYY-MM-DD}.log por día (opt-in SID)."""
from __future__ import annotations

import logging
import threading
from datetime import date
from pathlib import Path


class DailyNamedFileHandler(logging.Handler):
    def __init__(self, log_dir: str | Path, backup_count: int = 30):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backup_count = max(1, backup_count)
        self._lock = threading.RLock()
        self._current_date: date | None = None
        self._stream = None

    def _path_for(self, day: date) -> Path:
        return self.log_dir / f"{day.isoformat()}.log"

    def _ensure_stream(self) -> None:
        today = date.today()
        if self._current_date == today and self._stream is not None:
            return
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._current_date = today
        self._stream = open(self._path_for(today), "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self._lock:
                self._ensure_stream()
                assert self._stream is not None
                msg = self.format(record)
                self._stream.write(msg + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        super().close()
