from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.splitter import detectar_pagares_actual_por_barcode, validar_orden_pdf_sucursales

try:
    from logging_config import configure_logging
    configure_logging()
except ImportError:
    pass

app = FastAPI(
    title="PagareSplit-SID",
    version="0.1.0",
    description="Servicio dedicado para separar lotes PDF de pagarés sin ejecutar OCR pesado.",
)

_settings = get_settings()
_validation_semaphore = asyncio.Semaphore(max(1, _settings.validation_max_concurrent))
_debug_log_path = Path(__file__).resolve().parents[2] / "debug-9bf231.log"


def _write_debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "9bf231",
            "runId": "queue-control",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _debug_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "PagareSplit-SID"}


@app.post("/detectar-pagares-actual")
async def detectar_pagares_actual(
    file: UploadFile = File(...),
    dpi: int = Form(default=160),
    solo_rangos: bool = Form(default=False),
    separar_qr: bool = Form(default=False),
    separar_barcode: bool = Form(default=True),
):
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sube un archivo .pdf")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(400, "PDF vacío")

    max_bytes = max(1, settings.max_pdf_mb) * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(413, f"PDF demasiado grande para separación ({settings.max_pdf_mb} MB máximo)")

    safe_dpi = max(72, min(int(dpi or settings.default_dpi), 300))
    async with _validation_semaphore:
        return await asyncio.to_thread(
            detectar_pagares_actual_por_barcode,
            pdf_bytes=pdf_bytes,
            dpi=safe_dpi,
            solo_rangos=solo_rangos,
            separar_qr=separar_qr,
            separar_barcode=separar_barcode,
        )


@app.post("/validar-orden-sucursales")
async def validar_orden_sucursales(
    file: UploadFile = File(...),
    dpi: int = Form(default=160),
):
    """Flujo sucursales (DocNative): detecta PDF con operaciones intercaladas."""
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sube un archivo .pdf")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(400, "PDF vacío")

    max_bytes = max(1, settings.max_pdf_mb) * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(413, f"PDF demasiado grande para validación ({settings.max_pdf_mb} MB máximo)")

    safe_dpi = max(72, min(int(dpi or settings.default_dpi), 300))
    filename = file.filename or "upload.pdf"

    if _validation_semaphore.locked():
        # #region agent log
        _write_debug_log(
            "H2",
            "main.py:validar_orden_sucursales",
            "waiting_for_slot",
            {"archivo": filename, "maxConcurrent": settings.validation_max_concurrent},
        )
        # #endregion

    wait_started = time.perf_counter()
    async with _validation_semaphore:
        wait_ms = int((time.perf_counter() - wait_started) * 1000)
        # #region agent log
        _write_debug_log(
            "H2",
            "main.py:validar_orden_sucursales",
            "slot_acquired",
            {
                "archivo": filename,
                "waitMs": wait_ms,
                "maxConcurrent": settings.validation_max_concurrent,
            },
        )
        # #endregion

        return await asyncio.to_thread(
            validar_orden_pdf_sucursales,
            pdf_bytes=pdf_bytes,
            dpi=safe_dpi,
        )
