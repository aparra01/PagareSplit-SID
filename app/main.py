from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.splitter import detectar_pagares_actual_por_barcode

app = FastAPI(
    title="PagareSplit-SID",
    version="0.1.0",
    description="Servicio dedicado para separar lotes PDF de pagarés sin ejecutar OCR pesado.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "PagareSplit-SID"}


@app.post("/detectar-pagares-actual")
async def detectar_pagares_actual(
    file: UploadFile = File(...),
    dpi: int = Form(default=160),
    solo_rangos: bool = Form(default=False),
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
    return detectar_pagares_actual_por_barcode(pdf_bytes=pdf_bytes, dpi=safe_dpi, solo_rangos=solo_rangos)
