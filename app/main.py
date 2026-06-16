from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import get_settings
from app.pdf_preprocess import normalizar_pdf
from app.scan_orientation import BlankDocumentError
from app.splitter import detectar_pagares_actual_por_barcode

app = FastAPI(
    title="PagareSplit-SID",
    version="0.2.0",
    description="Servicio dedicado para separar lotes PDF de pagarés sin ejecutar OCR pesado.",
)


async def _read_pdf_upload(file: UploadFile) -> bytes:
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Sube un archivo .pdf")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(400, "PDF vacío")

    max_bytes = max(1, settings.max_pdf_mb) * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(413, f"PDF demasiado grande para separación ({settings.max_pdf_mb} MB máximo)")
    return pdf_bytes


def _form_bool(value: bool | None, default: bool) -> bool:
    return default if value is None else value


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "PagareSplit-SID"}


@app.post("/normalizar-pdf")
async def normalizar_pdf_endpoint(
    file: UploadFile = File(...),
    eliminar_blancos: bool | None = Form(default=None),
    corregir_orientacion: bool | None = Form(default=None),
    aplicar_deskew: bool | None = Form(default=None),
    aplicar_mejora_imagen: bool | None = Form(default=None),
):
    """Devuelve PDF normalizado: blancos, orientación, deskew y mejora tipo PaperStream."""
    settings = get_settings()
    pdf_bytes = await _read_pdf_upload(file)
    try:
        result = normalizar_pdf(
            pdf_bytes,
            settings=settings,
            eliminar_blancos=eliminar_blancos,
            corregir_orientacion=corregir_orientacion,
            aplicar_deskew=aplicar_deskew,
            aplicar_mejora_imagen=aplicar_mejora_imagen,
        )
    except BlankDocumentError as exc:
        raise HTTPException(422, str(exc)) from exc

    filename = (file.filename or "documento.pdf").rsplit(".", 1)[0] + "_normalizado.pdf"
    headers = {
        "X-Blank-Pages-Removed": str(len(result.blank_pages_original)),
        "X-Output-Pages": str(len(result.kept_original_pages)),
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(result.pdf_bytes, media_type="application/pdf", headers=headers)


@app.post("/detectar-pagares-actual")
async def detectar_pagares_actual(
    file: UploadFile = File(...),
    dpi: int = Form(default=160),
    solo_rangos: bool = Form(default=False),
    separar_qr: bool = Form(default=False),
    separar_barcode: bool = Form(default=True),
    eliminar_blancos: bool | None = Form(default=None),
    corregir_orientacion: bool | None = Form(default=None),
    aplicar_deskew: bool | None = Form(default=None),
    aplicar_mejora_imagen: bool | None = Form(default=None),
    incluir_pdf_normalizado: bool = Form(default=False),
):
    settings = get_settings()
    pdf_bytes = await _read_pdf_upload(file)
    safe_dpi = max(72, min(int(dpi or settings.default_dpi), 300))

    payload = detectar_pagares_actual_por_barcode(
        pdf_bytes=pdf_bytes,
        dpi=safe_dpi,
        solo_rangos=solo_rangos,
        separar_qr=separar_qr,
        separar_barcode=separar_barcode,
        eliminar_blancos=eliminar_blancos,
        corregir_orientacion=corregir_orientacion,
        aplicar_deskew=aplicar_deskew,
        aplicar_mejora_imagen=aplicar_mejora_imagen,
        settings=settings,
    )

    if payload.get("modo") == "blank_document":
        raise HTTPException(422, payload.get("error", "Documento en blanco"))

    if not incluir_pdf_normalizado:
        return payload

    try:
        normalized = normalizar_pdf(
            pdf_bytes,
            settings=settings,
            eliminar_blancos=_form_bool(eliminar_blancos, settings.eliminar_blancos),
            corregir_orientacion=_form_bool(corregir_orientacion, settings.corregir_orientacion),
            aplicar_deskew=_form_bool(aplicar_deskew, settings.aplicar_deskew),
            aplicar_mejora_imagen=_form_bool(aplicar_mejora_imagen, settings.aplicar_mejora_imagen),
        )
    except BlankDocumentError as exc:
        raise HTTPException(422, str(exc)) from exc

    import json

    from fastapi.responses import Response as FastAPIResponse

    boundary = "pagaresplit-boundary"
    json_part = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=utf-8\r\n\r\n"
    ).encode("utf-8") + json_part + (
        f"\r\n--{boundary}\r\n"
        "Content-Type: application/pdf\r\n"
        "Content-Disposition: attachment; filename=\"documento_normalizado.pdf\"\r\n\r\n"
    ).encode("utf-8") + normalized.pdf_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    return FastAPIResponse(
        content=body,
        media_type=f"multipart/mixed; boundary={boundary}",
    )
