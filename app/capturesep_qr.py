"""Detección rápida de hojas marcadora QR CAPTURESEP (CaptureSoft / PyVision)."""

from __future__ import annotations

import json
from typing import Any

import fitz
from PIL import Image

CAPTURESEP_PREFIX = "CAPTURESEP"

# Páginas de pagaré suelen superar este umbral; no llevan QR marcadora CAPTURESEP.
_MAX_TEXTO_PAGINA_SIN_QR = 480

# Escaneo por niveles: la hoja separadora tiene el QR centrado (Formas / módulo Separadores).
_QR_SCAN_TIERS: tuple[tuple[float, float], ...] = (
    (1.25, 0.50),  # ~5 ms/pág: recorte central, suficiente para hojas marcadora
    (1.75, 0.65),  # respaldo si el QR quedó más abajo por título largo
    (2.00, 1.00),  # último recurso: página completa (solo hojas marcadora probables)
)


def decode_capturesep_payload(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            cliente = str(data.get("cliente") or data.get("c") or data.get("client") or "").strip()
            separador = str(data.get("separador") or data.get("s") or data.get("separator") or "").strip()
            version = int(data.get("v") or data.get("version") or 1)
            sep_id_raw = data.get("separador_id") or data.get("sid") or data.get("separadorId")
            payload: dict[str, Any] = {"cliente": cliente or "QR", "separador": separador, "version": version}
            if sep_id_raw not in (None, ""):
                try:
                    sep_id = int(sep_id_raw)
                    if sep_id > 0:
                        payload["separador_id"] = sep_id
                except (TypeError, ValueError):
                    pass
            if payload.get("separador_id") or payload["separador"]:
                return payload

    parts = [p.strip() for p in text.split("|")]
    if len(parts) >= 4 and parts[0].upper() == CAPTURESEP_PREFIX:
        try:
            version = int(parts[1] or "1")
        except ValueError:
            version = 1
        cliente = parts[2]
        if version >= 2:
            try:
                sep_id = int(parts[3])
            except ValueError:
                sep_id = 0
            if cliente and sep_id > 0:
                return {"cliente": cliente, "separador": "", "version": version, "separador_id": sep_id}
            return None
        separador = "|".join(parts[3:]).strip()
        if cliente and separador:
            return {"cliente": cliente, "separador": separador, "version": version}
        return None

    if len(parts) == 2 and parts[0] and parts[1]:
        return {"cliente": parts[0], "separador": parts[1], "version": 1}

    return {"cliente": "QR", "separador": text[:180] or "QR", "version": 1}


def _limpiar_texto_qr(texto: str) -> str:
    if not texto:
        return texto
    if "\x00" in texto:
        texto = texto.replace("\x00", "")
    return texto.strip()


def _recorte_central(image: Image.Image, fraccion: float) -> Image.Image:
    if fraccion >= 0.999:
        return image
    w, h = image.size
    cw = max(32, int(w * fraccion))
    ch = max(32, int(h * fraccion))
    left = (w - cw) // 2
    top = (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


def _leer_qr_zxing(
    image: Image.Image,
    *,
    try_rotate: bool = False,
    try_downscale: bool = False,
    try_invert: bool = False,
) -> str | None:
    try:
        import zxingcpp
    except ImportError:
        return None

    for barcode in zxingcpp.read_barcodes(
        image,
        formats=zxingcpp.BarcodeFormat.QRCode,
        try_rotate=try_rotate,
        try_downscale=try_downscale,
        try_invert=try_invert,
        return_errors=False,
    ):
        text = (getattr(barcode, "text", "") or "").strip()
        if text:
            return text
    return None


def _render_pagina_rgb(page: fitz.Page, scale: float) -> Image.Image:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _texto_pagina(page: fitz.Page) -> str:
    return page.get_text("text").strip()


def _es_raw_capturesep(raw: str) -> bool:
    text = (raw or "").strip()
    if not text:
        return False
    if text.upper().startswith(CAPTURESEP_PREFIX):
        return True
    if text.startswith("{"):
        return decode_capturesep_payload(text) is not None
    return decode_capturesep_payload(text) is not None


def _parece_hoja_marcadora_qr(page: fitz.Page, text: str | None = None) -> bool:
    """Heurística para no gastar tiers costosos en portadas de pagaré con logos."""
    text = text if text is not None else _texto_pagina(page)
    text_len = len(text)
    if text_len >= _MAX_TEXTO_PAGINA_SIN_QR:
        return False
    upper = text.upper()
    if "CAPTURESEP" in upper or "SEPARADOR" in upper or "INSERTAR ENTRE" in upper:
        return True
    images = page.get_images(full=True)
    if not images:
        return 0 < text_len < 220
    if text_len <= 120:
        return True
    for img in images:
        width = int(img[2] or 0)
        height = int(img[3] or 0)
        if width < 80 or height < 80:
            continue
        ratio = width / max(height, 1)
        if 0.75 <= ratio <= 1.33 and min(width, height) >= 180:
            return True
    return text_len < 180


def _qr_desde_imagenes_embebidas(doc: fitz.Document, page: fitz.Page) -> str | None:
    """Lee el QR desde la imagen embebida (hoja separadora generada por CaptureSoft)."""
    images = page.get_images(full=True)
    if not images:
        return None

    ranked: list[tuple[int, int]] = []
    for img in images:
        xref = int(img[0])
        width = int(img[2] or 0)
        height = int(img[3] or 0)
        if width < 64 or height < 64:
            continue
        ranked.append((width * height, xref))
    ranked.sort(reverse=True)

    for _, xref in ranked[:4]:
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception:
            continue

        side = max(pil.size)
        if side > 640:
            ratio = 640 / side
            pil = pil.resize(
                (max(32, int(pil.width * ratio)), max(32, int(pil.height * ratio))),
                Image.Resampling.BILINEAR,
            )

        for try_rotate, try_downscale, try_invert in (
            (False, False, False),
            (True, True, True),
        ):
            raw = _leer_qr_zxing(
                pil,
                try_rotate=try_rotate,
                try_downscale=try_downscale,
                try_invert=try_invert,
            )
            if raw and _es_raw_capturesep(raw):
                return _limpiar_texto_qr(raw)
    return None


def _detectar_qr_en_pagina(page: fitz.Page, doc: fitz.Document | None = None) -> str | None:
    """Busca CAPTURESEP: imagen embebida → recorte central → tiers profundos solo en marcadora."""
    text = _texto_pagina(page)
    if len(text) >= _MAX_TEXTO_PAGINA_SIN_QR:
        return None

    probable_marcadora = _parece_hoja_marcadora_qr(page, text)

    if doc is not None and probable_marcadora:
        raw = _qr_desde_imagenes_embebidas(doc, page)
        if raw:
            return raw

    for tier_idx, (scale, center_frac) in enumerate(_QR_SCAN_TIERS):
        if tier_idx > 0 and not probable_marcadora:
            break
        try:
            image = _render_pagina_rgb(page, scale)
            if center_frac < 1.0:
                image = _recorte_central(image, center_frac)
        except Exception:
            continue

        raw = _leer_qr_zxing(
            image,
            try_rotate=tier_idx >= 2,
            try_downscale=tier_idx >= 1,
            try_invert=tier_idx >= 1,
        )
        if raw and _es_raw_capturesep(raw):
            return _limpiar_texto_qr(raw)
    return None


def detectar_marcadores_qr_capturesep(
    *,
    pdf_bytes: bytes,
    max_pages: int = 5000,
    scale: float | None = None,  # ignorado: se mantiene por compatibilidad de firma
) -> list[dict[str, Any]]:
    """Devuelve marcadores CAPTURESEP por página (1-based)."""
    del scale  # niveles fijos optimizados para hojas marcadora
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    marcadores: list[dict[str, Any]] = []
    try:
        cap = min(len(doc), max(1, max_pages))
        for idx in range(cap):
            page_num = idx + 1
            page = doc.load_page(idx)
            if len(_texto_pagina(page)) >= _MAX_TEXTO_PAGINA_SIN_QR:
                continue
            raw = _detectar_qr_en_pagina(page, doc)
            if not raw:
                continue
            payload = decode_capturesep_payload(raw)
            if payload:
                marcadores.append({"pagina_1_based": page_num, "payload": payload, "raw": raw})
    finally:
        doc.close()

    marcadores.sort(key=lambda m: m["pagina_1_based"])
    return marcadores


def segmentos_entre_paginas_qr(total_pages: int, paginas_qr: list[int]) -> list[list[int]]:
    """Páginas de contenido entre hojas QR (excluye las hojas marcadora)."""
    if total_pages <= 0:
        return []
    separadores = sorted({p for p in paginas_qr if 1 <= p <= total_pages})
    if not separadores:
        return [list(range(1, total_pages + 1))]

    segmentos: list[list[int]] = []
    start = 1
    for sep_page in separadores:
        end = sep_page - 1
        if end >= start:
            segmentos.append(list(range(start, end + 1)))
        start = sep_page + 1
    if start <= total_pages:
        segmentos.append(list(range(start, total_pages + 1)))
    return [seg for seg in segmentos if seg]


def pdf_solo_hojas_qr(total_pages: int, paginas_qr: list[int]) -> bool:
    """True si todas las páginas del PDF son hojas marcadora QR."""
    if total_pages <= 0:
        return False
    qr_set = {p for p in paginas_qr if 1 <= p <= total_pages}
    return len(qr_set) == total_pages
