"""Detección liviana de pagarés formato actual por layout + Code39."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image, ImageOps

from app.barcode_pdf import barcodes_pdf_en_memoria

BARCODE_SCAN_DPI_FALLBACKS = (160, 200)
LAYOUT_START_SIM_THRESHOLD = 0.45


def _digits(text: str) -> str:
    return re.sub(r"\D+", "", text or "")


def _barcode_valido(fmt: str, digits: str) -> bool:
    if len(digits) < 6:
        return False
    fmt_u = (fmt or "").upper()
    if fmt_u and "CODE" not in fmt_u and "39" not in fmt_u:
        return False
    return True


def _paginas_con_barcode_valido(por_pagina: list[list[dict[str, Any]]]) -> list[tuple[int, str]]:
    starts: list[tuple[int, str]] = []
    seen: set[str] = set()

    for page_idx, barcodes in enumerate(por_pagina, start=1):
        for bc in barcodes:
            digits = _digits(str(bc.get("texto") or ""))
            fmt = str(bc.get("formato") or "")
            if not _barcode_valido(fmt, digits) or digits in seen:
                continue
            seen.add(digits)
            starts.append((page_idx, digits))
            break

    starts.sort(key=lambda x: x[0])
    return starts


def _merge_barcodes_por_pagina(
    base: list[list[dict[str, Any]]],
    extra: list[list[dict[str, Any]]],
) -> list[list[dict[str, Any]]]:
    total = max(len(base), len(extra))
    merged: list[list[dict[str, Any]]] = []
    for i in range(total):
        page_items: list[dict[str, Any]] = []
        seen = set()
        for source in (base, extra):
            if i >= len(source):
                continue
            for bc in source[i]:
                key = (
                    str(bc.get("formato") or ""),
                    str(bc.get("texto") or ""),
                    str(bc.get("region") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                page_items.append(bc)
        merged.append(page_items)
    return merged


def _paginas_inicio_por_layout(
    *,
    pdf_path: Path | None = None,
    pdf_bytes: bytes | None = None,
    threshold: float = LAYOUT_START_SIM_THRESHOLD,
) -> tuple[list[int], list[dict[str, Any]]]:
    if pdf_bytes is not None:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elif pdf_path is not None:
        doc = fitz.open(pdf_path)
    else:
        return [], []

    perfiles: list[Any] = []
    scores: list[dict[str, Any]] = []
    try:
        mat = fitz.Matrix(1, 1)
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).resize((220, 310))
            gray = ImageOps.grayscale(img)
            arr = np.array(gray, dtype=float)[:170, :]
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            perfiles.append(arr.ravel())
    finally:
        doc.close()

    if not perfiles:
        return [], []

    ref = perfiles[0]
    starts = [1]
    for idx, perfil in enumerate(perfiles, start=1):
        sim = float(np.mean(ref * perfil))
        scores.append({"page": idx, "similarity": round(sim, 4), "inicioLayout": idx == 1 or sim >= threshold})
        if idx == 1:
            continue
        if sim >= threshold and idx - starts[-1] >= 2:
            starts.append(idx)

    return starts, scores


def _codigo_por_pagina_desde_barcodes(por_pagina: list[list[dict[str, Any]]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for page_idx, barcodes in enumerate(por_pagina, start=1):
        for bc in barcodes:
            digits = _digits(str(bc.get("texto") or ""))
            fmt = str(bc.get("formato") or "")
            if not _barcode_valido(fmt, digits):
                continue
            out[page_idx] = digits
            break
    return out


def _codigos_unicos(barcodes: list[dict[str, Any]]) -> list[str]:
    codigos: list[str] = []
    seen: set[str] = set()
    for bc in barcodes:
        digits = _digits(str(bc.get("texto") or ""))
        if not digits or digits in seen:
            continue
        seen.add(digits)
        codigos.append(digits)
    return codigos


def detectar_pagares_actual_por_barcode(
    *,
    pdf_path: Path | None = None,
    pdf_bytes: bytes | None = None,
    dpi: int = 160,
    solo_rangos: bool = False,
) -> dict[str, Any]:
    dpi_inicial = max(72, min(int(dpi or 160), 300))
    layout_starts, layout_scores = _paginas_inicio_por_layout(pdf_path=pdf_path, pdf_bytes=pdf_bytes)
    total_pages = len(layout_scores)
    dpis_usados = [dpi_inicial]
    uso_layout = len(layout_starts) >= 2

    if uso_layout:
        paginas_barcode = {
            p
            for start in layout_starts
            for p in (start, start + 1, start + 2)
            if 1 <= p <= total_pages
        }
        if solo_rangos:
            por_pagina = [[] for _ in range(total_pages)]
            codigo_por_pagina: dict[int, str] = {}
        else:
            por_pagina = barcodes_pdf_en_memoria(
                pdf_path=pdf_path,
                pdf_bytes=pdf_bytes,
                dpi=dpi_inicial,
                pages_1based=paginas_barcode,
            )
            codigo_por_pagina = _codigo_por_pagina_desde_barcodes(por_pagina)
            if any(page not in codigo_por_pagina for page in layout_starts):
                for dpi_retry in BARCODE_SCAN_DPI_FALLBACKS:
                    if dpi_retry <= dpi_inicial or dpi_retry in dpis_usados:
                        continue
                    extra = barcodes_pdf_en_memoria(
                        pdf_path=pdf_path,
                        pdf_bytes=pdf_bytes,
                        dpi=dpi_retry,
                        pages_1based=paginas_barcode,
                    )
                    dpis_usados.append(dpi_retry)
                    por_pagina = _merge_barcodes_por_pagina(por_pagina, extra)
                    codigo_por_pagina = _codigo_por_pagina_desde_barcodes(por_pagina)
                    if all(page in codigo_por_pagina for page in layout_starts):
                        break
        starts: list[tuple[int, str | None]] = [(page, codigo_por_pagina.get(page)) for page in layout_starts]
    elif solo_rangos and total_pages > 0:
        por_pagina = [[] for _ in range(total_pages)]
        codigo_por_pagina = {}
        starts = [(1, None)]
    else:
        por_pagina = barcodes_pdf_en_memoria(pdf_path=pdf_path, pdf_bytes=pdf_bytes, dpi=dpi_inicial)
        total_pages = len(por_pagina)
        starts = [(p, code) for p, code in _paginas_con_barcode_valido(por_pagina)]

        if total_pages > 1 and len(starts) <= 1:
            for dpi_retry in BARCODE_SCAN_DPI_FALLBACKS:
                if dpi_retry <= dpi_inicial or dpi_retry in dpis_usados:
                    continue
                extra = barcodes_pdf_en_memoria(pdf_path=pdf_path, pdf_bytes=pdf_bytes, dpi=dpi_retry)
                dpis_usados.append(dpi_retry)
                por_pagina = _merge_barcodes_por_pagina(por_pagina, extra)
                starts = [(p, code) for p, code in _paginas_con_barcode_valido(por_pagina)]
                if len(starts) > 1:
                    break
        codigo_por_pagina = _codigo_por_pagina_desde_barcodes(por_pagina)

    pagares: list[dict[str, Any]] = []
    for i, (start, code) in enumerate(starts):
        next_start = starts[i + 1][0] if i + 1 < len(starts) else total_pages + 1
        end = max(start, next_start - 1)
        paginas = list(range(start, end + 1))
        codigo_operacion = code
        if not codigo_operacion:
            for p in paginas:
                if codigo_por_pagina.get(p):
                    codigo_operacion = codigo_por_pagina[p]
                    break
        pagares.append(
            {
                "indice": i + 1,
                "pagina_inicio": start,
                "pagina_fin": end,
                "codigo_operacion": codigo_operacion,
                "paginas": paginas,
                "n_hojas": len(paginas),
            }
        )

    return {
        "total_paginas": total_pages,
        "total_pagares": len(pagares),
        "pagares": pagares,
        "modo": (
            "layout_portada_rapido"
            if uso_layout and solo_rangos
            else ("layout_unico_rapido" if solo_rangos else ("layout_portada_barcode_code39" if uso_layout else "barcode_code39"))
        ),
        "dpi_usado": max(dpis_usados) if dpis_usados else dpi_inicial,
        "layout_por_pagina": layout_scores,
        "barcodes_por_pagina": [
            {
                "page": idx,
                "count": len(items),
                "codigos": _codigos_unicos(items),
            }
            for idx, items in enumerate(por_pagina, start=1)
        ],
    }
