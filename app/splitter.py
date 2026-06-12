"""Detección liviana de pagarés formato actual por layout + Code39 + QR CAPTURESEP."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image, ImageOps

from app.barcode_pdf import barcodes_pdf_en_memoria
from app.capturesep_qr import (
    detectar_marcadores_qr_capturesep,
    pdf_solo_hojas_qr,
    segmentos_entre_paginas_qr,
)

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


def _extraer_paginas_pdf(pdf_bytes: bytes, pages_1based: list[int]) -> bytes:
    if not pages_1based:
        return pdf_bytes
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    dst = fitz.open()
    try:
        for page_num in pages_1based:
            idx = page_num - 1
            if 0 <= idx < len(src):
                dst.insert_pdf(src, from_page=idx, to_page=idx)
        return dst.tobytes()
    finally:
        src.close()
        dst.close()


def _remapear_pagares_a_original(pagares: list[dict[str, Any]], pages_1based: list[int]) -> list[dict[str, Any]]:
    page_map = {subset_idx + 1: orig for subset_idx, orig in enumerate(pages_1based)}
    remapped: list[dict[str, Any]] = []
    for item in pagares:
        paginas_orig = [page_map[p] for p in item.get("paginas", []) if p in page_map]
        if not paginas_orig:
            continue
        remapped.append(
            {
                **item,
                "pagina_inicio": paginas_orig[0],
                "pagina_fin": paginas_orig[-1],
                "paginas": paginas_orig,
                "n_hojas": len(paginas_orig),
            }
        )
    return remapped


def _detectar_pagares_actual_por_barcode_core(
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


def detectar_pagares_actual_por_barcode(
    *,
    pdf_path: Path | None = None,
    pdf_bytes: bytes | None = None,
    dpi: int = 160,
    solo_rangos: bool = False,
    separar_qr: bool = False,
    separar_barcode: bool = True,
) -> dict[str, Any]:
    if pdf_bytes is None and pdf_path is not None:
        pdf_bytes = pdf_path.read_bytes()
    if pdf_bytes is None:
        return {
            "total_paginas": 0,
            "total_pagares": 0,
            "pagares": [],
            "modo": "sin_pdf",
        }

    marcadores_qr: list[dict[str, Any]] = []
    if separar_qr:
        marcadores_qr = detectar_marcadores_qr_capturesep(pdf_bytes=pdf_bytes)

    if separar_qr and not marcadores_qr and not separar_barcode:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()
        paginas = list(range(1, total_pages + 1)) if total_pages > 0 else []
        pagares = []
        if paginas:
            pagares = [
                {
                    "indice": 1,
                    "pagina_inicio": 1,
                    "pagina_fin": total_pages,
                    "codigo_operacion": None,
                    "paginas": paginas,
                    "n_hojas": len(paginas),
                }
            ]
        return {
            "total_paginas": total_pages,
            "total_pagares": len(pagares),
            "pagares": pagares,
            "modo": "capturesep_qr_sin_marcadores",
            "dpi_usado": dpi,
            "marcadores_qr": [],
            "paginas_qr": [],
        }

    if marcadores_qr:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()
        paginas_qr = [int(m["pagina_1_based"]) for m in marcadores_qr]

        if pdf_solo_hojas_qr(total_pages, paginas_qr):
            return {
                "total_paginas": total_pages,
                "total_pagares": 0,
                "pagares": [],
                "modo": "capturesep_qr_solo_marcadores",
                "dpi_usado": dpi,
                "marcadores_qr": marcadores_qr,
                "paginas_qr": paginas_qr,
            }

        segmentos = segmentos_entre_paginas_qr(total_pages, paginas_qr)

        pagares: list[dict[str, Any]] = []
        modo_partes: list[str] = ["capturesep_qr"]

        for segmento in segmentos:
            if not segmento:
                continue
            if separar_barcode:
                subset_bytes = _extraer_paginas_pdf(pdf_bytes, segmento)
                sub = _detectar_pagares_actual_por_barcode_core(
                    pdf_bytes=subset_bytes,
                    dpi=dpi,
                    solo_rangos=solo_rangos,
                )
                sub_pagares = _remapear_pagares_a_original(sub.get("pagares", []), segmento)
                if sub_pagares:
                    pagares.extend(sub_pagares)
                    if sub.get("modo"):
                        modo_partes.append(str(sub["modo"]))
                    continue

            pagares.append(
                {
                    "indice": 0,
                    "pagina_inicio": segmento[0],
                    "pagina_fin": segmento[-1],
                    "codigo_operacion": None,
                    "paginas": segmento,
                    "n_hojas": len(segmento),
                }
            )

        for i, pagare in enumerate(pagares, start=1):
            pagare["indice"] = i

        modo = "+".join(dict.fromkeys(modo_partes))
        return {
            "total_paginas": total_pages,
            "total_pagares": len(pagares),
            "pagares": pagares,
            "modo": modo,
            "dpi_usado": dpi,
            "marcadores_qr": marcadores_qr,
            "paginas_qr": paginas_qr,
        }

    result = _detectar_pagares_actual_por_barcode_core(
        pdf_path=pdf_path,
        pdf_bytes=pdf_bytes,
        dpi=dpi,
        solo_rangos=solo_rangos,
    )
    if separar_qr:
        result["marcadores_qr"] = []
        result["paginas_qr"] = []
    return result
