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
LAYOUT_START_SIM_THRESHOLD_RELAXED = 0.35
LAYOUT_START_MIN_GAP = 2


def _starts_con_prefijo_huerfano(starts: list[tuple[int, str | None]]) -> list[tuple[int, str | None]]:
    if starts and starts[0][0] > 1:
        return [(1, None), *starts]
    return starts


def _filtrar_starts_con_barcode_obligatorio(
    starts: list[tuple[int, str | None]],
    *,
    solo_rangos: bool,
) -> list[tuple[int, str | None]]:
    if solo_rangos:
        return starts
    return [(page, code) for page, code in starts if code]


def _filtrar_pagares_con_barcode_obligatorio(
    pagares: list[dict[str, Any]],
    *,
    solo_rangos: bool,
) -> list[dict[str, Any]]:
    if solo_rangos:
        return pagares
    filtered = [p for p in pagares if str(p.get("codigo_operacion") or "").strip()]
    for i, pagare in enumerate(filtered, start=1):
        pagare["indice"] = i
    return filtered


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


def _similitud_por_pagina(layout_scores: list[dict[str, Any]]) -> dict[int, float]:
    return {int(item["page"]): float(item["similarity"]) for item in layout_scores}


def _layout_starts_desde_scores(
    layout_scores: list[dict[str, Any]],
    threshold: float,
    *,
    min_gap: int = LAYOUT_START_MIN_GAP,
) -> list[int]:
    starts = [1]
    for item in layout_scores:
        idx = int(item["page"])
        if idx == 1:
            continue
        if float(item["similarity"]) >= threshold and idx - starts[-1] >= min_gap:
            starts.append(idx)
    return starts


def _codigo_operacion_para_inicio(
    page: int,
    codigo_por_pagina: dict[int, str],
    *,
    total_pages: int,
    ventana: int = 2,
) -> str | None:
    directo = codigo_por_pagina.get(page)
    if directo:
        return directo
    for offset in range(1, ventana + 1):
        for candidata in (page + offset, page - offset):
            if 1 <= candidata <= total_pages:
                codigo = codigo_por_pagina.get(candidata)
                if codigo:
                    return codigo
    return None


def _corte_barcode_sospechoso(
    starts: list[tuple[int, str | None]],
    layout_scores: list[dict[str, Any]],
) -> bool:
    if len(starts) < 2 or not layout_scores:
        return False
    sim = _similitud_por_pagina(layout_scores)
    page2 = starts[1][0]
    # Barcode distinto muy pronto en una hoja que no parece portada (p. ej. desembolso).
    return page2 <= 5 and sim.get(page2, 0.0) < LAYOUT_START_SIM_THRESHOLD_RELAXED


def _refinar_starts_barcode_con_layout(
    starts: list[tuple[int, str | None]],
    layout_scores: list[dict[str, Any]],
    codigo_por_pagina: dict[int, str],
    *,
    total_pages: int,
) -> tuple[list[tuple[int, str | None]], bool]:
    if not _corte_barcode_sospechoso(starts, layout_scores):
        return starts, False
    relaxed = _layout_starts_desde_scores(layout_scores, LAYOUT_START_SIM_THRESHOLD_RELAXED)
    if len(relaxed) < 2:
        return starts, False
    refinados = [
        (page, _codigo_operacion_para_inicio(page, codigo_por_pagina, total_pages=total_pages))
        for page in relaxed
    ]
    return refinados, True


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


def _pagare_dict(
    *,
    start: int,
    end: int,
    codigo_operacion: str | None,
    paginas: list[int],
) -> dict[str, Any]:
    return {
        "pagina_inicio": start,
        "pagina_fin": end,
        "codigo_operacion": codigo_operacion,
        "paginas": paginas,
        "n_hojas": len(paginas),
    }


def _partir_pagares_por_reaparicion_codigo(
    pagares: list[dict[str, Any]],
    codigo_por_pagina: dict[int, str],
) -> tuple[list[dict[str, Any]], bool]:
    """Corta un bloque cuando reaparece el barcode de una operación ya segmentada."""
    if not pagares:
        return pagares, False

    resultado: list[dict[str, Any]] = []
    recortado = False

    for item in pagares:
        start = int(item["pagina_inicio"])
        end = int(item["pagina_fin"])
        codigo_actual = str(item.get("codigo_operacion") or "").strip() or None
        codigos_previos = {
            str(p.get("codigo_operacion") or "").strip()
            for p in resultado
            if str(p.get("codigo_operacion") or "").strip()
        }

        corte: int | None = None
        codigo_corte: str | None = None
        for page in range(start, end + 1):
            code_page = codigo_por_pagina.get(page)
            if code_page and code_page in codigos_previos:
                corte = page
                codigo_corte = code_page
                break

        if corte is None:
            paginas = list(range(start, end + 1))
            codigo_bloque = codigo_actual
            if not codigo_bloque:
                for page in paginas:
                    if codigo_por_pagina.get(page):
                        codigo_bloque = codigo_por_pagina[page]
                        break
            resultado.append(_pagare_dict(start=start, end=end, codigo_operacion=codigo_bloque, paginas=paginas))
            continue

        if corte > start:
            paginas = list(range(start, corte))
            codigo_bloque = codigo_actual
            if not codigo_bloque:
                for page in paginas:
                    if codigo_por_pagina.get(page):
                        codigo_bloque = codigo_por_pagina[page]
                        break
            resultado.append(_pagare_dict(start=start, end=corte - 1, codigo_operacion=codigo_bloque, paginas=paginas))
            recortado = True

        paginas_cola = list(range(corte, end + 1))
        resultado.append(
            _pagare_dict(
                start=corte,
                end=end,
                codigo_operacion=codigo_corte or codigo_por_pagina.get(corte),
                paginas=paginas_cola,
            )
        )
        recortado = True

    for idx, pagare in enumerate(resultado, start=1):
        pagare["indice"] = idx
    return resultado, recortado


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


def _excluir_paginas_qr_de_pagares(
    pagares: list[dict[str, Any]],
    paginas_qr: list[int],
) -> list[dict[str, Any]]:
    """Garantiza que las hojas marcadora QR no aparezcan en los rangos de salida."""
    qr_set = {int(p) for p in paginas_qr if p is not None}
    qr_set = {p for p in qr_set if p >= 1}
    if not qr_set:
        return pagares

    limpios: list[dict[str, Any]] = []
    for item in pagares:
        paginas = [p for p in item.get("paginas", []) if int(p) not in qr_set]
        if not paginas:
            continue
        limpios.append(
            {
                **item,
                "pagina_inicio": paginas[0],
                "pagina_fin": paginas[-1],
                "paginas": paginas,
                "n_hojas": len(paginas),
            }
        )
    for i, pagare in enumerate(limpios, start=1):
        pagare["indice"] = i
    return limpios


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
    refinado_layout = False

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
        starts, refinado_layout = _refinar_starts_barcode_con_layout(
            starts,
            layout_scores,
            codigo_por_pagina,
            total_pages=total_pages,
        )
        if refinado_layout:
            uso_layout = True

    if solo_rangos:
        starts = _starts_con_prefijo_huerfano(starts)
    else:
        starts = _filtrar_starts_con_barcode_obligatorio(starts, solo_rangos=False)

    pagares: list[dict[str, Any]] = []
    for i, (start, code) in enumerate(starts):
        next_start = starts[i + 1][0] if i + 1 < len(starts) else total_pages + 1
        end = max(start, next_start - 1)
        paginas = list(range(start, end + 1))
        codigo_operacion = code
        if not codigo_operacion and solo_rangos:
            for p in paginas:
                if codigo_por_pagina.get(p):
                    codigo_operacion = codigo_por_pagina[p]
                    break
        pagares.append(
            _pagare_dict(
                start=start,
                end=end,
                codigo_operacion=codigo_operacion,
                paginas=paginas,
            )
        )

    pagares, recortado_reaparicion = _partir_pagares_por_reaparicion_codigo(pagares, codigo_por_pagina)
    pagares = _filtrar_pagares_con_barcode_obligatorio(pagares, solo_rangos=solo_rangos)
    modo_base = (
        "layout_portada_rapido"
        if uso_layout and solo_rangos
        else (
            "layout_unico_rapido"
            if solo_rangos
            else (
                "layout_refinado_barcode_code39"
                if refinado_layout
                else ("layout_portada_barcode_code39" if uso_layout else "barcode_code39")
            )
        )
    )
    if recortado_reaparicion:
        modo_base = f"{modo_base}+recorte_reaparicion"

    result = {
        "total_paginas": total_pages,
        "total_pagares": len(pagares),
        "pagares": pagares,
        "modo": modo_base,
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
    return result


def _forward_fill_codigo_por_pagina(codigo_por_pagina: dict[int, str], total_pages: int) -> dict[int, str]:
    filled = dict(codigo_por_pagina)
    last_code: str | None = None
    for page in range(1, total_pages + 1):
        code = filled.get(page)
        if code:
            last_code = code
            continue
        if last_code:
            filled[page] = last_code
    return filled


def _bloques_no_contiguos(pages: list[int]) -> bool:
    if len(pages) <= 1:
        return False
    for i in range(1, len(pages)):
        if pages[i] != pages[i - 1] + 1:
            return True
    return False


def _detectar_pdf_intercalado(codigo_por_pagina: dict[int, str], total_pages: int) -> tuple[bool, dict[str, list[int]]]:
    if total_pages <= 0 or len(codigo_por_pagina) < 2:
        return False, {}

    filled = _forward_fill_codigo_por_pagina(codigo_por_pagina, total_pages)
    pages_by_code: dict[str, list[int]] = {}
    for page, code in sorted(filled.items()):
        pages_by_code.setdefault(code, []).append(page)

    if len(pages_by_code) < 2:
        return False, pages_by_code

    for pages in pages_by_code.values():
        if _bloques_no_contiguos(pages):
            return True, pages_by_code

    previous: str | None = None
    seen_since_change: set[str] = set()
    for page in range(1, total_pages + 1):
        code = filled.get(page)
        if not code:
            continue
        if previous and code != previous and code in seen_since_change:
            return True, pages_by_code
        if previous and code != previous:
            seen_since_change.add(previous)
        previous = code

    return False, pages_by_code


def _mensaje_pdf_intercalado(pages_by_code: dict[str, list[int]]) -> str:
    parts: list[str] = []
    for code, pages in sorted(pages_by_code.items(), key=lambda item: item[1][0]):
        if not pages:
            continue
        ranges: list[str] = []
        start = pages[0]
        end = pages[0]
        for page in pages[1:]:
            if page == end + 1:
                end = page
                continue
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = page
            end = page
        ranges.append(str(start) if start == end else f"{start}-{end}")
        parts.append(f"{code}: pág. {','.join(ranges)}")
    detail = "; ".join(parts)
    return f"PDF mal ordenado: operaciones intercaladas ({detail}). Re-escanee separando cada pagaré."


def validar_orden_pdf_sucursales(
    *,
    pdf_path: Path | None = None,
    pdf_bytes: bytes | None = None,
    dpi: int = 160,
) -> dict[str, Any]:
    """Valida orden de operaciones para flujo sucursales (DocNative)."""
    dpi_inicial = max(72, min(int(dpi or 160), 300))
    por_pagina = barcodes_pdf_en_memoria(pdf_path=pdf_path, pdf_bytes=pdf_bytes, dpi=dpi_inicial)
    total_pages = len(por_pagina)
    codigo_por_pagina = _codigo_por_pagina_desde_barcodes(por_pagina)

    if total_pages > 1 and len(codigo_por_pagina) < 2:
        if len(codigo_por_pagina) == 0:
            # Sin códigos no puede haber intercalado (requiere >=2 operaciones distintas).
            return {
                "total_paginas": total_pages,
                "intercalado": False,
                "codigos_detectados": [],
                "paginas_por_codigo": {},
                "mensaje": None,
            }

        for dpi_retry in BARCODE_SCAN_DPI_FALLBACKS:
            if dpi_retry <= dpi_inicial:
                continue
            extra = barcodes_pdf_en_memoria(
                pdf_path=pdf_path,
                pdf_bytes=pdf_bytes,
                dpi=dpi_retry,
            )
            por_pagina = _merge_barcodes_por_pagina(por_pagina, extra)
            codigo_por_pagina = _codigo_por_pagina_desde_barcodes(por_pagina)
            if len(codigo_por_pagina) >= 2:
                break

    intercalado, pages_by_code = _detectar_pdf_intercalado(codigo_por_pagina, total_pages)
    return {
        "total_paginas": total_pages,
        "intercalado": intercalado,
        "codigos_detectados": sorted(pages_by_code.keys()),
        "paginas_por_codigo": pages_by_code,
        "mensaje": _mensaje_pdf_intercalado(pages_by_code) if intercalado else None,
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
        if paginas and solo_rangos:
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

            if not solo_rangos:
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

        pagares = _excluir_paginas_qr_de_pagares(pagares, paginas_qr)
        pagares = _filtrar_pagares_con_barcode_obligatorio(pagares, solo_rangos=solo_rangos)

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
