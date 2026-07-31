from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# zxing-cpp falla en imágenes full-page RGB muy grandes (~>45M px).
_MAX_FULL_RAW_PIXELS = 45_000_000


def _barcode_bbox(position: Any) -> list[float]:
    if not position:
        return []
    bbox: list[float] = []
    for point_name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        point = getattr(position, point_name, None)
        if point is not None:
            bbox.extend([float(point.x), float(point.y)])
    return bbox


def _barcode_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    width, height = image.size
    top25 = max(1, int(height * 0.25))
    top35 = max(1, int(height * 0.35))
    top50 = max(1, int(height * 0.50))
    half_w = max(1, int(width * 0.50))
    crops = [
        ("full", image),
        ("top_25", image.crop((0, 0, width, top25))),
        ("top_35", image.crop((0, 0, width, top35))),
        ("top_50", image.crop((0, 0, width, top50))),
        ("top_50_left", image.crop((0, 0, half_w, top50))),
        ("top_50_right", image.crop((half_w, 0, width, top50))),
    ]
    variants: list[tuple[str, Image.Image]] = []
    seen: set[str] = set()
    for name, crop in crops:
        prepared = [("raw", crop)]
        gray = ImageOps.grayscale(crop)
        auto = ImageOps.autocontrast(gray)
        sharp = ImageEnhance.Sharpness(ImageEnhance.Contrast(auto).enhance(1.8)).enhance(2.0)
        bw = sharp.point(lambda p: 255 if p > 150 else 0).convert("L")
        prepared.extend(
            [
                ("autocontrast", auto),
                ("sharp", sharp.filter(ImageFilter.SHARPEN)),
                ("bw", bw),
            ]
        )
        for suffix, variant in prepared:
            key = f"{name}:{suffix}:{variant.size[0]}x{variant.size[1]}"
            if key in seen:
                continue
            seen.add(key)
            variants.append((key, variant))
    return variants


def _read_barcodes_safe(variant: Image.Image) -> Iterable[Any]:
    try:
        import zxingcpp
    except ImportError:
        return []

    source = np.asarray(variant)
    try:
        return zxingcpp.read_barcodes(
            source,
            try_rotate=True,
            try_downscale=True,
            return_errors=False,
        )
    except (TypeError, MemoryError, ValueError):
        return []


def barcodes_pdf_en_memoria(
    *,
    pdf_path: Path | None = None,
    pdf_bytes: bytes | None = None,
    dpi: int = 160,
    pages_1based: set[int] | None = None,
) -> list[list[dict[str, Any]]]:
    try:
        import zxingcpp
    except ImportError:
        return []

    if pdf_bytes is not None:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    elif pdf_path is not None:
        doc = fitz.open(pdf_path)
    else:
        return []

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    por_pagina: list[list[dict[str, Any]]] = []
    try:
        for i in range(len(doc)):
            page_num = i + 1
            if pages_1based is not None and page_num not in pages_1based:
                por_pagina.append([])
                continue

            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            found: list[dict[str, Any]] = []
            seen_found: set[tuple[str, str]] = set()
            page_pixels = image.size[0] * image.size[1]

            for region_name, variant in _barcode_variants(image):
                if region_name.startswith("full:raw") and page_pixels > _MAX_FULL_RAW_PIXELS:
                    continue
                for barcode in _read_barcodes_safe(variant):
                    text = (getattr(barcode, "text", "") or "").strip()
                    if not text:
                        continue
                    fmt = str(getattr(barcode, "format", "")).replace("BarcodeFormat.", "")
                    found_key = (fmt, text)
                    if found_key in seen_found:
                        continue
                    seen_found.add(found_key)
                    found.append(
                        {
                            "texto": text,
                            "formato": fmt,
                            "bbox": _barcode_bbox(getattr(barcode, "position", None)),
                            "region": region_name,
                            "pagina": f"pagina_{page_num:03d}.png",
                        }
                    )
            por_pagina.append(found)
    finally:
        doc.close()
    return por_pagina
