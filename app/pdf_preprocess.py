"""Normaliza PDF escaneado: quita blancos, corrige orientación y deskew."""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import fitz
import numpy as np
from PIL import Image

from app.blank_detection import is_blank_page_rgb
from app.config import Settings, get_settings
from app.scan_orientation import (
    BlankDocumentError,
    PageTransform,
    analyze_page_orientation,
    apply_page_transform,
    build_reference_profile,
)
from app.scan_enhance import enhance_scanned_page


@dataclass
class PreprocessResult:
    pdf_bytes: bytes
    kept_original_pages: list[int]
    blank_pages_original: list[int] = field(default_factory=list)
    transforms: list[PageTransform] = field(default_factory=list)

    @property
    def original_to_normalized(self) -> dict[int, int]:
        return {orig: idx for idx, orig in enumerate(self.kept_original_pages, start=1)}


@dataclass
class _PagePlan:
    orig_page: int
    is_blank: bool
    coarse: int = 0
    deskew_deg: float = 0.0
    needs_enhance: bool = False
    passthrough: bool = False


def _render_page(page: fitz.Page, dpi: int, *, max_side_px: int = 5200) -> Image.Image:
    rect = page.rect
    zoom = dpi / 72.0
    if max(rect.width, rect.height) * zoom > max_side_px:
        zoom = max_side_px / max(rect.width, rect.height)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _page_needs_enhance(rgb: np.ndarray, *, std_threshold: float = 38.0) -> bool:
    if rgb.size == 0:
        return False
    return float(rgb.mean(axis=2).std()) >= std_threshold


def _pil_to_pdf_page(doc: fitz.Document, image: Image.Image, *, jpeg_quality: int = 85) -> None:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    jpeg = buf.getvalue()
    w, h = image.size
    page = doc.new_page(width=w, height=h)
    rect = fitz.Rect(0, 0, w, h)
    page.insert_image(rect, stream=jpeg)


def _blank_kwargs(cfg: Settings) -> dict:
    return {
        "mean_threshold": cfg.blank_mean_threshold,
        "deviation_threshold": cfg.blank_deviation_threshold,
        "dark_threshold": cfg.blank_dark_pixel_threshold,
        "analysis_max_side": cfg.blank_analysis_max_side,
        "center_margin_frac": cfg.blank_center_margin_frac,
        "use_center_fallback": cfg.blank_use_center_fallback,
    }


def _eliminar_solo_blancos(pdf_bytes: bytes, cfg: Settings) -> PreprocessResult:
    """Ruta rápida: solo quita blancos, copia el resto sin tocar."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(src)
    src.close()

    workers = max(1, min(cfg.preprocess_workers, page_count, (os.cpu_count() or 4)))
    blank_kwargs = _blank_kwargs(cfg)
    kept: list[int] = []
    blanks: list[int] = []
    transforms: list[PageTransform] = []

    def _check_page(page_index: int) -> tuple[int, bool]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            thumb = _render_page(
                doc.load_page(page_index),
                cfg.preprocess_analysis_dpi,
                max_side_px=cfg.blank_analysis_max_side,
            )
        finally:
            doc.close()
        rgb = np.array(thumb.convert("RGB"))
        return page_index, is_blank_page_rgb(rgb, **blank_kwargs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_check_page, range(page_count)))

    for page_index, is_blank in sorted(results):
        orig_page = page_index + 1
        if is_blank:
            blanks.append(orig_page)
            transforms.append(PageTransform(orig_page, 0, 0.0, descartada=True))
        else:
            kept.append(orig_page)
            transforms.append(PageTransform(orig_page, 0, 0.0, descartada=False))

    if not kept:
        raise BlankDocumentError("No quedaron páginas útiles después de eliminar blancos.")

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()
    try:
        for orig_page in kept:
            out.insert_pdf(src, from_page=orig_page - 1, to_page=orig_page - 1)
        result_bytes = out.tobytes()
    finally:
        out.close()
        src.close()

    return PreprocessResult(
        pdf_bytes=result_bytes,
        kept_original_pages=kept,
        blank_pages_original=blanks,
        transforms=transforms,
    )


def _orient_kwargs(cfg: Settings, do_orient: bool, do_deskew: bool) -> dict:
    return {
        "corregir_orientacion": do_orient,
        "aplicar_deskew": do_deskew,
        "line_ratio": cfg.orientation_line_ratio,
        "flip_min_margin": cfg.flip_min_margin,
        "deskew_max_angle": cfg.deskew_max_angle,
        "deskew_min_angle": cfg.deskew_min_angle,
    }


def _analyze_thumb(
    pdf_bytes: bytes,
    page_index: int,
    *,
    cfg: Settings,
    do_blanks: bool,
) -> tuple[int, Image.Image | None]:
    """Rasteriza miniatura y devuelve (índice, imagen) o plan en blanco."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        thumb = _render_page(
            doc.load_page(page_index),
            cfg.preprocess_analysis_dpi,
            max_side_px=1200,
        )
    finally:
        doc.close()

    rgb = np.array(thumb.convert("RGB"))
    if do_blanks and is_blank_page_rgb(
        rgb,
        **_blank_kwargs(cfg),
    ):
        return page_index, None

    return page_index, thumb


def _build_plan(
    thumb: Image.Image,
    orig_page: int,
    *,
    cfg: Settings,
    do_orient: bool,
    do_deskew: bool,
    do_enhance: bool,
    ref_profile: np.ndarray | None,
) -> _PagePlan:
    rgb = np.array(thumb.convert("RGB"))
    coarse = 0
    deskew_deg = 0.0
    if do_orient or do_deskew:
        coarse, deskew_deg = analyze_page_orientation(
            thumb,
            ref_profile=ref_profile,
            **_orient_kwargs(cfg, do_orient, do_deskew),
        )

    needs_geom = coarse != 0 or abs(deskew_deg) >= cfg.deskew_min_angle
    needs_enhance = do_enhance and (needs_geom or _page_needs_enhance(rgb))
    passthrough = not needs_geom and not needs_enhance

    return _PagePlan(
        orig_page=orig_page,
        is_blank=False,
        coarse=coarse,
        deskew_deg=deskew_deg,
        needs_enhance=needs_enhance,
        passthrough=passthrough,
    )


def _process_full_page(
    pdf_bytes: bytes,
    plan: _PagePlan,
    *,
    cfg: Settings,
    do_enhance: bool,
) -> tuple[int, Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        image = _render_page(doc.load_page(plan.orig_page - 1), cfg.preprocess_dpi)
    finally:
        doc.close()

    if plan.coarse != 0 or abs(plan.deskew_deg) >= cfg.deskew_min_angle:
        image = apply_page_transform(
            image,
            plan.coarse,
            plan.deskew_deg,
            deskew_min_angle=cfg.deskew_min_angle,
        )

    if do_enhance and plan.needs_enhance:
        image = enhance_scanned_page(
            image,
            auto_crop=cfg.mejora_auto_crop,
            whiten_bg=cfg.mejora_whiten_background,
            despeckle_enabled=cfg.mejora_despeckle,
            reduce_streaks=cfg.mejora_reduce_streaks,
            emphasize=cfg.mejora_emphasize,
            crop_margin_px=cfg.mejora_crop_margin_px,
            bg_aggressiveness=cfg.mejora_bg_aggressiveness,
            contrast=cfg.mejora_contrast,
            sharpness=cfg.mejora_sharpness,
            max_side_px=cfg.preprocess_enhance_max_side_px,
        )

    return plan.orig_page, image


def normalizar_pdf(
    pdf_bytes: bytes,
    *,
    settings: Settings | None = None,
    eliminar_blancos: bool | None = None,
    corregir_orientacion: bool | None = None,
    aplicar_deskew: bool | None = None,
    aplicar_mejora_imagen: bool | None = None,
) -> PreprocessResult:
    cfg = settings or get_settings()
    do_blanks = cfg.eliminar_blancos if eliminar_blancos is None else eliminar_blancos
    do_orient = cfg.corregir_orientacion if corregir_orientacion is None else corregir_orientacion
    do_deskew = cfg.aplicar_deskew if aplicar_deskew is None else aplicar_deskew
    do_enhance = cfg.aplicar_mejora_imagen if aplicar_mejora_imagen is None else aplicar_mejora_imagen

    if do_blanks and not do_orient and not do_deskew and not do_enhance:
        return _eliminar_solo_blancos(pdf_bytes, cfg)

    if not do_blanks and not do_orient and not do_deskew and not do_enhance:
        src = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            n = len(src)
            return PreprocessResult(
                pdf_bytes=pdf_bytes,
                kept_original_pages=list(range(1, n + 1)),
            )
        finally:
            src.close()

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(src)
    src.close()

    workers = max(1, min(cfg.preprocess_workers, page_count, (os.cpu_count() or 4)))
    thumbs: dict[int, Image.Image] = {}
    blanks: list[int] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        thumb_futures = [
            pool.submit(
                _analyze_thumb,
                pdf_bytes,
                i,
                cfg=cfg,
                do_blanks=do_blanks,
            )
            for i in range(page_count)
        ]
        for fut in thumb_futures:
            page_index, thumb = fut.result()
            orig_page = page_index + 1
            if thumb is None:
                blanks.append(orig_page)
            else:
                thumbs[orig_page] = thumb

    ref_profile = (
        build_reference_profile(list(thumbs.values()))
        if do_orient and thumbs
        else None
    )

    plans: list[_PagePlan] = []
    transforms: list[PageTransform] = []
    for orig_page in sorted(thumbs):
        plan = _build_plan(
            thumbs[orig_page],
            orig_page,
            cfg=cfg,
            do_orient=do_orient,
            do_deskew=do_deskew,
            do_enhance=do_enhance,
            ref_profile=ref_profile,
        )
        plans.append(plan)
        transforms.append(
            PageTransform(orig_page, plan.coarse, plan.deskew_deg, descartada=False)
        )

    for orig_page in blanks:
        transforms.append(PageTransform(orig_page, 0, 0.0, descartada=True))

    if not plans:
        raise BlankDocumentError("No quedaron páginas útiles después de eliminar blancos.")

    passthrough_pages = [p.orig_page for p in plans if p.passthrough]
    raster_jobs = [p for p in plans if not p.passthrough]
    raster_map: dict[int, Image.Image] = {}

    if raster_jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            raster_futures = [
                pool.submit(
                    _process_full_page,
                    pdf_bytes,
                    plan,
                    cfg=cfg,
                    do_enhance=do_enhance,
                )
                for plan in raster_jobs
            ]
            for fut in raster_futures:
                orig_page, image = fut.result()
                raster_map[orig_page] = image

    kept = [p.orig_page for p in plans]
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = fitz.open()
    try:
        passthrough_set = set(passthrough_pages)
        for orig_page in kept:
            if orig_page in passthrough_set:
                out.insert_pdf(src, from_page=orig_page - 1, to_page=orig_page - 1)
            else:
                _pil_to_pdf_page(
                    out,
                    raster_map[orig_page],
                    jpeg_quality=cfg.preprocess_jpeg_quality,
                )
        result_bytes = out.tobytes()
    finally:
        out.close()
        src.close()

    transforms.sort(key=lambda t: t.pagina_original)

    return PreprocessResult(
        pdf_bytes=result_bytes,
        kept_original_pages=kept,
        blank_pages_original=blanks,
        transforms=transforms,
    )
