"""
Detección de hojas en blanco — alineada con TwainBridge / fi-7160.

Réplica de ``ImagePostProcessor.IsBlankPage`` (TwainBridgeCS):
- Muestreo espaciado sobre la imagen
- Luma = (R+G+B)/3
- Salida temprana si hay demasiados píxeles oscuros (< dark_threshold)
- Blanco si mean >= 230 y desviación <= 18 (valores por defecto del driver)

Extensión PDF: si la página completa falla por sombra en bordes, se repite
el mismo criterio TWAIN sobre la zona central (sin márgenes).
"""

from __future__ import annotations

import cv2
import numpy as np


def _resize_for_analysis(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb
    scale = max_side / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _crop_center(rgb: np.ndarray, margin_frac: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    y0, y1 = int(h * margin_frac), int(h * (1.0 - margin_frac))
    x0, x1 = int(w * margin_frac), int(w * (1.0 - margin_frac))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return rgb
    return rgb[y0:y1, x0:x1]


def _twain_is_blank(
    rgb: np.ndarray,
    *,
    mean_threshold: int,
    deviation_threshold: int,
    dark_threshold: int,
) -> bool:
    """Misma lógica que Twain.Api.Infrastructure.Imaging.ImagePostProcessor.IsBlankPage."""
    if rgb.size == 0:
        return False

    h, w = rgb.shape[:2]
    sample_step = max(4, min(w, h) // 250)
    sample_count = 0
    mean = 0.0
    squared = 0.0
    dark_pixel_count = 0

    for y in range(0, h, sample_step):
        for x in range(0, w, sample_step):
            r, g, b = int(rgb[y, x, 0]), int(rgb[y, x, 1]), int(rgb[y, x, 2])
            luma = (r + g + b) / 3.0
            mean += luma
            squared += luma * luma
            sample_count += 1
            if luma < dark_threshold:
                dark_pixel_count += 1
                dynamic_max = max(4, sample_count // 35)
                if dark_pixel_count > dynamic_max:
                    return False

    if sample_count == 0:
        return False

    mean /= sample_count
    variance = max(0.0, (squared / sample_count) - (mean * mean))
    deviation = float(np.sqrt(variance))
    return mean >= float(mean_threshold) and deviation <= float(deviation_threshold)


def is_blank_page_rgb(
    rgb: np.ndarray,
    *,
    mean_threshold: int,
    deviation_threshold: int,
    dark_threshold: int,
    analysis_max_side: int = 1200,
    center_margin_frac: float = 0.12,
    use_center_fallback: bool = True,
    **_ignored: object,
) -> bool:
    """¿Hoja en blanco? Criterio TWAIN fi-7160 (+ centro si hay sombra de PDF)."""
    if rgb.size == 0:
        return False

    sample = _resize_for_analysis(rgb, analysis_max_side)
    twain_kw = {
        "mean_threshold": mean_threshold,
        "deviation_threshold": deviation_threshold,
        "dark_threshold": dark_threshold,
    }

    if _twain_is_blank(sample, **twain_kw):
        return True

    if use_center_fallback:
        center = _crop_center(sample, center_margin_frac)
        if _twain_is_blank(center, **twain_kw):
            return True

    return False
