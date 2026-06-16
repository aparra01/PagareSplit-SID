"""
Mejora de imagen tipo PaperStream IP (fi-7160 / Ricoh).

Réplica software de funciones documentadas en el driver nativo:
- Auto crop (recorte al contenido)
- Background smoothing / blanqueo de fondo (similar a BGAdjust / Color cleanup)
- Despeckle / reducción de ruido (IdtcNoiseRemovalSensitivity aproximado)
- Vertical streaks reduction
- Image emphasis (contraste + nitidez para OCR/barcode)

Referencias: PaperStream IP — auto crop, deskew, iDTC, edge repair, vertical streaks.
El driver WIA del fi-7160 expone menos funciones que TWAIN PaperStream IP.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def auto_crop_content(
    bgr: np.ndarray,
    *,
    margin_px: int = 12,
    min_ink_ratio: float = 0.002,
) -> np.ndarray:
    """Recorte automático al bounding box del contenido (PaperStream auto crop)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = cv2.countNonZero(binary)
    if ink < int(binary.size * min_ink_ratio):
        return bgr

    coords = cv2.findNonZero(binary)
    if coords is None:
        return bgr

    x, y, w, h = cv2.boundingRect(coords)
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(bgr.shape[1], x + w + margin_px)
    y1 = min(bgr.shape[0], y + h + margin_px)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return bgr
    return bgr[y0:y1, x0:x1]


def whiten_background(
    bgr: np.ndarray,
    *,
    aggressiveness: float = 0.65,
) -> np.ndarray:
    """
    Suaviza y blanquea el fondo gris/amarillo del escaneo WIA.
    Inspirado en PaperStream BGAdjust / Advanced Cleanup (fondo).
    """
    aggressiveness = float(np.clip(aggressiveness, 0.0, 1.0))
    if aggressiveness <= 0:
        return bgr

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = max(31, (min(bgr.shape[:2]) // 20) | 1)
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    background = np.maximum(background, 1.0)

    normalized = gray / background
    normalized = np.clip(normalized, 0.0, 1.0)
    # Mezcla con original para no quemar tinta
    enhanced_gray = (1.0 - aggressiveness) * gray + aggressiveness * (normalized * 255.0)
    enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)

    # Preservar color relativo aplicando el mismo factor por canal
    scale = (enhanced_gray.astype(np.float32) + 1.0) / (gray + 1.0)
    out = np.clip(bgr.astype(np.float32) * scale[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return out


def despeckle(bgr: np.ndarray, *, strength: int = 1) -> np.ndarray:
    """Elimina motas pequeñas (opening morfológico suave)."""
    if strength <= 0:
        return bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 + strength, 2 + strength))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    mask = cleaned.astype(np.float32) / 255.0
    # Suavizar solo en zonas de tinta
    blurred = cv2.medianBlur(gray, 3)
    result = gray.astype(np.float32)
    result = result * (1.0 - mask * 0.3) + blurred.astype(np.float32) * (mask * 0.3)
    result = np.clip(result, 0, 255).astype(np.uint8)
    scale = (result.astype(np.float32) + 1.0) / (gray.astype(np.float32) + 1.0)
    return np.clip(bgr.astype(np.float32) * scale[:, :, np.newaxis], 0, 255).astype(np.uint8)


def reduce_vertical_streaks(bgr: np.ndarray, *, sensitivity: float = 0.55) -> np.ndarray:
    """
    Reduce rayas verticales del sensor/ADF (PaperStream vertical streaks reduction).
    Detecta columnas con mediana anómala respecto a vecinas.
    """
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    if sensitivity <= 0:
        return bgr

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    if w < 20:
        return bgr

    col_med = np.median(gray, axis=0)
    kernel = 15
    smoothed = np.convolve(col_med, np.ones(kernel) / kernel, mode="same")
    diff = smoothed - col_med
    threshold = np.percentile(np.abs(diff), 99) * sensitivity
    if threshold < 1.0:
        return bgr

    for x in range(w):
        if abs(diff[x]) > threshold and col_med[x] < np.median(col_med) * 0.92:
            left = max(0, x - 3)
            right = min(w, x + 4)
            neighbors = np.concatenate([col_med[left:x], col_med[x + 1 : right]])
            if neighbors.size > 0:
                gray[:, x] = np.median(neighbors)

    scale = (gray + 1.0) / (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0)
    return np.clip(bgr.astype(np.float32) * scale[:, :, np.newaxis], 0, 255).astype(np.uint8)


def emphasize_document(image: Image.Image, *, contrast: float = 1.35, sharpness: float = 1.4) -> Image.Image:
    """Énfasis de texto (PaperStream Image Emphasis / OCR optimization ligero)."""
    gray = ImageOps.grayscale(image)
    auto = ImageOps.autocontrast(gray, cutoff=1)
    sharp = ImageEnhance.Sharpness(ImageEnhance.Contrast(auto).enhance(contrast)).enhance(sharpness)
    sharp = sharp.filter(ImageFilter.SHARPEN)
    # Mezclar luminancia mejorada con color original
    if image.mode != "RGB":
        image = image.convert("RGB")
    orig = np.array(image, dtype=np.float32)
    lum = np.array(sharp, dtype=np.float32)
    orig_gray = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) + 1.0
    scale = lum / orig_gray
    scale = np.clip(scale, 0.85, 1.25)
    out = np.clip(orig * scale[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def enhance_scanned_page(
    image: Image.Image,
    *,
    auto_crop: bool = True,
    whiten_bg: bool = True,
    despeckle_enabled: bool = True,
    reduce_streaks: bool = True,
    emphasize: bool = True,
    crop_margin_px: int = 12,
    bg_aggressiveness: float = 0.65,
    streak_sensitivity: float = 0.55,
    contrast: float = 1.35,
    sharpness: float = 1.4,
    max_side_px: int = 3200,
) -> Image.Image:
    """Pipeline completo de mejora post-escaneo (orden similar a PaperStream IP)."""
    w, h = image.size
    if max(w, h) > max_side_px > 0:
        scale = max_side_px / max(w, h)
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    bgr = _pil_to_bgr(image)

    if whiten_bg:
        bgr = whiten_background(bgr, aggressiveness=bg_aggressiveness)
    if despeckle_enabled:
        bgr = despeckle(bgr, strength=1)
    if reduce_streaks:
        bgr = reduce_vertical_streaks(bgr, sensitivity=streak_sensitivity)
    if auto_crop:
        bgr = auto_crop_content(bgr, margin_px=crop_margin_px)

    result = _bgr_to_pil(bgr)
    if emphasize:
        result = emphasize_document(result, contrast=contrast, sharpness=sharpness)
    return result
