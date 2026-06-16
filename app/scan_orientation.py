"""Orientación de escaneo: 90°/270° (OpenCV), 180° (pagaré), deskew fino."""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.blank_detection import is_blank_page_rgb

try:
    from deskew import determine_skew
except ImportError:  # pragma: no cover
    determine_skew = None  # type: ignore[assignment,misc]


class BlankDocumentError(Exception):
    """Todas las páginas del PDF quedaron en blanco tras el filtro."""


@dataclass
class PageTransform:
    pagina_original: int
    coarse_grados: int
    deskew_grados: float
    descartada: bool = False


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def rotate_bound_bgr(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=(255, 255, 255))


def _horizontal_vertical_scores(bgr: np.ndarray) -> tuple[int, int]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    return cv2.countNonZero(horizontal), cv2.countNonZero(vertical)


def _layout_profile(gray: Image.Image) -> np.ndarray:
    small = gray.resize((220, 310))
    arr = np.array(small, dtype=float)[:170, :]
    return ((arr - arr.mean()) / (arr.std() + 1e-6)).ravel()


def _profile_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(a * b))


def _top_bottom_ink_ratio(gray: Image.Image) -> float:
    arr = np.array(gray, dtype=float)
    h = arr.shape[0]
    band = max(1, h // 3)
    top = arr[:band, :].mean()
    bottom = arr[h - band :, :].mean()
    denom = top + bottom + 1e-6
    return float(top / denom)


def _barcode_valido(fmt: str, digits: str) -> bool:
    if len(digits) < 6:
        return False
    fmt_u = (fmt or "").upper()
    if fmt_u and "CODE" not in fmt_u and "39" not in fmt_u:
        return False
    return True


def _digits(text: str) -> str:
    return re.sub(r"\D+", "", text or "")


def _has_valid_barcode_top(image: Image.Image) -> bool:
    try:
        import zxingcpp
    except ImportError:
        return False

    width, height = image.size
    top35 = max(1, int(height * 0.35))
    crop = image.crop((0, 0, width, top35))
    gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
    for barcode in zxingcpp.read_barcodes(
        gray,
        try_rotate=True,
        try_downscale=True,
        try_invert=True,
        return_errors=False,
    ):
        text = (getattr(barcode, "text", "") or "").strip()
        fmt = str(getattr(barcode, "format", "")).replace("BarcodeFormat.", "")
        if _barcode_valido(fmt, _digits(text)):
            return True
    return False


def _choose_90_270(bgr: np.ndarray, line_ratio: float) -> int:
    h, w = bgr.shape[:2]
    is_portrait = h > w * 1.05
    is_landscape = w > h * 1.05

    # Pagaré formato actual escaneado bien → vertical. No forzar 90°/270°.
    if is_portrait or not is_landscape:
        return 0

    h0, v0 = _horizontal_vertical_scores(bgr)
    baseline = h0 - 0.35 * v0

    if v0 <= h0 * line_ratio:
        return 0

    candidates: list[tuple[int, float]] = [(0, baseline)]
    for angle in (90, 270):
        rotated = rotate_bound_bgr(bgr, angle)
        h_score, v_score = _horizontal_vertical_scores(rotated)
        candidates.append((angle, h_score - 0.35 * v_score))

    best_angle, best_score = max(candidates, key=lambda x: x[1])
    if best_angle == 0:
        return 0

    min_gain = max(12000.0, baseline * 0.18)
    if best_score - baseline < min_gain:
        return 0
    return int(best_angle)


def _choose_180(
    bgr: np.ndarray,
    ref_profile: np.ndarray | None,
    *,
    flip_min_margin: float,
) -> int:
    pil = _bgr_to_pil(bgr)
    h, w = bgr.shape[:2]
    if h > w * 1.05:
        # Páginas verticales: voltear solo con señal fuerte (barcode o similitud clara).
        flip_min_margin = max(flip_min_margin, 0.22)
    skip_barcode = h > w * 1.05  # zxing en vertical ruidoso es lento y poco fiable

    scores: dict[int, float] = {}
    sims: dict[int, float] = {}
    barcode_at: dict[int, bool] = {}
    for angle in (0, 180):
        rotated = pil if angle == 0 else pil.rotate(180, expand=True)
        g = ImageOps.grayscale(rotated)
        prof = _layout_profile(g)
        sim = _profile_similarity(prof, ref_profile) if ref_profile is not None else 0.0
        sims[angle] = sim
        top_ratio = _top_bottom_ink_ratio(g)
        barcode_at[angle] = False if skip_barcode else _has_valid_barcode_top(rotated)
        barcode_hit = 1.0 if barcode_at[angle] else 0.0
        scores[angle] = sim * 0.55 + top_ratio * 0.25 + barcode_hit * 0.20

    if barcode_at.get(180) and not barcode_at.get(0):
        return 180

    if scores[0] >= scores[180]:
        return 0

    sim0, sim180 = sims[0], sims[180]
    abs_gain = scores[180] - scores[0]
    sim_gain = sim180 - sim0

    # Perfil de referencia no encaja (escaneo ruidoso): no voltear por ruido.
    if ref_profile is not None and max(sim0, sim180) < 0.08:
        return 0

    if abs_gain < 0.06 or sim_gain < 0.05:
        return 0

    margin = abs_gain / (max(scores[180], scores[0], 0.01))
    return 180 if margin >= flip_min_margin else 0


def apply_deskew_bgr(
    bgr: np.ndarray,
    *,
    max_angle: float,
    min_angle: float,
) -> tuple[np.ndarray, float]:
    if determine_skew is None:
        return bgr, 0.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(gray, max_angle=max_angle, min_angle=min_angle)
    if angle is None or abs(angle) < min_angle or abs(angle) > max_angle:
        return bgr, 0.0
    return rotate_bound_bgr(bgr, angle), float(angle)


def analyze_page_orientation(
    image: Image.Image,
    *,
    ref_profile: np.ndarray | None,
    corregir_orientacion: bool,
    aplicar_deskew: bool,
    line_ratio: float,
    flip_min_margin: float,
    deskew_max_angle: float,
    deskew_min_angle: float,
) -> tuple[int, float]:
    """Detecta rotación gruesa y deskew sin modificar la imagen."""
    bgr = _pil_to_bgr(image)
    coarse = 0
    deskew_deg = 0.0

    if corregir_orientacion:
        coarse_90 = _choose_90_270(bgr, line_ratio)
        if coarse_90:
            bgr = rotate_bound_bgr(bgr, coarse_90)
            coarse = coarse_90

        coarse_180 = _choose_180(bgr, ref_profile, flip_min_margin=flip_min_margin)
        if coarse_180:
            coarse = (coarse + coarse_180) % 360

    if aplicar_deskew:
        _, deskew_deg = apply_deskew_bgr(
            bgr,
            max_angle=deskew_max_angle,
            min_angle=deskew_min_angle,
        )
        reject_deskew = (
            (coarse != 0 and abs(deskew_deg) < deskew_min_angle * 2)
            or (coarse == 0 and abs(deskew_deg) < max(deskew_min_angle * 1.5, 1.2))
        )
        if reject_deskew:
            deskew_deg = 0.0

    return coarse, deskew_deg


def apply_page_transform(
    image: Image.Image,
    coarse: int,
    deskew_deg: float,
    *,
    deskew_min_angle: float,
) -> Image.Image:
    """Aplica rotación gruesa y deskew ya detectados."""
    if coarse == 0 and abs(deskew_deg) < deskew_min_angle:
        return image

    bgr = _pil_to_bgr(image)
    if coarse:
        bgr = rotate_bound_bgr(bgr, float(coarse))
    if abs(deskew_deg) >= deskew_min_angle:
        bgr = rotate_bound_bgr(bgr, deskew_deg)
    return _bgr_to_pil(bgr)


def normalize_page_image(
    image: Image.Image,
    *,
    ref_profile: np.ndarray | None,
    corregir_orientacion: bool,
    aplicar_deskew: bool,
    line_ratio: float,
    flip_min_margin: float,
    deskew_max_angle: float,
    deskew_min_angle: float,
) -> tuple[Image.Image, int, float]:
    """Devuelve imagen corregida, rotación gruesa (0/90/180/270) y deskew en grados."""
    coarse, deskew_deg = analyze_page_orientation(
        image,
        ref_profile=ref_profile,
        corregir_orientacion=corregir_orientacion,
        aplicar_deskew=aplicar_deskew,
        line_ratio=line_ratio,
        flip_min_margin=flip_min_margin,
        deskew_max_angle=deskew_max_angle,
        deskew_min_angle=deskew_min_angle,
    )
    corrected = apply_page_transform(
        image,
        coarse,
        deskew_deg,
        deskew_min_angle=deskew_min_angle,
    )
    return corrected, coarse, deskew_deg


def build_reference_profile(images: list[Image.Image]) -> np.ndarray | None:
    for image in images:
        if not is_blank_page_rgb(
            np.array(image.convert("RGB")),
            mean_threshold=230,
            deviation_threshold=18,
            dark_threshold=175,
            use_center_fallback=True,
        ):
            return _layout_profile(ImageOps.grayscale(image))
    return None
