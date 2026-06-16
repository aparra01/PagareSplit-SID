from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8006
    reload: bool = False
    max_pdf_mb: int = 150
    default_dpi: int = 160

    eliminar_blancos: bool = True
    corregir_orientacion: bool = False
    aplicar_deskew: bool = False
    aplicar_mejora_imagen: bool = False
    preprocess_dpi: int = 200
    preprocess_analysis_dpi: int = 96
    preprocess_workers: int = 4
    preprocess_enhance_max_side_px: int = 3200
    preprocess_jpeg_quality: int = 92
    orientation_line_ratio: float = 1.8
    flip_min_margin: float = 0.18
    deskew_max_angle: float = 8.0
    deskew_min_angle: float = 0.8
    # Blancos — mismos umbrales que TwainBridge / fi-7160 (appsettings ScannerApi)
    blank_mean_threshold: int = 230
    blank_deviation_threshold: int = 18
    blank_dark_pixel_threshold: int = 175
    blank_analysis_max_side: int = 1200
    blank_center_margin_frac: float = 0.12
    blank_use_center_fallback: bool = True

    # Mejora de imagen (desactivada por defecto; activar con PAGARE_SPLIT_APLICAR_MEJORA_IMAGEN=true)
    mejora_auto_crop: bool = False
    mejora_whiten_background: bool = True
    mejora_despeckle: bool = True
    mejora_reduce_streaks: bool = True
    mejora_emphasize: bool = True
    mejora_bg_aggressiveness: float = 0.45
    mejora_crop_margin_px: int = 12
    mejora_contrast: float = 1.15
    mejora_sharpness: float = 1.2

    model_config = SettingsConfigDict(
        env_prefix="PAGARE_SPLIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
