from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8006
    reload: bool = False
    max_pdf_mb: int = 150
    default_dpi: int = 160

    model_config = SettingsConfigDict(
        env_prefix="PAGARE_SPLIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
