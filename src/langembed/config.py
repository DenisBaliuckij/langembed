"""Configuration: YAML loader and environment settings.

`load_config` depends only on stdlib + PyYAML so it is importable without the
serve extras. `get_settings` lazily imports pydantic-settings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_settings() -> Any:
    """Return environment-backed settings (DB / Redis). Imports lazily."""
    from pydantic import computed_field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")

        postgres_user: str = "langembed"
        postgres_password: str = "langembed"
        postgres_host: str = "localhost"
        postgres_port: int = 5432
        postgres_db: str = "langembed"
        redis_url: str = "redis://localhost:6379/0"

        @computed_field  # type: ignore[misc]
        @property
        def database_url(self) -> str:
            return (
                f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )

    return Settings()
