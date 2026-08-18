from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from the environment or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llama_base_url: str = "http://localhost:8085/v1"
    llama_model: str = "qwen3"
    llama_timeout: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
