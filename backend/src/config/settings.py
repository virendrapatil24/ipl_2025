from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and environment variables."""

    # API Keys
    openai_api_key: str
    anthropic_api_key: Optional[str] = None

    # Data paths
    data_dir: Path = Path("data")
    processed_data_dir: Path = Path("processed_data")
    matches_file: Path = data_dir / "matches.csv"
    deliveries_file: Path = data_dir / "deliveries.csv"
    squads_file: Path = data_dir / "squads.csv"

    # Vector store settings
    vector_store_dir: Path = Path("vector_store")
    vector_store_path: str = "vector_store"

    # Model settings
    default_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    ollama_base_url: str = "http://localhost:11434"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # CORS settings
    allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        encoding = "utf-8"


# Create settings instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.processed_data_dir.mkdir(exist_ok=True)
settings.vector_store_dir.mkdir(exist_ok=True)
