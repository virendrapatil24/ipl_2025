from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None

    # Model Settings
    DEFAULT_MODEL: str = "gpt-4"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Data Paths
    DATA_DIR: str = "data"
    PROCESSED_DATA_DIR: str = "processed_data"

    # Vector Store
    VECTOR_STORE_PATH: str = "vector_store"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
