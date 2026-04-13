from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """
    Application configuration loaded from environment variables and .env file.

    All fields are validated at startup. Missing required fields (like hf_token) will raise a clear error before any code runs.

    Pydantic-settings reads values in this priority order:
        1. Environment variables
        2. .env file
        3. Default values defined here
    """

    # Tells pydantic-settings to look for a .env file in the working directory
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # --- Required ---
    # No default value = required field. Will fail loudly if missing.
    hf_token: str

    # --- Models ---
    # The embedding model runs LOCALLY via sentence-transformers (no API cost).
    embedding_model: str = "HuggingFaceH4/zephyr-7b-beta"

    # --- ChromaDB ---
    chroma_dir: str = ".chroma"
    collection_name: str = "cortex"

    # --- Chunking ---
    # Chunk size in characters (~200-250 tokens for all-MiniLM-L6-v2's 256-token limit)
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Retrieval ---
    top_k: int = 5
    # Set to True to enable multi-query expansion (uses extra API calls)
    use_query_expansion: bool = False

    # --- @property: computed attribute ---
    # A @property looks like an attribute but is computed on access.
    # chroma_path always reflects the current chroma_dir value.
    @property
    def chroma_path(self) -> Path:
        """
        Return ChromaDB storage path as a Path object.
        """
        return Path(self.chroma_dir)


# Module-level singleton: one Config instance shared across the whole app.
# Loaded one at import time, cached for all subsequent imports.
config = Config()
