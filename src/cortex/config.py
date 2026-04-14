from pathlib import Path

from pydantic import field_validator
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
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # --- Credentials ---
    # Tokens are optional at schema level; provider-specific code validates them.
    hf_token: str = ""
    openai_api_key: str = ""

    # --- Models ---
    # The embedding model runs LOCALLY via sentence-transformers (no API cost).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # The generation model runs remotely via Hugging Face Inference API.
    # Which provider to use for answer generation.
    # - "hf-inference" is free (rate-limited) but supports only a limited set of models.
    # - Other providers may require enabling them in your HF account and can consume credits.
    # - "openai" uses the OpenAI API.
    generation_provider: str = "hf-inference"

    # Model used for answer generation and (optionally) query expansion.
    #
    # NOTE: with provider="hf-inference", the set of supported chat models can be very limited.
    # If you enable a stronger provider, set GENERATION_MODEL accordingly in your .env.
    generation_model: str = "katanemo/Arch-Router-1.5B"

    # OpenAI settings (used when GENERATION_PROVIDER=openai)
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

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

    # --- Scraper / security ---
    # Comma-separated allowlist of domains permitted for scraping.
    # By default, scraping is locked down (empty allowlist).
    allowed_domains: list[str] = []
    allow_subdomains: bool = False
    max_redirects: int = 3
    deny_private_ips: bool = True

    # --- Logging (quiet, file-based) ---
    log_file: Path = Path("logs/cortex.log")
    log_level: str = "INFO"
    log_max_bytes: int = 1_000_000
    log_backups: int = 3

    # --- @property: computed attribute ---
    # A @property looks like an attribute but is computed on access.
    # chroma_path always reflects the current chroma_dir value.
    @property
    def chroma_path(self) -> Path:
        """
        Return ChromaDB storage path as a Path object.
        """

        return Path(self.chroma_dir)

    @field_validator("allowed_domains", mode="before")
    @classmethod
    def _parse_allowed_domains(cls, v):  # noqa: ANN001
        if v is None:
            return []

        if isinstance(v, list):
            return [str(x).strip().lower().strip(".") for x in v if str(x).strip()]

        if isinstance(v, str):
            raw = v.strip()

            if not raw:
                return []

            parts = [
                p.strip().lower().strip(".") for p in raw.replace(";", ",").split(",")
            ]

            return [p for p in parts if p]
            
        return v


# Module-level singleton: one Config instance shared across the whole app.
# Loaded one at import time, cached for all subsequent imports.
config = Config()
