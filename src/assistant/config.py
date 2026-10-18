"""Central configuration, read from environment variables and the `.env` file.

All settings live in one typed object so that every module reads configuration the
same way, and tests can build a `Settings` with explicit values instead of touching
the real environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder containing pyproject.toml (src/assistant/config.py -> ../../)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class Settings(BaseSettings):
    """Typed application settings. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = "mock"
    llm_model: str = ""
    llm_api_key: str = Field(default="", repr=False)  # never printed in logs or reprs
    ollama_base_url: str = "http://localhost:11434"

    # Paths (relative paths are resolved against the project root)
    data_dir: Path = Path("data")
    index_dir: Path = Path("indexes")
    log_dir: Path = Path("logs")

    # Retrieval
    # Sizes are in tokens of the embedding model's tokenizer. bge-small reads at most 512
    # tokens, so the default stays below that limit (see docs/decisions.md).
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # SQL tool
    sql_row_limit: int = Field(default=100, gt=0)
    sql_timeout_seconds: float = Field(default=5.0, gt=0)

    @model_validator(mode="after")
    def _check_and_resolve(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        for name in ("data_dir", "index_dir", "log_dir"):
            path: Path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, PROJECT_ROOT / path)
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def index_dir_for(self, chunk_size: int) -> Path:
        """One index folder per chunk size, so the chunk-size ablation can keep all three."""
        return self.index_dir / f"cs{chunk_size}"

    @property
    def sales_db_path(self) -> Path:
        return self.data_dir / "sales.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (read once, then cached)."""
    return Settings()
