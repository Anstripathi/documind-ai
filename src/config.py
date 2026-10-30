"""Central configuration. Values can be overridden with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_file_mb: int = 20
    max_pages: int = 200
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    min_score: float = 0.20
    max_question_chars: int = 1000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout: int = 120

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


def load_settings() -> Settings:
    def _int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except ValueError:
            return default

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except ValueError:
            return default

    d = Settings()
    return Settings(
        max_file_mb=_int("DOCUMIND_MAX_FILE_MB", d.max_file_mb),
        max_pages=_int("DOCUMIND_MAX_PAGES", d.max_pages),
        chunk_size=_int("DOCUMIND_CHUNK_SIZE", d.chunk_size),
        chunk_overlap=_int("DOCUMIND_CHUNK_OVERLAP", d.chunk_overlap),
        top_k=_int("DOCUMIND_TOP_K", d.top_k),
        min_score=_float("DOCUMIND_MIN_SCORE", d.min_score),
        max_question_chars=_int(
            "DOCUMIND_MAX_QUESTION_CHARS", d.max_question_chars
        ),
        embedding_model=os.getenv(
            "DOCUMIND_EMBEDDING_MODEL", d.embedding_model
        ),
        llm_provider=os.getenv(
            "DOCUMIND_LLM_PROVIDER", d.llm_provider
        ).strip().lower(),
        llm_model=os.getenv("DOCUMIND_LLM_MODEL", d.llm_model),
        ollama_url=os.getenv("DOCUMIND_OLLAMA_URL", d.ollama_url),
        ollama_timeout=_int(
            "DOCUMIND_OLLAMA_TIMEOUT", d.ollama_timeout
        ),
    )
