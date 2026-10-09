import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration."""

    APP_NAME: str = "PaperMatcher-AI"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8002
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ]
    DATASET_PATH: str = "data/arXiv_scientific_dataset.csv"
    MODELS_DIR: str = "models"

    # Hybrid Retrieval Settings
    RETRIEVAL_MODE: str = "hybrid"  # "hybrid" | "tfidf" | "semantic"
    TFIDF_WEIGHT: float = 0.40
    SEMANTIC_WEIGHT: float = 0.60
    SCORE_NORMALIZATION: str = "minmax"  # "minmax" | "clip" | "none"

    # Sentence-BERT Model & Embeddings Settings
    SBERT_MODEL_NAME: str = "all-MiniLM-L6-v2"
    SBERT_MAX_SEQ_LENGTH: int = 128
    SBERT_EMBEDDINGS_PATH: str = "models/sbert_embeddings.npy"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_project_root(self) -> Path:
        """Resolve project root directory (4 levels up from config.py)."""
        return Path(__file__).resolve().parent.parent.parent.parent

    def get_dataset_file_path(self) -> Path:
        """Resolve dataset file path relative to project root or absolute."""
        candidate = Path(self.DATASET_PATH)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        # Check relative to current working directory
        if candidate.exists():
            return candidate.resolve()

        # Check relative to project root
        project_root = self.get_project_root()
        if (project_root / candidate).exists():
            return (project_root / candidate).resolve()

        return candidate.resolve()

    def get_models_dir(self) -> Path:
        """Resolve models directory relative to project root or absolute."""
        candidate = Path(self.MODELS_DIR)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        project_root = self.get_project_root()
        target = (project_root / candidate).resolve()
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_sbert_embeddings_file_path(self) -> Path:
        """Resolve precomputed SBERT embeddings file path."""
        candidate = Path(self.SBERT_EMBEDDINGS_PATH)
        if candidate.is_absolute():
            return candidate
        return (self.get_project_root() / candidate).resolve()


settings = Settings()
