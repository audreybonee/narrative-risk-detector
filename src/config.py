"""
Configuration settings for the Emergent Narrative Detection System.

All settings can be overridden via environment variables.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CHROMA_DIR: Path = DATA_DIR / "chroma"

    # Embedding settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # MiniLM output dimension
    EMBEDDING_BATCH_SIZE: int = 32

    # ChromaDB settings
    CHROMA_COLLECTION_NAME: str = "news_articles"

    # HDBSCAN clustering settings
    HDBSCAN_MIN_CLUSTER_SIZE: int = 3
    HDBSCAN_MIN_SAMPLES: int = 2
    HDBSCAN_CLUSTER_SELECTION_EPSILON: float = 0.0
    HDBSCAN_METRIC: str = "euclidean"

    # Narrative detection thresholds
    VELOCITY_EMERGING_THRESHOLD: float = 0.5      # articles/hour for emerging
    VELOCITY_SPREADING_THRESHOLD: float = 2.0     # articles/hour for spreading
    VELOCITY_SPIKE_MULTIPLIER: float = 3.0        # multiplier for spike detection

    SOURCE_DIVERSITY_THRESHOLD: int = 3           # outlet types for established
    BIAS_SPREAD_CROSS_SPECTRUM: float = 0.7       # std dev for cross-spectrum

    FRAME_COHERENCE_THRESHOLD: float = 0.85       # embedding similarity for tight frame
    FRAME_COHERENCE_LOOSE: float = 0.70           # embedding similarity for loose frame

    # Time window settings
    DEFAULT_TIME_WINDOW: str = "4h"
    SUPPORTED_TIME_WINDOWS: list[str] = ["1h", "4h", "12h", "1d", "1w"]

    # Signal severity thresholds
    SIGNAL_SEVERITY_HIGH_VELOCITY: float = 5.0    # articles/hour
    SIGNAL_SEVERITY_HIGH_OUTLETS: int = 10        # unique outlets
    SIGNAL_SEVERITY_HIGH_DIVERSITY: int = 4       # outlet types

    # Hugging Face settings (for LLM labeling)
    HF_TOKEN: Optional[str] = None
    HF_MODEL_ID: str = "meta-llama/Llama-3.1-8B-Instruct"

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # Dashboard settings
    DASHBOARD_PORT: int = 8501

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)