"""
Tests for the Emergent Narrative Detection System.
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModels:
    """Tests for data models."""

    def test_article_creation(self):
        """Test Article model creation."""
        from src.models import Article, OutletType, OutletBias, WireSource

        article = Article(
            id="test_001",
            outlet="Test News",
            outlet_type=OutletType.MAJOR_NATIONAL,
            outlet_bias=OutletBias.CENTER,
            title="Test Headline",
            body="Test body content",
            published_at=datetime.now(),
        )

        assert article.id == "test_001"
        assert article.combined_text == "Test Headline\n\nTest body content"
        assert article.bias_numeric == 0.0

    def test_outlet_bias_numeric(self):
        """Test OutletBias numeric conversion."""
        from src.models import OutletBias

        assert OutletBias.LEFT.numeric_value == -1.0
        assert OutletBias.CENTER.numeric_value == 0.0
        assert OutletBias.RIGHT.numeric_value == 1.0

    def test_pattern_type_signal_strength(self):
        """Test PatternType signal strength."""
        from src.models import PatternType

        assert PatternType.WIRE_ECHO.signal_strength == "low"
        assert PatternType.PR_AMPLIFICATION.signal_strength == "medium"
        assert PatternType.SYNCHRONIZED_MESSAGING.signal_strength == "high"

    def test_narrative_stage_order(self):
        """Test NarrativeStage ordering."""
        from src.models import NarrativeStage

        assert NarrativeStage.NASCENT.sort_order < NarrativeStage.EMERGING.sort_order
        assert NarrativeStage.EMERGING.sort_order < NarrativeStage.SPREADING.sort_order


class TestIngestion:
    """Tests for data ingestion."""

    def test_parse_time_window(self):
        """Test time window parsing."""
        from src.ingestion import parse_time_window
        from datetime import timedelta

        assert parse_time_window("1h") == timedelta(hours=1)
        assert parse_time_window("4h") == timedelta(hours=4)
        assert parse_time_window("1d") == timedelta(days=1)
        assert parse_time_window("1w") == timedelta(weeks=1)

    def test_parse_time_window_invalid(self):
        """Test invalid time window raises error."""
        from src.ingestion import parse_time_window

        with pytest.raises(ValueError):
            parse_time_window("invalid")


class TestProcessing:
    """Tests for processing components."""

    def test_compute_similarity(self):
        """Test cosine similarity computation."""
        from src.processing import compute_similarity

        # Identical vectors
        vec1 = [1.0, 0.0, 0.0]
        assert compute_similarity(vec1, vec1) == pytest.approx(1.0)

        # Orthogonal vectors
        vec2 = [0.0, 1.0, 0.0]
        assert compute_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_compute_centroid(self):
        """Test centroid computation."""
        from src.processing import compute_centroid

        embeddings = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        centroid = compute_centroid(embeddings)
        assert centroid == pytest.approx([0.5, 0.5])


class TestConfig:
    """Tests for configuration."""

    def test_settings_defaults(self):
        """Test default settings."""
        from src.config import settings

        assert settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert settings.HDBSCAN_MIN_CLUSTER_SIZE == 3
        assert settings.API_PORT == 8000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])