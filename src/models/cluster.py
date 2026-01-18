"""
NarrativeCluster model for the Emergent Narrative Detection System.

Represents a cluster of articles sharing a common narrative.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, computed_field
import statistics

from .enums import NarrativeStage, OutletType, PatternType
from .article import Article


class NarrativeCluster(BaseModel):
    """
    A cluster of articles representing an emergent narrative.

    Contains metrics for tracking narrative spread, velocity, and characteristics.
    """

    id: str = Field(..., description="Unique cluster identifier")
    topic: Optional[str] = Field(default=None, description="Narrative topic/theme")
    representative_headline: str = Field(..., description="Most representative headline")
    articles: list[Article] = Field(default_factory=list, description="Articles in cluster")

    # Temporal bounds
    first_seen: datetime = Field(..., description="Earliest article timestamp")
    last_seen: datetime = Field(..., description="Latest article timestamp")

    # Cluster metadata
    centroid_embedding: Optional[list[float]] = Field(default=None, exclude=True)

    @computed_field
    @property
    def size(self) -> int:
        """Number of articles in cluster."""
        return len(self.articles)

    @computed_field
    @property
    def unique_outlets(self) -> int:
        """Number of unique outlets covering this narrative."""
        return len(set(a.outlet for a in self.articles))

    @computed_field
    @property
    def outlet_types(self) -> list[str]:
        """List of outlet types covering this narrative."""
        return list(set(a.outlet_type.value for a in self.articles))

    @computed_field
    @property
    def source_diversity(self) -> int:
        """Number of unique outlet types (wire, national, local, independent)."""
        return len(set(a.outlet_type for a in self.articles))

    @computed_field
    @property
    def bias_spread(self) -> float:
        """
        Standard deviation of outlet bias values.

        Higher values indicate cross-spectrum coverage.
        Lower values indicate echo chamber dynamics.
        """
        if len(self.articles) < 2:
            return 0.0
        bias_values = [a.bias_numeric for a in self.articles]
        return statistics.stdev(bias_values)

    @computed_field
    @property
    def bias_mean(self) -> float:
        """Mean outlet bias (-1.0 to 1.0 scale)."""
        if not self.articles:
            return 0.0
        return statistics.mean(a.bias_numeric for a in self.articles)

    @computed_field
    @property
    def duration_hours(self) -> float:
        """Duration of narrative activity in hours."""
        delta = self.last_seen - self.first_seen
        return delta.total_seconds() / 3600

    @computed_field
    @property
    def velocity(self) -> float:
        """
        Articles per hour (echo velocity).

        Measures how quickly the narrative is spreading.
        """
        if self.duration_hours == 0:
            return float(self.size)  # All at once
        return self.size / self.duration_hours

    @computed_field
    @property
    def wire_origin_ratio(self) -> float:
        """Proportion of articles from wire services."""
        if not self.articles:
            return 0.0
        wire_count = sum(1 for a in self.articles if a.is_wire_origin)
        return wire_count / len(self.articles)

    @computed_field
    @property
    def pr_origin_ratio(self) -> float:
        """Proportion of articles from PR sources."""
        if not self.articles:
            return 0.0
        pr_count = sum(1 for a in self.articles if a.is_pr_origin)
        return pr_count / len(self.articles)

    @computed_field
    @property
    def stage(self) -> NarrativeStage:
        """
        Determine current lifecycle stage based on metrics.

        Stages:
        - NASCENT: 1-2 sources, < 6 hours old
        - EMERGING: 3-5 sources, cross-outlet spread beginning
        - SPREADING: 6+ sources, clear velocity increase
        - ESTABLISHED: Wide coverage (source_diversity >= 3), consistent framing
        - DECLINING: Velocity decreasing (checked externally)
        - DORMANT: No new articles for extended period (checked externally)
        """
        if self.unique_outlets <= 2 and self.duration_hours < 6:
            return NarrativeStage.NASCENT
        elif self.unique_outlets <= 5:
            return NarrativeStage.EMERGING
        elif self.source_diversity < 3:
            return NarrativeStage.SPREADING
        else:
            return NarrativeStage.ESTABLISHED

    @computed_field
    @property
    def dominant_pattern_type(self) -> Optional[str]:
        """Most common pattern type among labeled articles."""
        pattern_counts: dict[str, int] = {}
        for article in self.articles:
            if article.pattern_type:
                pt = article.pattern_type.value
                pattern_counts[pt] = pattern_counts.get(pt, 0) + 1

        if not pattern_counts:
            return None
        return max(pattern_counts, key=pattern_counts.get)

    @computed_field
    @property
    def common_frame(self) -> Optional[str]:
        """Extract common coordinated or convergent frame if present."""
        # Check coordinated frames first (higher signal)
        coord_frames = [a.coordinated_frame for a in self.articles if a.coordinated_frame]
        if coord_frames:
            # Return most common
            from collections import Counter
            return Counter(coord_frames).most_common(1)[0][0]

        # Then check convergent frames
        conv_frames = [a.convergent_frame for a in self.articles if a.convergent_frame]
        if conv_frames:
            from collections import Counter
            return Counter(conv_frames).most_common(1)[0][0]

        return None

    def get_keywords_frequency(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Get most frequent keywords across all articles."""
        from collections import Counter
        all_keywords: list[str] = []
        for article in self.articles:
            all_keywords.extend(article.keywords)
        return Counter(all_keywords).most_common(top_n)

    def get_timeline(self) -> list[dict]:
        """Get chronological timeline of article publications."""
        sorted_articles = sorted(self.articles, key=lambda a: a.published_at)
        return [
            {
                "timestamp": a.published_at.isoformat(),
                "outlet": a.outlet,
                "outlet_type": a.outlet_type.value,
                "title": a.title,
            }
            for a in sorted_articles
        ]

    def to_summary_dict(self) -> dict:
        """Generate a summary dictionary for API responses."""
        return {
            "id": self.id,
            "topic": self.topic,
            "representative_headline": self.representative_headline,
            "size": self.size,
            "unique_outlets": self.unique_outlets,
            "source_diversity": self.source_diversity,
            "bias_spread": round(self.bias_spread, 3),
            "bias_mean": round(self.bias_mean, 3),
            "velocity": round(self.velocity, 2),
            "stage": self.stage.value,
            "duration_hours": round(self.duration_hours, 1),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "wire_origin_ratio": round(self.wire_origin_ratio, 2),
            "pr_origin_ratio": round(self.pr_origin_ratio, 2),
            "dominant_pattern": self.dominant_pattern_type,
            "common_frame": self.common_frame,
            "top_keywords": self.get_keywords_frequency(5),
        }