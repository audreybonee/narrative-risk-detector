"""
Weak supervision labeling functions for coordination detection.
Combines multiple heuristic signals to generate pseudo-labels.
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from src.models.cluster import Cluster
from src.processing.coordination import CoordinationSignals
from src.config import settings


class CoordinationLabel(str, Enum):
    """Coordination label categories."""
    ORGANIC = "organic"
    UNKNOWN = "unknown"
    SUSPICIOUS = "suspicious"
    LIKELY_COORDINATED = "likely_coordinated"


@dataclass
class WeakLabel:
    """Weak supervision label with confidence and provenance."""
    label: CoordinationLabel
    confidence: float
    labeling_functions_fired: List[str] = field(default_factory=list)
    votes: Dict[str, int] = field(default_factory=dict)


@dataclass
class LabelingFunctionResult:
    """Result from a single labeling function."""
    name: str
    label: Optional[CoordinationLabel]
    weight: float = 1.0


class WeakSupervisionLabeler:
    """
    Apply weak supervision labeling functions.

    Combines multiple heuristic signals:
    - Coordination ratio thresholds
    - Near-duplicate burst detection
    - Temporal velocity spikes
    - Semantic coherence patterns

    Labels are aggregated via weighted voting.
    """

    def __init__(
        self,
        coordination_ratio_suspicious: Optional[float] = None,
        coordination_ratio_likely: Optional[float] = None,
        duplicate_burst_threshold: float = 0.5,
        velocity_zscore_threshold: float = 2.0,
        coherence_threshold: float = 0.85,
    ):
        """
        Initialize weak supervision labeler.

        Args:
            coordination_ratio_suspicious: Threshold for suspicious (default from settings)
            coordination_ratio_likely: Threshold for likely coordinated
            duplicate_burst_threshold: Near-duplicate burst score threshold
            velocity_zscore_threshold: Temporal velocity z-score threshold
            coherence_threshold: Semantic coherence threshold
        """
        self.coord_suspicious = (
            coordination_ratio_suspicious or settings.coordination_ratio_suspicious
        )
        self.coord_likely = (
            coordination_ratio_likely or settings.coordination_ratio_likely
        )
        self.dup_threshold = duplicate_burst_threshold
        self.velocity_threshold = velocity_zscore_threshold
        self.coherence_threshold = coherence_threshold

    # ==========================================================================
    # Labeling Functions
    # ==========================================================================

    def _lf_high_coordination_ratio(
        self,
        signals: CoordinationSignals,
        cluster: Cluster
    ) -> LabelingFunctionResult:
        """
        Label based on coordination ratio.

        High ratio = likely coordinated
        Low ratio = likely organic
        """
        name = "lf_coordination_ratio"

        if signals.coordination_ratio > self.coord_likely:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.LIKELY_COORDINATED,
                weight=1.5  # High confidence signal
            )
        elif signals.coordination_ratio > self.coord_suspicious:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.SUSPICIOUS,
                weight=1.0
            )
        elif signals.coordination_ratio < 1.2:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.ORGANIC,
                weight=0.8
            )

        return LabelingFunctionResult(name=name, label=None)

    def _lf_duplicate_burst(
        self,
        signals: CoordinationSignals,
        cluster: Cluster
    ) -> LabelingFunctionResult:
        """
        Label based on near-duplicate content bursts.

        High duplicate score = coordinated messaging with templates
        """
        name = "lf_duplicate_burst"

        if signals.duplicate_burst_score > 0.7:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.LIKELY_COORDINATED,
                weight=1.3
            )
        elif signals.duplicate_burst_score > self.dup_threshold:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.SUSPICIOUS,
                weight=1.0
            )
        elif signals.duplicate_burst_score < 0.1:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.ORGANIC,
                weight=0.7
            )

        return LabelingFunctionResult(name=name, label=None)

    def _lf_temporal_burst(
        self,
        signals: CoordinationSignals,
        cluster: Cluster
    ) -> LabelingFunctionResult:
        """
        Label based on temporal velocity anomalies.

        Sudden spikes with high coherence = suspicious
        """
        name = "lf_temporal_burst"

        if signals.velocity_zscore > 4.0 and signals.semantic_coherence > 0.8:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.LIKELY_COORDINATED,
                weight=1.2
            )
        elif signals.burst_detected and signals.semantic_coherence > 0.7:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.SUSPICIOUS,
                weight=0.9
            )

        return LabelingFunctionResult(name=name, label=None)

    def _lf_high_coherence_low_diversity(
        self,
        signals: CoordinationSignals,
        cluster: Cluster
    ) -> LabelingFunctionResult:
        """
        Label based on coherence vs author diversity.

        High coherence + low author diversity = suspicious
        Many similar messages from few authors
        """
        name = "lf_coherence_diversity"

        if (signals.semantic_coherence > self.coherence_threshold and
            signals.coordination_ratio > 1.5):
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.SUSPICIOUS,
                weight=0.8
            )

        return LabelingFunctionResult(name=name, label=None)

    def _lf_small_cluster_organic(
        self,
        signals: CoordinationSignals,
        cluster: Cluster
    ) -> LabelingFunctionResult:
        """
        Small clusters with low ratio are likely organic.
        """
        name = "lf_small_cluster"

        if cluster.size < 10 and signals.coordination_ratio < 1.5:
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.ORGANIC,
                weight=0.6
            )

        return LabelingFunctionResult(name=name, label=None)

    def _lf_large_cluster_diverse(
        self,
        signals: CoordinationSignals,
        cluster: Cluster
    ) -> LabelingFunctionResult:
        """
        Large clusters with many unique authors are likely organic.
        """
        name = "lf_large_diverse"

        if (cluster.size > 50 and
            signals.coordination_ratio < 1.3 and
            cluster.unique_authors > 40):
            return LabelingFunctionResult(
                name=name,
                label=CoordinationLabel.ORGANIC,
                weight=1.0
            )

        return LabelingFunctionResult(name=name, label=None)

    # ==========================================================================
    # Label Aggregation
    # ==========================================================================

    def get_all_labeling_functions(self) -> List[Callable]:
        """Get all labeling functions."""
        return [
            self._lf_high_coordination_ratio,
            self._lf_duplicate_burst,
            self._lf_temporal_burst,
            self._lf_high_coherence_low_diversity,
            self._lf_small_cluster_organic,
            self._lf_large_cluster_diverse,
        ]

    def label_cluster(
        self,
        cluster: Cluster,
        signals: CoordinationSignals
    ) -> WeakLabel:
        """
        Apply all labeling functions and aggregate.

        Uses weighted voting to combine signals.

        Args:
            cluster: Cluster to label
            signals: Coordination signals for cluster

        Returns:
            WeakLabel with aggregated result
        """
        # Collect votes from all labeling functions
        votes: Dict[CoordinationLabel, float] = {
            CoordinationLabel.ORGANIC: 0.0,
            CoordinationLabel.SUSPICIOUS: 0.0,
            CoordinationLabel.LIKELY_COORDINATED: 0.0,
        }

        fired_functions = []
        vote_details = {}

        for lf in self.get_all_labeling_functions():
            result = lf(signals, cluster)

            if result.label is not None:
                votes[result.label] += result.weight
                fired_functions.append(result.name)
                vote_details[result.name] = result.label.value

        # Determine winner
        if not fired_functions:
            return WeakLabel(
                label=CoordinationLabel.UNKNOWN,
                confidence=0.0,
                labeling_functions_fired=[],
                votes={}
            )

        # Weighted majority vote
        total_weight = sum(votes.values())
        best_label = max(votes.keys(), key=lambda k: votes[k])
        confidence = votes[best_label] / total_weight if total_weight > 0 else 0.0

        # Require minimum confidence for non-UNKNOWN
        if confidence < 0.4:
            return WeakLabel(
                label=CoordinationLabel.UNKNOWN,
                confidence=confidence,
                labeling_functions_fired=fired_functions,
                votes=vote_details
            )

        return WeakLabel(
            label=best_label,
            confidence=confidence,
            labeling_functions_fired=fired_functions,
            votes=vote_details
        )

    def label_clusters_batch(
        self,
        clusters: List[Cluster],
        signals_list: List[CoordinationSignals]
    ) -> List[WeakLabel]:
        """
        Label multiple clusters.

        Args:
            clusters: Clusters to label
            signals_list: Corresponding signals

        Returns:
            List of WeakLabels
        """
        return [
            self.label_cluster(cluster, signals)
            for cluster, signals in zip(clusters, signals_list)
        ]

    def get_high_confidence_labels(
        self,
        clusters: List[Cluster],
        signals_list: List[CoordinationSignals],
        min_confidence: float = 0.7
    ) -> List[tuple[Cluster, WeakLabel]]:
        """
        Get clusters with high-confidence labels for training.

        Args:
            clusters: All clusters
            signals_list: Corresponding signals
            min_confidence: Minimum confidence threshold

        Returns:
            List of (cluster, label) tuples
        """
        results = []

        for cluster, signals in zip(clusters, signals_list):
            label = self.label_cluster(cluster, signals)

            if (label.label != CoordinationLabel.UNKNOWN and
                label.confidence >= min_confidence):
                results.append((cluster, label))

        return results


def get_weak_labeler() -> WeakSupervisionLabeler:
    """Get singleton weak supervision labeler."""
    if not hasattr(get_weak_labeler, "_instance"):
        get_weak_labeler._instance = WeakSupervisionLabeler()
    return get_weak_labeler._instance
