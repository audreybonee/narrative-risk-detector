"""
Drift monitoring for embeddings and topic distributions.
Detects when model performance may degrade due to data distribution shift.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from src.models.document import DocumentWithEmbedding
from src.models.cluster import Cluster


@dataclass
class DriftMetrics:
    """Metrics from drift detection."""

    timestamp: datetime

    # Embedding drift
    embedding_mean_shift: float = 0.0
    embedding_variance_change: float = 0.0
    embedding_cosine_drift: float = 0.0

    # Topic drift
    topic_distribution_drift: float = 0.0  # Jensen-Shannon divergence
    new_topics_ratio: float = 0.0
    disappeared_topics_ratio: float = 0.0

    # Velocity drift
    volume_change_ratio: float = 0.0
    velocity_zscore: float = 0.0

    # Coordination drift
    coordination_ratio_drift: float = 0.0
    alert_rate_change: float = 0.0

    # Overall
    drift_detected: bool = False
    drift_severity: str = "none"  # none, low, medium, high

    # Details
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    n_documents: int = 0
    n_clusters: int = 0


@dataclass
class BaselineStatistics:
    """Baseline statistics for drift comparison."""

    # Embedding statistics
    embedding_mean: np.ndarray = field(default_factory=lambda: np.array([]))
    embedding_std: np.ndarray = field(default_factory=lambda: np.array([]))
    embedding_centroid: np.ndarray = field(default_factory=lambda: np.array([]))

    # Topic distribution
    topic_distribution: Dict[int, float] = field(default_factory=dict)
    topic_labels: List[int] = field(default_factory=list)

    # Volume statistics
    avg_volume_per_window: float = 0.0
    std_volume_per_window: float = 0.0

    # Coordination statistics
    avg_coordination_ratio: float = 0.0
    std_coordination_ratio: float = 0.0
    avg_alert_rate: float = 0.0

    # Metadata
    baseline_start: Optional[datetime] = None
    baseline_end: Optional[datetime] = None
    n_documents: int = 0
    n_windows: int = 0


class DriftMonitor:
    """
    Monitor embedding and topic drift to detect distribution shifts.

    Drift can indicate:
    - New coordination tactics
    - Emerging narratives
    - Model degradation requiring retraining

    Uses multiple drift detection methods:
    - Embedding space: mean shift, variance change, centroid distance
    - Topic distribution: Jensen-Shannon divergence
    - Volume: posting velocity changes
    - Coordination: ratio distribution shifts
    """

    def __init__(
        self,
        window_size_hours: int = 24,
        baseline_windows: int = 7,
        embedding_threshold: float = 0.15,
        topic_threshold: float = 0.2,
        volume_zscore_threshold: float = 3.0,
        coordination_threshold: float = 0.5,
    ):
        """
        Initialize drift monitor.

        Args:
            window_size_hours: Size of monitoring windows
            baseline_windows: Number of windows for baseline
            embedding_threshold: Threshold for embedding drift
            topic_threshold: Threshold for topic drift (JS divergence)
            volume_zscore_threshold: Z-score threshold for volume anomaly
            coordination_threshold: Threshold for coordination ratio drift
        """
        self.window_size_hours = window_size_hours
        self.baseline_windows = baseline_windows
        self.embedding_threshold = embedding_threshold
        self.topic_threshold = topic_threshold
        self.volume_zscore_threshold = volume_zscore_threshold
        self.coordination_threshold = coordination_threshold

        # Baseline statistics
        self.baseline: Optional[BaselineStatistics] = None

        # Historical metrics for trend analysis
        self.metrics_history: deque = deque(maxlen=100)

        # Window buffers
        self._embedding_buffer: List[np.ndarray] = []
        self._cluster_buffer: List[Cluster] = []
        self._window_volumes: deque = deque(maxlen=baseline_windows * 2)

    def compute_baseline(
        self,
        documents: List[DocumentWithEmbedding],
        clusters: List[Cluster],
    ) -> BaselineStatistics:
        """
        Compute baseline statistics from historical data.

        Args:
            documents: Historical documents with embeddings
            clusters: Historical clusters

        Returns:
            BaselineStatistics
        """
        if not documents:
            return BaselineStatistics()

        # Extract embeddings
        embeddings = np.array([d.embedding for d in documents])

        # Embedding statistics
        embedding_mean = np.mean(embeddings, axis=0)
        embedding_std = np.std(embeddings, axis=0)
        embedding_centroid = embedding_mean  # Centroid is the mean

        # Topic distribution from cluster labels
        cluster_labels = []
        for doc in documents:
            # Find which cluster this doc belongs to
            for cluster in clusters:
                if doc.id in cluster.document_ids:
                    cluster_labels.append(cluster.id)
                    break
            else:
                cluster_labels.append(-1)  # Noise

        # Compute topic distribution
        unique_labels = set(cluster_labels)
        topic_distribution = {}
        for label in unique_labels:
            if label != -1:
                count = cluster_labels.count(label)
                topic_distribution[label] = count / len(cluster_labels)

        # Volume statistics (documents per window)
        sorted_docs = sorted(documents, key=lambda d: d.timestamp)
        min_time = sorted_docs[0].timestamp
        max_time = sorted_docs[-1].timestamp

        window_delta = timedelta(hours=self.window_size_hours)
        window_volumes = []
        current_start = min_time

        while current_start < max_time:
            current_end = current_start + window_delta
            count = sum(1 for d in documents if current_start <= d.timestamp < current_end)
            window_volumes.append(count)
            current_start = current_end

        avg_volume = np.mean(window_volumes) if window_volumes else 0.0
        std_volume = np.std(window_volumes) if window_volumes else 1.0

        # Coordination statistics
        coordination_ratios = [c.coordination_ratio for c in clusters if c.size > 1]
        avg_coord_ratio = np.mean(coordination_ratios) if coordination_ratios else 1.0
        std_coord_ratio = np.std(coordination_ratios) if coordination_ratios else 0.5

        self.baseline = BaselineStatistics(
            embedding_mean=embedding_mean,
            embedding_std=embedding_std,
            embedding_centroid=embedding_centroid,
            topic_distribution=topic_distribution,
            topic_labels=list(unique_labels - {-1}),
            avg_volume_per_window=avg_volume,
            std_volume_per_window=std_volume if std_volume > 0 else 1.0,
            avg_coordination_ratio=avg_coord_ratio,
            std_coordination_ratio=std_coord_ratio if std_coord_ratio > 0 else 0.5,
            baseline_start=min_time,
            baseline_end=max_time,
            n_documents=len(documents),
            n_windows=len(window_volumes),
        )

        return self.baseline

    def detect_drift(
        self,
        documents: List[DocumentWithEmbedding],
        clusters: List[Cluster],
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
    ) -> DriftMetrics:
        """
        Detect drift in current window vs baseline.

        Args:
            documents: Current window documents
            clusters: Current window clusters
            window_start: Window start time
            window_end: Window end time

        Returns:
            DriftMetrics with drift indicators
        """
        if self.baseline is None:
            # No baseline - compute from current data
            self.compute_baseline(documents, clusters)
            return DriftMetrics(
                timestamp=datetime.now(),
                drift_detected=False,
                drift_severity="none",
                window_start=window_start,
                window_end=window_end,
                n_documents=len(documents),
                n_clusters=len(clusters),
            )

        if not documents:
            return DriftMetrics(
                timestamp=datetime.now(),
                drift_detected=False,
                drift_severity="none",
            )

        # Compute current statistics
        embeddings = np.array([d.embedding for d in documents])

        # 1. Embedding drift
        embedding_metrics = self._compute_embedding_drift(embeddings)

        # 2. Topic drift
        topic_metrics = self._compute_topic_drift(documents, clusters)

        # 3. Volume drift
        volume_metrics = self._compute_volume_drift(len(documents))

        # 4. Coordination drift
        coord_metrics = self._compute_coordination_drift(clusters)

        # Determine overall drift
        drift_scores = [
            embedding_metrics["cosine_drift"] / self.embedding_threshold,
            topic_metrics["js_divergence"] / self.topic_threshold,
            abs(volume_metrics["zscore"]) / self.volume_zscore_threshold,
            coord_metrics["ratio_drift"] / self.coordination_threshold,
        ]

        max_drift = max(drift_scores)

        if max_drift >= 1.5:
            severity = "high"
            detected = True
        elif max_drift >= 1.0:
            severity = "medium"
            detected = True
        elif max_drift >= 0.7:
            severity = "low"
            detected = True
        else:
            severity = "none"
            detected = False

        metrics = DriftMetrics(
            timestamp=datetime.now(),
            embedding_mean_shift=embedding_metrics["mean_shift"],
            embedding_variance_change=embedding_metrics["variance_change"],
            embedding_cosine_drift=embedding_metrics["cosine_drift"],
            topic_distribution_drift=topic_metrics["js_divergence"],
            new_topics_ratio=topic_metrics["new_topics_ratio"],
            disappeared_topics_ratio=topic_metrics["disappeared_ratio"],
            volume_change_ratio=volume_metrics["change_ratio"],
            velocity_zscore=volume_metrics["zscore"],
            coordination_ratio_drift=coord_metrics["ratio_drift"],
            alert_rate_change=coord_metrics["alert_rate_change"],
            drift_detected=detected,
            drift_severity=severity,
            window_start=window_start,
            window_end=window_end,
            n_documents=len(documents),
            n_clusters=len(clusters),
        )

        # Store for trend analysis
        self.metrics_history.append(metrics)

        return metrics

    def _compute_embedding_drift(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute embedding space drift metrics."""
        current_mean = np.mean(embeddings, axis=0)
        current_std = np.std(embeddings, axis=0)

        # Mean shift (Euclidean distance)
        mean_shift = np.linalg.norm(current_mean - self.baseline.embedding_mean)

        # Variance change
        baseline_var = np.mean(self.baseline.embedding_std ** 2)
        current_var = np.mean(current_std ** 2)
        variance_change = abs(current_var - baseline_var) / (baseline_var + 1e-6)

        # Cosine drift from centroid
        cosine_sim = np.dot(current_mean, self.baseline.embedding_centroid) / (
            np.linalg.norm(current_mean) * np.linalg.norm(self.baseline.embedding_centroid) + 1e-6
        )
        cosine_drift = 1 - cosine_sim

        return {
            "mean_shift": float(mean_shift),
            "variance_change": float(variance_change),
            "cosine_drift": float(cosine_drift),
        }

    def _compute_topic_drift(
        self,
        documents: List[DocumentWithEmbedding],
        clusters: List[Cluster],
    ) -> Dict[str, float]:
        """Compute topic distribution drift."""
        # Build current topic distribution
        current_topics: Dict[int, int] = {}
        for cluster in clusters:
            cluster_size = len([d for d in documents if d.id in cluster.document_ids])
            if cluster_size > 0:
                current_topics[cluster.id] = cluster_size

        if not current_topics:
            return {
                "js_divergence": 0.0,
                "new_topics_ratio": 0.0,
                "disappeared_ratio": 0.0,
            }

        # Normalize to distribution
        total = sum(current_topics.values())
        current_dist = {k: v / total for k, v in current_topics.items()}

        # Align distributions for JS divergence
        all_topics = set(self.baseline.topic_labels) | set(current_topics.keys())

        p = np.array([self.baseline.topic_distribution.get(t, 0.0) for t in all_topics])
        q = np.array([current_dist.get(t, 0.0) for t in all_topics])

        # Add small epsilon to avoid division by zero
        p = p + 1e-10
        q = q + 1e-10
        p = p / p.sum()
        q = q / q.sum()

        js_div = jensenshannon(p, q)

        # New and disappeared topics
        baseline_topics = set(self.baseline.topic_labels)
        current_topic_ids = set(current_topics.keys())

        new_topics = current_topic_ids - baseline_topics
        disappeared = baseline_topics - current_topic_ids

        new_ratio = len(new_topics) / len(current_topic_ids) if current_topic_ids else 0.0
        disappeared_ratio = len(disappeared) / len(baseline_topics) if baseline_topics else 0.0

        return {
            "js_divergence": float(js_div),
            "new_topics_ratio": float(new_ratio),
            "disappeared_ratio": float(disappeared_ratio),
        }

    def _compute_volume_drift(self, current_volume: int) -> Dict[str, float]:
        """Compute volume drift metrics."""
        self._window_volumes.append(current_volume)

        expected = self.baseline.avg_volume_per_window
        std = self.baseline.std_volume_per_window

        change_ratio = (current_volume - expected) / (expected + 1e-6)
        zscore = (current_volume - expected) / (std + 1e-6)

        return {
            "change_ratio": float(change_ratio),
            "zscore": float(zscore),
        }

    def _compute_coordination_drift(self, clusters: List[Cluster]) -> Dict[str, float]:
        """Compute coordination pattern drift."""
        if not clusters:
            return {
                "ratio_drift": 0.0,
                "alert_rate_change": 0.0,
            }

        # Current coordination ratios
        ratios = [c.coordination_ratio for c in clusters if c.size > 1]
        if not ratios:
            return {
                "ratio_drift": 0.0,
                "alert_rate_change": 0.0,
            }

        current_avg = np.mean(ratios)

        # Drift from baseline
        ratio_drift = abs(current_avg - self.baseline.avg_coordination_ratio) / (
            self.baseline.std_coordination_ratio + 1e-6
        )

        # Alert rate (clusters with high coordination)
        high_coord_threshold = self.baseline.avg_coordination_ratio + 2 * self.baseline.std_coordination_ratio
        high_coord_count = sum(1 for r in ratios if r > high_coord_threshold)
        current_alert_rate = high_coord_count / len(ratios)

        alert_rate_change = current_alert_rate - self.baseline.avg_alert_rate

        return {
            "ratio_drift": float(ratio_drift),
            "alert_rate_change": float(alert_rate_change),
        }

    def update_baseline(
        self,
        documents: List[DocumentWithEmbedding],
        clusters: List[Cluster],
        decay_factor: float = 0.1,
    ) -> None:
        """
        Update baseline with new data using exponential smoothing.

        Args:
            documents: New documents
            clusters: New clusters
            decay_factor: Weight for new data (0-1)
        """
        if not documents or self.baseline is None:
            return

        embeddings = np.array([d.embedding for d in documents])
        new_mean = np.mean(embeddings, axis=0)
        new_std = np.std(embeddings, axis=0)

        # Exponential smoothing
        self.baseline.embedding_mean = (
            (1 - decay_factor) * self.baseline.embedding_mean +
            decay_factor * new_mean
        )
        self.baseline.embedding_std = (
            (1 - decay_factor) * self.baseline.embedding_std +
            decay_factor * new_std
        )
        self.baseline.embedding_centroid = self.baseline.embedding_mean

        # Update volume statistics
        new_volume = len(documents)
        self.baseline.avg_volume_per_window = (
            (1 - decay_factor) * self.baseline.avg_volume_per_window +
            decay_factor * new_volume
        )

        # Update coordination statistics
        ratios = [c.coordination_ratio for c in clusters if c.size > 1]
        if ratios:
            new_avg_ratio = np.mean(ratios)
            self.baseline.avg_coordination_ratio = (
                (1 - decay_factor) * self.baseline.avg_coordination_ratio +
                decay_factor * new_avg_ratio
            )

        self.baseline.n_documents += len(documents)
        self.baseline.n_windows += 1
        self.baseline.baseline_end = datetime.now()

    def get_drift_trend(self, n_windows: int = 10) -> Dict[str, List[float]]:
        """
        Get drift trends over recent windows.

        Args:
            n_windows: Number of recent windows

        Returns:
            Dictionary of metric -> list of values
        """
        recent = list(self.metrics_history)[-n_windows:]

        return {
            "embedding_drift": [m.embedding_cosine_drift for m in recent],
            "topic_drift": [m.topic_distribution_drift for m in recent],
            "volume_zscore": [m.velocity_zscore for m in recent],
            "coordination_drift": [m.coordination_ratio_drift for m in recent],
            "timestamps": [m.timestamp for m in recent],
        }

    def should_retrain(self, consecutive_windows: int = 3) -> Tuple[bool, str]:
        """
        Determine if model should be retrained based on drift.

        Args:
            consecutive_windows: Number of consecutive high-drift windows

        Returns:
            Tuple of (should_retrain, reason)
        """
        recent = list(self.metrics_history)[-consecutive_windows:]

        if len(recent) < consecutive_windows:
            return False, "Not enough data"

        # Check for consistent high drift
        high_drift_count = sum(1 for m in recent if m.drift_severity in ("medium", "high"))

        if high_drift_count >= consecutive_windows:
            severities = [m.drift_severity for m in recent]
            return True, f"Consistent drift detected: {severities}"

        # Check for sustained embedding drift
        embedding_drifts = [m.embedding_cosine_drift for m in recent]
        if all(d > self.embedding_threshold * 0.8 for d in embedding_drifts):
            return True, f"Sustained embedding drift: avg={np.mean(embedding_drifts):.3f}"

        # Check for sustained topic drift
        topic_drifts = [m.topic_distribution_drift for m in recent]
        if all(d > self.topic_threshold * 0.8 for d in topic_drifts):
            return True, f"Sustained topic drift: avg={np.mean(topic_drifts):.3f}"

        return False, "Drift within acceptable range"

    def to_dict(self) -> Dict:
        """Export monitor state for persistence."""
        return {
            "window_size_hours": self.window_size_hours,
            "baseline_windows": self.baseline_windows,
            "thresholds": {
                "embedding": self.embedding_threshold,
                "topic": self.topic_threshold,
                "volume_zscore": self.volume_zscore_threshold,
                "coordination": self.coordination_threshold,
            },
            "baseline": {
                "embedding_mean": self.baseline.embedding_mean.tolist() if self.baseline else [],
                "embedding_std": self.baseline.embedding_std.tolist() if self.baseline else [],
                "avg_volume": self.baseline.avg_volume_per_window if self.baseline else 0,
                "avg_coordination": self.baseline.avg_coordination_ratio if self.baseline else 0,
                "n_documents": self.baseline.n_documents if self.baseline else 0,
            },
            "n_history": len(self.metrics_history),
        }


def get_drift_monitor() -> DriftMonitor:
    """Get singleton drift monitor instance."""
    if not hasattr(get_drift_monitor, "_instance"):
        get_drift_monitor._instance = DriftMonitor()
    return get_drift_monitor._instance
