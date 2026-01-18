"""
Active learning pipeline for analyst feedback integration.
Manages the feedback loop between model predictions and human review.
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

from src.models.signal import Alert
from src.models.cluster import Cluster
from src.config import settings


@dataclass
class AnalystFeedback:
    """Analyst feedback on an alert or cluster."""
    alert_id: str
    analyst_id: str
    timestamp: datetime
    verdict: str  # confirmed, false_positive, needs_review, unsure
    confidence: float = 1.0  # Analyst's confidence in their verdict
    notes: Optional[str] = None
    time_spent_seconds: Optional[int] = None


@dataclass
class UncertaintySample:
    """Sample selected for labeling based on uncertainty."""
    alert_id: str
    cluster_id: str
    model_confidence: float
    uncertainty_score: float
    selection_reason: str


class ActiveLearningManager:
    """
    Manage active learning feedback loop.

    Responsibilities:
    - Track analyst feedback
    - Select high-uncertainty samples for review
    - Trigger model retraining
    - Monitor annotation quality
    """

    def __init__(
        self,
        retrain_threshold: Optional[int] = None,
        feedback_store_path: Optional[Path] = None,
    ):
        """
        Initialize active learning manager.

        Args:
            retrain_threshold: Number of new labels before triggering retrain
            feedback_store_path: Path to persist feedback (optional)
        """
        self.retrain_threshold = retrain_threshold or settings.retrain_threshold
        self.feedback_store_path = feedback_store_path

        # In-memory feedback buffer
        self.feedback_buffer: List[AnalystFeedback] = []

        # Load existing feedback if store exists
        if self.feedback_store_path and self.feedback_store_path.exists():
            self._load_feedback()

    def _load_feedback(self) -> None:
        """Load feedback from persistent store."""
        if not self.feedback_store_path:
            return

        try:
            with open(self.feedback_store_path) as f:
                data = json.load(f)

            self.feedback_buffer = [
                AnalystFeedback(
                    alert_id=item["alert_id"],
                    analyst_id=item["analyst_id"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    verdict=item["verdict"],
                    confidence=item.get("confidence", 1.0),
                    notes=item.get("notes"),
                    time_spent_seconds=item.get("time_spent_seconds"),
                )
                for item in data
            ]
        except Exception:
            self.feedback_buffer = []

    def _save_feedback(self) -> None:
        """Save feedback to persistent store."""
        if not self.feedback_store_path:
            return

        data = [
            {
                "alert_id": fb.alert_id,
                "analyst_id": fb.analyst_id,
                "timestamp": fb.timestamp.isoformat(),
                "verdict": fb.verdict,
                "confidence": fb.confidence,
                "notes": fb.notes,
                "time_spent_seconds": fb.time_spent_seconds,
            }
            for fb in self.feedback_buffer
        ]

        self.feedback_store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_store_path, "w") as f:
            json.dump(data, f, indent=2)

    def record_feedback(self, feedback: AnalystFeedback) -> None:
        """
        Record analyst feedback.

        Args:
            feedback: Analyst feedback to record
        """
        self.feedback_buffer.append(feedback)
        self._save_feedback()

    def should_retrain(self) -> bool:
        """Check if we have enough new labels to retrain."""
        return len(self.feedback_buffer) >= self.retrain_threshold

    def get_training_labels(self) -> List[Tuple[str, str, float]]:
        """
        Get (alert_id, label, weight) tuples for training.

        Converts verdicts to labels:
        - confirmed -> coordinated (1)
        - false_positive -> organic (0)
        - needs_review, unsure -> excluded

        Weight is based on analyst confidence.
        """
        labels = []

        # Group by alert_id, take most recent
        alert_feedback: Dict[str, AnalystFeedback] = {}
        for fb in self.feedback_buffer:
            if (fb.alert_id not in alert_feedback or
                fb.timestamp > alert_feedback[fb.alert_id].timestamp):
                alert_feedback[fb.alert_id] = fb

        for alert_id, fb in alert_feedback.items():
            if fb.verdict == "confirmed":
                labels.append((alert_id, "coordinated", fb.confidence))
            elif fb.verdict == "false_positive":
                labels.append((alert_id, "organic", fb.confidence))
            # Skip needs_review and unsure

        return labels

    def clear_buffer(self) -> None:
        """Clear feedback buffer after retraining."""
        self.feedback_buffer = []
        self._save_feedback()

    def select_for_labeling(
        self,
        alerts: List[Alert],
        model_confidences: Dict[str, float],
        n_samples: int = 10,
        strategy: str = "uncertainty"
    ) -> List[UncertaintySample]:
        """
        Select alerts for analyst review using active learning.

        Strategies:
        - uncertainty: Select most uncertain predictions
        - diversity: Select diverse samples
        - mixed: Combine uncertainty and diversity

        Args:
            alerts: Candidate alerts
            model_confidences: Model confidence scores by alert_id
            n_samples: Number of samples to select
            strategy: Selection strategy

        Returns:
            List of UncertaintySample objects
        """
        # Filter out already-reviewed alerts
        reviewed_ids = {fb.alert_id for fb in self.feedback_buffer}
        unreviewed = [a for a in alerts if a.id not in reviewed_ids]

        if not unreviewed:
            return []

        samples = []

        if strategy == "uncertainty":
            # Uncertainty sampling: select where model is most uncertain
            for alert in unreviewed:
                conf = model_confidences.get(alert.id, 0.5)
                # Uncertainty is highest when confidence is near 0.5
                uncertainty = 1 - abs(conf - 0.5) * 2

                samples.append(UncertaintySample(
                    alert_id=alert.id,
                    cluster_id=alert.cluster_id,
                    model_confidence=conf,
                    uncertainty_score=uncertainty,
                    selection_reason="high_uncertainty"
                ))

            # Sort by uncertainty descending
            samples.sort(key=lambda x: x.uncertainty_score, reverse=True)

        elif strategy == "diversity":
            # Simple diversity: select from different severity levels
            by_severity: Dict[str, List[Alert]] = defaultdict(list)
            for alert in unreviewed:
                by_severity[alert.severity.value].append(alert)

            # Round-robin from each severity
            while len(samples) < n_samples:
                for severity in ["critical", "high", "medium", "low"]:
                    if by_severity[severity] and len(samples) < n_samples:
                        alert = by_severity[severity].pop(0)
                        conf = model_confidences.get(alert.id, 0.5)
                        samples.append(UncertaintySample(
                            alert_id=alert.id,
                            cluster_id=alert.cluster_id,
                            model_confidence=conf,
                            uncertainty_score=1 - abs(conf - 0.5) * 2,
                            selection_reason=f"diversity_{severity}"
                        ))

        else:  # mixed
            # Half uncertainty, half diversity
            uncertainty_samples = self.select_for_labeling(
                unreviewed,
                model_confidences,
                n_samples // 2,
                "uncertainty"
            )
            remaining = [a for a in unreviewed
                        if a.id not in {s.alert_id for s in uncertainty_samples}]
            diversity_samples = self.select_for_labeling(
                remaining,
                model_confidences,
                n_samples - len(uncertainty_samples),
                "diversity"
            )
            samples = uncertainty_samples + diversity_samples

        return samples[:n_samples]

    def compute_metrics(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Compute feedback metrics for monitoring.

        Args:
            since: Only include feedback after this time

        Returns:
            Dictionary of metrics
        """
        if since is None:
            since = datetime.utcnow() - timedelta(days=7)

        recent = [
            fb for fb in self.feedback_buffer
            if fb.timestamp >= since
        ]

        if not recent:
            return {
                "total_reviewed": 0,
                "confirmation_rate": 0.0,
                "false_positive_rate": 0.0,
                "unsure_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_time_seconds": 0.0,
            }

        confirmed = sum(1 for fb in recent if fb.verdict == "confirmed")
        false_pos = sum(1 for fb in recent if fb.verdict == "false_positive")
        unsure = sum(1 for fb in recent if fb.verdict in ("needs_review", "unsure"))

        times = [fb.time_spent_seconds for fb in recent if fb.time_spent_seconds]

        return {
            "total_reviewed": len(recent),
            "confirmation_rate": confirmed / len(recent),
            "false_positive_rate": false_pos / len(recent),
            "unsure_rate": unsure / len(recent),
            "avg_confidence": sum(fb.confidence for fb in recent) / len(recent),
            "avg_time_seconds": sum(times) / len(times) if times else 0.0,
        }

    def get_analyst_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get per-analyst statistics.

        Returns:
            Dictionary of analyst_id -> stats
        """
        stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "confirmed": 0, "false_positive": 0, "unsure": 0}
        )

        for fb in self.feedback_buffer:
            stats[fb.analyst_id]["total"] += 1
            if fb.verdict == "confirmed":
                stats[fb.analyst_id]["confirmed"] += 1
            elif fb.verdict == "false_positive":
                stats[fb.analyst_id]["false_positive"] += 1
            else:
                stats[fb.analyst_id]["unsure"] += 1

        return dict(stats)

    def estimate_label_quality(self) -> float:
        """
        Estimate label quality based on inter-annotator agreement.

        Returns:
            Agreement score (0-1)
        """
        # Group feedback by alert
        by_alert: Dict[str, List[AnalystFeedback]] = defaultdict(list)
        for fb in self.feedback_buffer:
            by_alert[fb.alert_id].append(fb)

        # Only consider alerts with multiple annotations
        multi_annotated = [
            feedbacks for feedbacks in by_alert.values()
            if len(feedbacks) > 1
        ]

        if not multi_annotated:
            return 1.0  # No disagreements if single annotator

        # Count agreements
        agreements = 0
        total_pairs = 0

        for feedbacks in multi_annotated:
            verdicts = [fb.verdict for fb in feedbacks]
            for i in range(len(verdicts)):
                for j in range(i + 1, len(verdicts)):
                    total_pairs += 1
                    if verdicts[i] == verdicts[j]:
                        agreements += 1

        return agreements / total_pairs if total_pairs > 0 else 1.0


def get_active_learning_manager() -> ActiveLearningManager:
    """Get singleton active learning manager."""
    if not hasattr(get_active_learning_manager, "_instance"):
        get_active_learning_manager._instance = ActiveLearningManager(
            feedback_store_path=Path(settings.processed_data_dir) / "feedback.json"
        )
    return get_active_learning_manager._instance
