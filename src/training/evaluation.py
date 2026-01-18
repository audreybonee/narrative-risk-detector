"""
Evaluation metrics for coordination detection.
Includes precision@k, lead time, and clustering quality metrics.
"""

from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, silhouette_score, confusion_matrix
)

from src.models.cluster import Cluster
from src.models.signal import Alert, AlertSeverity


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics."""

    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    auc_pr: float = 0.0  # Precision-Recall AUC

    # Precision at k
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)

    # Lead time
    avg_lead_time_hours: float = 0.0
    median_lead_time_hours: float = 0.0
    lead_times: List[float] = field(default_factory=list)

    # Clustering quality
    cluster_purity: float = 0.0
    silhouette: float = 0.0
    n_clusters: int = 0

    # Confusion matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Metadata
    eval_window_start: Optional[datetime] = None
    eval_window_end: Optional[datetime] = None
    n_samples: int = 0


class Evaluator:
    """
    Evaluate coordination detection performance.

    Key metrics:
    - Precision@k: accuracy of top-k alerts
    - Lead time: hours before narrative peak
    - Cluster purity: alignment with ground truth
    - Classification metrics: precision, recall, F1, AUC
    """

    def __init__(
        self,
        k_values: List[int] = None,
    ):
        """
        Initialize evaluator.

        Args:
            k_values: Values of k for precision@k
        """
        self.k_values = k_values or [5, 10, 20, 50]

    def precision_at_k(
        self,
        ranked_items: List[str],
        ground_truth: Dict[str, bool],
        k: int
    ) -> float:
        """
        Compute precision@k.

        Args:
            ranked_items: Items ranked by model (highest first)
            ground_truth: item_id -> is_positive
            k: Number of top items to consider

        Returns:
            Precision at k
        """
        if k == 0 or not ranked_items:
            return 0.0

        top_k = ranked_items[:k]
        true_positives = sum(
            1 for item in top_k
            if ground_truth.get(item, False)
        )

        return true_positives / k

    def recall_at_k(
        self,
        ranked_items: List[str],
        ground_truth: Dict[str, bool],
        k: int
    ) -> float:
        """
        Compute recall@k.

        Args:
            ranked_items: Items ranked by model
            ground_truth: item_id -> is_positive
            k: Number of top items to consider

        Returns:
            Recall at k
        """
        total_positives = sum(ground_truth.values())
        if total_positives == 0:
            return 0.0

        top_k = ranked_items[:k]
        true_positives = sum(
            1 for item in top_k
            if ground_truth.get(item, False)
        )

        return true_positives / total_positives

    def compute_lead_time(
        self,
        alert_time: datetime,
        peak_time: datetime
    ) -> float:
        """
        Compute lead time in hours.

        Positive = detected before peak
        Negative = detected after peak

        Args:
            alert_time: When alert was generated
            peak_time: When narrative peaked

        Returns:
            Lead time in hours
        """
        delta = peak_time - alert_time
        return delta.total_seconds() / 3600

    def evaluate_alerts(
        self,
        alerts: List[Alert],
        ground_truth: Dict[str, bool],
        narrative_peaks: Optional[Dict[str, datetime]] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate alert ranking performance.

        Args:
            alerts: Generated alerts
            ground_truth: alert_id -> is_true_positive
            narrative_peaks: cluster_id -> peak timestamp

        Returns:
            EvaluationMetrics
        """
        if not alerts:
            return EvaluationMetrics()

        # Sort alerts by severity and coordination ratio
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
        }
        ranked_alerts = sorted(
            alerts,
            key=lambda a: (severity_order[a.severity], -a.coordination_ratio)
        )
        ranked_ids = [a.id for a in ranked_alerts]

        # Precision/Recall at k
        precision_at_k = {}
        recall_at_k = {}
        for k in self.k_values:
            if k <= len(ranked_ids):
                precision_at_k[k] = self.precision_at_k(ranked_ids, ground_truth, k)
                recall_at_k[k] = self.recall_at_k(ranked_ids, ground_truth, k)

        # Lead times
        lead_times = []
        if narrative_peaks:
            for alert in alerts:
                if alert.cluster_id in narrative_peaks:
                    lead = self.compute_lead_time(
                        alert.window_end,
                        narrative_peaks[alert.cluster_id]
                    )
                    if lead > 0:  # Only positive lead times
                        lead_times.append(lead)

        avg_lead = np.mean(lead_times) if lead_times else 0.0
        median_lead = np.median(lead_times) if lead_times else 0.0

        # Binary classification metrics
        y_true = [ground_truth.get(a.id, False) for a in alerts]
        y_scores = [a.coordination_ratio for a in alerts]

        if any(y_true) and not all(y_true):
            auc_roc = roc_auc_score(y_true, y_scores)
            auc_pr = average_precision_score(y_true, y_scores)
        else:
            auc_roc = 0.5
            auc_pr = sum(y_true) / len(y_true) if y_true else 0.0

        # Threshold at median for binary metrics
        threshold = np.median(y_scores)
        y_pred = [s >= threshold for s in y_scores]

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            avg_lead_time_hours=avg_lead,
            median_lead_time_hours=median_lead,
            lead_times=lead_times,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            n_samples=len(alerts),
        )

    def evaluate_classifier(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate binary classifier performance.

        Args:
            y_true: True labels (0/1)
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        Returns:
            EvaluationMetrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if len(y_true) > 0 else 0.0
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUC metrics
        auc_roc = 0.5
        auc_pr = 0.0
        if y_proba is not None and len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_proba)
            auc_pr = average_precision_score(y_true, y_proba)

        # Precision at k (using probabilities)
        precision_at_k = {}
        recall_at_k = {}
        if y_proba is not None:
            # Sort by probability
            sorted_idx = np.argsort(-y_proba)
            ground_truth = {str(i): bool(y_true[i]) for i in range(len(y_true))}
            ranked = [str(i) for i in sorted_idx]

            for k in self.k_values:
                if k <= len(ranked):
                    precision_at_k[k] = self.precision_at_k(ranked, ground_truth, k)
                    recall_at_k[k] = self.recall_at_k(ranked, ground_truth, k)

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            n_samples=len(y_true),
        )

    def evaluate_clustering(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        document_labels: Optional[Dict[str, str]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate clustering quality.

        Args:
            embeddings: Document embeddings
            labels: Cluster labels (-1 for noise)
            document_labels: Optional ground truth labels
            document_ids: Document IDs for ground truth lookup

        Returns:
            EvaluationMetrics with clustering metrics
        """
        # Filter out noise
        mask = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Silhouette score
        silhouette = 0.0
        if mask.sum() > 1 and n_clusters > 1:
            try:
                silhouette = silhouette_score(
                    embeddings[mask],
                    labels[mask],
                    metric="cosine"
                )
            except Exception:
                silhouette = 0.0

        # Cluster purity
        purity = 0.0
        if document_labels and document_ids:
            purity = self._compute_purity(labels, document_labels, document_ids)

        return EvaluationMetrics(
            cluster_purity=purity,
            silhouette=silhouette,
            n_clusters=n_clusters,
            n_samples=len(labels),
        )

    def _compute_purity(
        self,
        cluster_labels: np.ndarray,
        ground_truth: Dict[str, str],
        document_ids: List[str],
    ) -> float:
        """
        Compute cluster purity.

        Purity = fraction of documents matching majority label in cluster.

        Args:
            cluster_labels: Assigned cluster labels
            ground_truth: doc_id -> true label
            document_ids: Document IDs

        Returns:
            Purity score (0-1)
        """
        total = 0
        correct = 0

        # Group by cluster
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label != -1:
                clusters[label].append(i)

        for cluster_idx, doc_indices in clusters.items():
            # Get true labels for docs in cluster
            true_labels = []
            for idx in doc_indices:
                doc_id = document_ids[idx]
                if doc_id in ground_truth:
                    true_labels.append(ground_truth[doc_id])

            if not true_labels:
                continue

            # Count majority class
            label_counts = defaultdict(int)
            for label in true_labels:
                label_counts[label] += 1

            majority_count = max(label_counts.values())
            correct += majority_count
            total += len(true_labels)

        return correct / total if total > 0 else 0.0

    def cross_validate(
        self,
        classifier,
        clusters: List[Cluster],
        signals_list,
        labels: List[int],
        n_folds: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Cross-validation with time-based splits.

        Args:
            classifier: Classifier with fit/predict methods
            clusters: All clusters
            signals_list: Corresponding signals
            labels: Ground truth labels
            n_folds: Number of folds

        Returns:
            Dictionary of metric_name -> list of fold scores
        """
        from src.training.time_split import ExpandingWindowSplitter
        from src.models.article import Document

        # Create dummy documents for time splitting
        docs = [
            Document(
                id=c.id,
                source="reddit",
                source_id=c.id,
                title="",
                timestamp=c.window_start,
                score=0,
                comment_count=0,
            )
            for c in clusters
        ]

        splitter = ExpandingWindowSplitter(n_splits=n_folds)

        metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
        }

        for train_docs, test_docs, _ in splitter.split(docs):
            train_ids = {d.id for d in train_docs}
            test_ids = {d.id for d in test_docs}

            train_idx = [i for i, c in enumerate(clusters) if c.id in train_ids]
            test_idx = [i for i, c in enumerate(clusters) if c.id in test_ids]

            if not train_idx or not test_idx:
                continue

            # Train
            train_clusters = [clusters[i] for i in train_idx]
            train_signals = [signals_list[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]

            # Create new classifier instance
            from src.training.classifier import CoordinationClassifier
            fold_classifier = CoordinationClassifier(model_type=classifier.model_type)

            try:
                fold_classifier.fit(train_clusters, train_signals, train_labels)
            except Exception:
                continue

            # Test
            test_clusters = [clusters[i] for i in test_idx]
            test_signals = [signals_list[i] for i in test_idx]
            test_labels = np.array([labels[i] for i in test_idx])

            y_pred, y_proba = fold_classifier.predict_batch(test_clusters, test_signals)

            # Compute metrics
            fold_metrics = self.evaluate_classifier(test_labels, y_pred, y_proba)

            metrics["accuracy"].append(fold_metrics.accuracy)
            metrics["precision"].append(fold_metrics.precision)
            metrics["recall"].append(fold_metrics.recall)
            metrics["f1"].append(fold_metrics.f1)
            metrics["auc"].append(fold_metrics.auc_roc)

        return metrics

    def summarize_cv_results(
        self,
        cv_results: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Summarize cross-validation results.

        Args:
            cv_results: Output from cross_validate

        Returns:
            Dictionary with mean and std for each metric
        """
        summary = {}
        for metric, scores in cv_results.items():
            if scores:
                summary[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "n_folds": len(scores),
                }
            else:
                summary[metric] = {"mean": 0.0, "std": 0.0, "n_folds": 0}

        return summary


def get_evaluator() -> Evaluator:
    """Get singleton evaluator instance."""
    if not hasattr(get_evaluator, "_instance"):
        get_evaluator._instance = Evaluator()
    return get_evaluator._instance
