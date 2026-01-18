"""
Coordination likelihood classifier.
Trained on weak labels + analyst feedback to predict coordination.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from src.models.cluster import Cluster
from src.processing.coordination import CoordinationSignals
from src.labeling.weak_supervision import WeakLabel, CoordinationLabel


class CoordinationClassifier:
    """
    Lightweight classifier for coordination likelihood.

    Features extracted from clusters and coordination signals:
    - Coordination ratio
    - Duplicate burst score
    - Temporal velocity
    - Semantic coherence
    - Cluster size metrics

    Trained on:
    - Weak supervision pseudo-labels
    - Analyst feedback (active learning)
    """

    FEATURE_NAMES = [
        "coordination_ratio",
        "duplicate_burst_score",
        "duplicate_groups",
        "max_duplicate_group_size",
        "posting_velocity",
        "velocity_zscore",
        "burst_detected",
        "semantic_coherence",
        "topic_concentration",
        "cluster_size",
        "unique_authors",
        "has_body_ratio",
        "log_cluster_size",
        "ratio_squared",
    ]

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        calibrate: bool = True,
    ):
        """
        Initialize classifier.

        Args:
            model_type: Model type (gradient_boosting, random_forest, logistic)
            calibrate: Whether to calibrate probabilities
        """
        self.model_type = model_type
        self.calibrate = calibrate
        self.scaler = StandardScaler()

        # Initialize base model
        if model_type == "gradient_boosting":
            self._base_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == "random_forest":
            self._base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42
            )
        else:  # logistic
            self._base_model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )

        self.model = self._base_model
        self._is_fitted = False
        self._feature_importance: Optional[Dict[str, float]] = None

    def _extract_features(
        self,
        cluster: Cluster,
        signals: CoordinationSignals,
        has_body_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Extract feature vector from cluster and signals.

        Args:
            cluster: Cluster object
            signals: Coordination signals
            has_body_ratio: Fraction of docs with body content

        Returns:
            Feature vector
        """
        return np.array([
            signals.coordination_ratio,
            signals.duplicate_burst_score,
            signals.duplicate_groups,
            signals.max_duplicate_group_size,
            signals.posting_velocity,
            signals.velocity_zscore,
            float(signals.burst_detected),
            signals.semantic_coherence,
            signals.topic_concentration,
            cluster.size,
            cluster.unique_authors,
            has_body_ratio,
            np.log1p(cluster.size),  # Log transform of size
            signals.coordination_ratio ** 2,  # Squared ratio for nonlinearity
        ])

    def _prepare_training_data(
        self,
        clusters: List[Cluster],
        signals_list: List[CoordinationSignals],
        labels: List[int],
        has_body_ratios: Optional[List[float]] = None,
        sample_weights: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Prepare training data.

        Args:
            clusters: Cluster objects
            signals_list: Corresponding signals
            labels: Binary labels (0=organic, 1=coordinated)
            has_body_ratios: Body content ratios per cluster
            sample_weights: Optional sample weights

        Returns:
            Tuple of (X, y, weights)
        """
        if has_body_ratios is None:
            has_body_ratios = [0.5] * len(clusters)

        X = np.array([
            self._extract_features(c, s, h)
            for c, s, h in zip(clusters, signals_list, has_body_ratios)
        ])
        y = np.array(labels)
        weights = np.array(sample_weights) if sample_weights else None

        return X, y, weights

    def fit(
        self,
        clusters: List[Cluster],
        signals_list: List[CoordinationSignals],
        labels: List[int],
        has_body_ratios: Optional[List[float]] = None,
        sample_weights: Optional[List[float]] = None,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train the classifier.

        Args:
            clusters: Training clusters
            signals_list: Corresponding signals
            labels: Binary labels (0=organic, 1=coordinated)
            has_body_ratios: Body content ratios
            sample_weights: Sample weights (e.g., from analyst confidence)
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        X, y, weights = self._prepare_training_data(
            clusters, signals_list, labels, has_body_ratios, sample_weights
        )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split for validation if calibrating
        if self.calibrate and len(X) > 50:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42, stratify=y
            )

            # Fit base model
            if weights is not None:
                w_train, _ = train_test_split(
                    weights, test_size=validation_split, random_state=42, stratify=y
                )
                self._base_model.fit(X_train, y_train, sample_weight=w_train)
            else:
                self._base_model.fit(X_train, y_train)

            # Calibrate
            self.model = CalibratedClassifierCV(
                self._base_model, method="isotonic", cv="prefit"
            )
            self.model.fit(X_val, y_val)
        else:
            # Just fit without calibration
            if weights is not None:
                self._base_model.fit(X_scaled, y, sample_weight=weights)
            else:
                self._base_model.fit(X_scaled, y)
            self.model = self._base_model

        self._is_fitted = True

        # Compute feature importance
        self._compute_feature_importance()

        # Return training metrics
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5,
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int(len(y) - y.sum()),
        }

    def fit_from_weak_labels(
        self,
        clusters: List[Cluster],
        signals_list: List[CoordinationSignals],
        weak_labels: List[WeakLabel],
        has_body_ratios: Optional[List[float]] = None,
        min_confidence: float = 0.5,
    ) -> Dict[str, float]:
        """
        Train from weak supervision labels.

        Args:
            clusters: Clusters
            signals_list: Signals
            weak_labels: Weak supervision labels
            has_body_ratios: Body ratios
            min_confidence: Minimum confidence to include

        Returns:
            Training metrics
        """
        # Filter to usable labels
        filtered_data = []
        for cluster, signals, weak_label, ratio in zip(
            clusters, signals_list, weak_labels,
            has_body_ratios or [0.5] * len(clusters)
        ):
            if weak_label.label == CoordinationLabel.UNKNOWN:
                continue
            if weak_label.confidence < min_confidence:
                continue

            # Convert to binary
            if weak_label.label == CoordinationLabel.ORGANIC:
                label = 0
            elif weak_label.label in (CoordinationLabel.SUSPICIOUS, CoordinationLabel.LIKELY_COORDINATED):
                label = 1
            else:
                continue

            filtered_data.append((cluster, signals, label, ratio, weak_label.confidence))

        if len(filtered_data) < 10:
            raise ValueError(f"Not enough labeled data: {len(filtered_data)} samples")

        clusters_f = [d[0] for d in filtered_data]
        signals_f = [d[1] for d in filtered_data]
        labels_f = [d[2] for d in filtered_data]
        ratios_f = [d[3] for d in filtered_data]
        weights_f = [d[4] for d in filtered_data]

        return self.fit(
            clusters_f, signals_f, labels_f,
            has_body_ratios=ratios_f,
            sample_weights=weights_f
        )

    def predict(
        self,
        cluster: Cluster,
        signals: CoordinationSignals,
        has_body_ratio: float = 0.5
    ) -> int:
        """
        Predict coordination label.

        Args:
            cluster: Cluster to classify
            signals: Coordination signals
            has_body_ratio: Body content ratio

        Returns:
            0 (organic) or 1 (coordinated)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        X = self._extract_features(cluster, signals, has_body_ratio)
        X_scaled = self.scaler.transform(X.reshape(1, -1))

        return int(self.model.predict(X_scaled)[0])

    def predict_proba(
        self,
        cluster: Cluster,
        signals: CoordinationSignals,
        has_body_ratio: float = 0.5
    ) -> float:
        """
        Predict coordination probability.

        Args:
            cluster: Cluster to classify
            signals: Coordination signals
            has_body_ratio: Body content ratio

        Returns:
            Probability of coordination (0-1)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        X = self._extract_features(cluster, signals, has_body_ratio)
        X_scaled = self.scaler.transform(X.reshape(1, -1))

        return float(self.model.predict_proba(X_scaled)[0, 1])

    def predict_batch(
        self,
        clusters: List[Cluster],
        signals_list: List[CoordinationSignals],
        has_body_ratios: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction.

        Args:
            clusters: Clusters to classify
            signals_list: Corresponding signals
            has_body_ratios: Body ratios

        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")

        if has_body_ratios is None:
            has_body_ratios = [0.5] * len(clusters)

        X = np.array([
            self._extract_features(c, s, h)
            for c, s, h in zip(clusters, signals_list, has_body_ratios)
        ])
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        return predictions, probabilities

    def _compute_feature_importance(self) -> None:
        """Compute and store feature importance."""
        if hasattr(self._base_model, "feature_importances_"):
            importances = self._base_model.feature_importances_
        elif hasattr(self._base_model, "coef_"):
            importances = np.abs(self._base_model.coef_[0])
        else:
            self._feature_importance = None
            return

        self._feature_importance = dict(zip(self.FEATURE_NAMES, importances))

    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dictionary of feature_name -> importance
        """
        if not self._is_fitted:
            return {}

        if self._feature_importance is None:
            self._compute_feature_importance()

        return self._feature_importance or {}

    def top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N most important features.

        Args:
            n: Number of features

        Returns:
            List of (feature_name, importance) tuples
        """
        importance = self.feature_importance()
        sorted_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "base_model": self._base_model,
                "scaler": self.scaler,
                "model_type": self.model_type,
                "calibrate": self.calibrate,
                "feature_importance": self._feature_importance,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "CoordinationClassifier":
        """
        Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded classifier
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        classifier = cls(
            model_type=data["model_type"],
            calibrate=data["calibrate"]
        )
        classifier.model = data["model"]
        classifier._base_model = data["base_model"]
        classifier.scaler = data["scaler"]
        classifier._feature_importance = data.get("feature_importance")
        classifier._is_fitted = True

        return classifier


def get_classifier() -> CoordinationClassifier:
    """Get singleton classifier instance."""
    if not hasattr(get_classifier, "_instance"):
        get_classifier._instance = CoordinationClassifier()
    return get_classifier._instance
