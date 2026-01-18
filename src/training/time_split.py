"""
Time-based train/test splitting for coordination detection.
Critical for realistic evaluation - prevents future data leakage.
"""

from typing import List, Tuple, Iterator, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from src.models.article import Document
from src.models.cluster import Cluster


@dataclass
class TimeSplit:
    """A single time-based train/test split."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    gap_days: int = 0

    @property
    def train_duration_days(self) -> float:
        return (self.train_end - self.train_start).days

    @property
    def test_duration_days(self) -> float:
        return (self.test_end - self.test_start).days


class TimeBasedSplitter:
    """
    Create train/val/test splits based on time.

    Critical for coordination detection evaluation:
    - Prevents temporal leakage
    - Simulates real deployment scenario
    - Supports walk-forward validation
    """

    def __init__(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        gap_days: int = 1,
    ):
        """
        Initialize splitter.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            gap_days: Gap between sets to prevent leakage
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gap_days = gap_days

    def split_documents(
        self,
        documents: List[Document]
    ) -> Tuple[List[Document], List[Document], List[Document]]:
        """
        Split documents into train/val/test by time.

        Args:
            documents: All documents

        Returns:
            Tuple of (train, val, test) document lists
        """
        if not documents:
            return [], [], []

        # Sort by timestamp
        sorted_docs = sorted(documents, key=lambda d: d.timestamp)

        # Get date range
        min_date = sorted_docs[0].timestamp
        max_date = sorted_docs[-1].timestamp
        total_days = (max_date - min_date).days

        if total_days == 0:
            # All same day - split by count
            n = len(sorted_docs)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))
            return sorted_docs[:train_end], sorted_docs[train_end:val_end], sorted_docs[val_end:]

        # Compute cutoff dates
        train_end = min_date + timedelta(days=total_days * self.train_ratio)
        val_start = train_end + timedelta(days=self.gap_days)
        val_end = val_start + timedelta(days=total_days * self.val_ratio)
        test_start = val_end + timedelta(days=self.gap_days)

        # Split
        train = [d for d in sorted_docs if d.timestamp < train_end]
        val = [d for d in sorted_docs if val_start <= d.timestamp < val_end]
        test = [d for d in sorted_docs if d.timestamp >= test_start]

        return train, val, test

    def split_clusters(
        self,
        clusters: List[Cluster]
    ) -> Tuple[List[Cluster], List[Cluster], List[Cluster]]:
        """
        Split clusters into train/val/test by window time.

        Args:
            clusters: All clusters

        Returns:
            Tuple of (train, val, test) cluster lists
        """
        if not clusters:
            return [], [], []

        # Sort by window start
        sorted_clusters = sorted(clusters, key=lambda c: c.window_start)

        # Get date range
        min_date = sorted_clusters[0].window_start
        max_date = sorted_clusters[-1].window_end
        total_days = (max_date - min_date).days

        if total_days == 0:
            n = len(sorted_clusters)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))
            return sorted_clusters[:train_end], sorted_clusters[train_end:val_end], sorted_clusters[val_end:]

        # Compute cutoffs
        train_end = min_date + timedelta(days=total_days * self.train_ratio)
        val_start = train_end + timedelta(days=self.gap_days)
        val_end = val_start + timedelta(days=total_days * self.val_ratio)
        test_start = val_end + timedelta(days=self.gap_days)

        train = [c for c in sorted_clusters if c.window_end < train_end]
        val = [c for c in sorted_clusters if val_start <= c.window_start < val_end]
        test = [c for c in sorted_clusters if c.window_start >= test_start]

        return train, val, test

    def get_split_info(
        self,
        documents: List[Document]
    ) -> TimeSplit:
        """
        Get split date information without splitting data.

        Args:
            documents: Documents to analyze

        Returns:
            TimeSplit with date boundaries
        """
        sorted_docs = sorted(documents, key=lambda d: d.timestamp)
        min_date = sorted_docs[0].timestamp
        max_date = sorted_docs[-1].timestamp
        total_days = (max_date - min_date).days

        train_end = min_date + timedelta(days=total_days * self.train_ratio)
        val_start = train_end + timedelta(days=self.gap_days)
        val_end = val_start + timedelta(days=total_days * self.val_ratio)
        test_start = val_end + timedelta(days=self.gap_days)

        return TimeSplit(
            train_start=min_date,
            train_end=train_end,
            test_start=test_start,
            test_end=max_date,
            gap_days=self.gap_days
        )


class WalkForwardValidator:
    """
    Walk-forward validation for time series.

    Expands training window over time while testing on future data.
    More realistic evaluation than single split.
    """

    def __init__(
        self,
        initial_train_days: int = 30,
        test_window_days: int = 7,
        step_days: int = 7,
        gap_days: int = 1,
    ):
        """
        Initialize walk-forward validator.

        Args:
            initial_train_days: Initial training window size
            test_window_days: Size of each test window
            step_days: Days to advance between folds
            gap_days: Gap between train and test
        """
        self.initial_train_days = initial_train_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.gap_days = gap_days

    def split(
        self,
        documents: List[Document]
    ) -> Iterator[Tuple[List[Document], List[Document]]]:
        """
        Generate walk-forward train/test splits.

        Training window expands with each fold.
        Test window moves forward.

        Args:
            documents: All documents

        Yields:
            (train_docs, test_docs) tuples
        """
        sorted_docs = sorted(documents, key=lambda d: d.timestamp)
        min_date = sorted_docs[0].timestamp
        max_date = sorted_docs[-1].timestamp

        # Start after initial training period
        current_train_end = min_date + timedelta(days=self.initial_train_days)

        while current_train_end + timedelta(days=self.gap_days + self.test_window_days) <= max_date:
            test_start = current_train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=self.test_window_days)

            train = [d for d in sorted_docs if d.timestamp < current_train_end]
            test = [d for d in sorted_docs if test_start <= d.timestamp < test_end]

            if train and test:
                yield train, test

            # Advance
            current_train_end += timedelta(days=self.step_days)

    def split_clusters(
        self,
        clusters: List[Cluster]
    ) -> Iterator[Tuple[List[Cluster], List[Cluster]]]:
        """
        Generate walk-forward splits for clusters.

        Args:
            clusters: All clusters

        Yields:
            (train_clusters, test_clusters) tuples
        """
        sorted_clusters = sorted(clusters, key=lambda c: c.window_start)
        min_date = sorted_clusters[0].window_start
        max_date = sorted_clusters[-1].window_end

        current_train_end = min_date + timedelta(days=self.initial_train_days)

        while current_train_end + timedelta(days=self.gap_days + self.test_window_days) <= max_date:
            test_start = current_train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=self.test_window_days)

            train = [c for c in sorted_clusters if c.window_end < current_train_end]
            test = [c for c in sorted_clusters if test_start <= c.window_start < test_end]

            if train and test:
                yield train, test

            current_train_end += timedelta(days=self.step_days)

    def n_splits(self, documents: List[Document]) -> int:
        """
        Estimate number of splits.

        Args:
            documents: Documents to analyze

        Returns:
            Approximate number of folds
        """
        if not documents:
            return 0

        sorted_docs = sorted(documents, key=lambda d: d.timestamp)
        min_date = sorted_docs[0].timestamp
        max_date = sorted_docs[-1].timestamp
        total_days = (max_date - min_date).days

        available_days = total_days - self.initial_train_days - self.gap_days - self.test_window_days
        if available_days <= 0:
            return 0

        return max(1, available_days // self.step_days)


class ExpandingWindowSplitter:
    """
    Expanding window splitter for cumulative training.

    Each fold uses all data up to a point for training.
    """

    def __init__(
        self,
        min_train_days: int = 14,
        test_window_days: int = 7,
        n_splits: int = 5,
    ):
        """
        Initialize expanding window splitter.

        Args:
            min_train_days: Minimum training window
            test_window_days: Test window size
            n_splits: Number of folds
        """
        self.min_train_days = min_train_days
        self.test_window_days = test_window_days
        self.n_splits = n_splits

    def split(
        self,
        documents: List[Document]
    ) -> Iterator[Tuple[List[Document], List[Document], TimeSplit]]:
        """
        Generate expanding window splits.

        Args:
            documents: All documents

        Yields:
            (train, test, split_info) tuples
        """
        sorted_docs = sorted(documents, key=lambda d: d.timestamp)
        min_date = sorted_docs[0].timestamp
        max_date = sorted_docs[-1].timestamp
        total_days = (max_date - min_date).days

        # Calculate fold boundaries
        available_for_test = total_days - self.min_train_days
        if available_for_test < self.test_window_days:
            return

        step = available_for_test // self.n_splits

        for i in range(self.n_splits):
            train_end = min_date + timedelta(days=self.min_train_days + i * step)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_window_days)

            if test_end > max_date:
                test_end = max_date

            train = [d for d in sorted_docs if d.timestamp < train_end]
            test = [d for d in sorted_docs if test_start <= d.timestamp < test_end]

            if train and test:
                split_info = TimeSplit(
                    train_start=min_date,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    gap_days=1
                )
                yield train, test, split_info
