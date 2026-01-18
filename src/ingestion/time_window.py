"""
Time window iterator for the Emergent Narrative Detection System.

Enables streaming simulation by iterating through articles in time windows.
"""

from datetime import datetime, timedelta
from typing import Iterator, Optional
import re

from src.models import Article
from src.config import settings


def parse_time_window(window_str: str) -> timedelta:
    """
    Parse a time window string into a timedelta.

    Supported formats: 1h, 4h, 12h, 1d, 1w

    Args:
        window_str: Time window string (e.g., "4h", "1d")

    Returns:
        timedelta object
    """
    pattern = r"^(\d+)([hdw])$"
    match = re.match(pattern, window_str.lower())

    if not match:
        raise ValueError(
            f"Invalid time window format: {window_str}. "
            f"Supported: {settings.SUPPORTED_TIME_WINDOWS}"
        )

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "h":
        return timedelta(hours=value)
    elif unit == "d":
        return timedelta(days=value)
    elif unit == "w":
        return timedelta(weeks=value)
    else:
        raise ValueError(f"Unknown time unit: {unit}")


class TimeWindowIterator:
    """
    Iterator that yields articles in time-based windows.

    Useful for simulating real-time streaming and testing
    time-sensitive narrative detection.
    """

    def __init__(
        self,
        articles: list[Article],
        window_size: str = "4h",
        step_size: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ):
        """
        Initialize the time window iterator.

        Args:
            articles: List of articles to iterate through
            window_size: Size of each window (e.g., "4h", "1d")
            step_size: Step between windows (defaults to window_size)
            start_time: Optional start time (defaults to earliest article)
            end_time: Optional end time (defaults to latest article)
        """
        self.articles = sorted(articles, key=lambda a: a.published_at)
        self.window_delta = parse_time_window(window_size)
        self.step_delta = parse_time_window(step_size) if step_size else self.window_delta

        if not self.articles:
            raise ValueError("Cannot iterate over empty article list")

        self.start_time = start_time or self.articles[0].published_at
        self.end_time = end_time or self.articles[-1].published_at

        self.current_time = self.start_time

    def __iter__(self) -> Iterator[tuple[datetime, datetime, list[Article]]]:
        """
        Iterate through time windows.

        Yields:
            Tuple of (window_start, window_end, articles_in_window)
        """
        while self.current_time <= self.end_time:
            window_start = self.current_time
            window_end = self.current_time + self.window_delta

            # Get articles in this window
            window_articles = [
                a for a in self.articles
                if window_start <= a.published_at < window_end
            ]

            yield window_start, window_end, window_articles

            self.current_time += self.step_delta

    def get_cumulative_windows(self) -> Iterator[tuple[datetime, datetime, list[Article]]]:
        """
        Iterate with cumulative windows (all articles up to window end).

        Useful for tracking narrative evolution over time.

        Yields:
            Tuple of (window_start, window_end, all_articles_up_to_window_end)
        """
        current = self.start_time

        while current <= self.end_time:
            window_end = current + self.window_delta

            # Get all articles up to this point
            cumulative_articles = [
                a for a in self.articles
                if a.published_at < window_end
            ]

            yield self.start_time, window_end, cumulative_articles

            current += self.step_delta


class SlidingWindowIterator:
    """
    Sliding window that maintains state for incremental processing.

    Tracks which articles are new in each window for efficient updates.
    """

    def __init__(
        self,
        articles: list[Article],
        window_size: str = "4h",
        slide_size: str = "1h",
    ):
        """
        Initialize sliding window.

        Args:
            articles: List of articles
            window_size: Total window size
            slide_size: How much to slide each iteration
        """
        self.articles = sorted(articles, key=lambda a: a.published_at)
        self.window_delta = parse_time_window(window_size)
        self.slide_delta = parse_time_window(slide_size)

        if not self.articles:
            raise ValueError("Cannot iterate over empty article list")

        self.start_time = self.articles[0].published_at
        self.end_time = self.articles[-1].published_at
        self.current_start = self.start_time

        self._seen_ids: set[str] = set()

    def __iter__(self) -> Iterator[dict]:
        """
        Iterate through sliding windows.

        Yields:
            Dict with window info and new/retained/expired articles
        """
        while self.current_start <= self.end_time:
            window_end = self.current_start + self.window_delta

            # Get articles in current window
            current_window = [
                a for a in self.articles
                if self.current_start <= a.published_at < window_end
            ]

            current_ids = set(a.id for a in current_window)

            # Determine new, retained, expired
            new_ids = current_ids - self._seen_ids
            retained_ids = current_ids & self._seen_ids
            expired_ids = self._seen_ids - current_ids

            new_articles = [a for a in current_window if a.id in new_ids]
            retained_articles = [a for a in current_window if a.id in retained_ids]

            yield {
                "window_start": self.current_start,
                "window_end": window_end,
                "all_articles": current_window,
                "new_articles": new_articles,
                "retained_articles": retained_articles,
                "expired_ids": expired_ids,
                "total_count": len(current_window),
                "new_count": len(new_articles),
            }

            # Update state
            self._seen_ids = current_ids
            self.current_start += self.slide_delta


def group_articles_by_topic(articles: list[Article]) -> dict[str, list[Article]]:
    """
    Group articles by their pattern_topic field.

    Args:
        articles: List of articles

    Returns:
        Dictionary mapping topic to list of articles
    """
    grouped: dict[str, list[Article]] = {}

    for article in articles:
        topic = article.pattern_topic or "unknown"
        if topic not in grouped:
            grouped[topic] = []
        grouped[topic].append(article)

    # Sort each group by time
    for topic in grouped:
        grouped[topic].sort(key=lambda a: a.published_at)

    return grouped


def get_time_series_counts(
    articles: list[Article],
    bucket_size: str = "1h",
) -> list[dict]:
    """
    Get article counts over time buckets.

    Useful for visualization and velocity analysis.

    Args:
        articles: List of articles
        bucket_size: Size of time buckets

    Returns:
        List of dicts with timestamp and count
    """
    if not articles:
        return []

    sorted_articles = sorted(articles, key=lambda a: a.published_at)
    bucket_delta = parse_time_window(bucket_size)

    start = sorted_articles[0].published_at
    end = sorted_articles[-1].published_at

    buckets = []
    current = start

    while current <= end:
        bucket_end = current + bucket_delta
        count = sum(
            1 for a in sorted_articles
            if current <= a.published_at < bucket_end
        )
        buckets.append({
            "timestamp": current.isoformat(),
            "count": count,
        })
        current = bucket_end

    return buckets