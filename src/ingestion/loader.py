"""
Data loader for the Emergent Narrative Detection System.

Handles loading news articles from CSV files.
"""

import pandas as pd
from pathlib import Path
from typing import Iterator, Optional

from src.models import Article, ArticleCreate
from src.config import settings


def load_articles_from_csv(
    filepath: Path | str,
    limit: Optional[int] = None,
) -> list[Article]:
    """
    Load articles from a CSV file.

    Args:
        filepath: Path to CSV file
        limit: Optional limit on number of articles to load

    Returns:
        List of Article objects
    """
    df = pd.read_csv(filepath)

    if limit:
        df = df.head(limit)

    articles = []
    for _, row in df.iterrows():
        try:
            article_create = ArticleCreate(**row.to_dict())
            article = article_create.to_article()
            articles.append(article)
        except Exception as e:
            print(f"Error loading article {row.get('id', 'unknown')}: {e}")
            continue

    return articles


def load_articles_iterator(
    filepath: Path | str,
    chunk_size: int = 100,
) -> Iterator[list[Article]]:
    """
    Load articles from CSV in chunks (for large files).

    Args:
        filepath: Path to CSV file
        chunk_size: Number of articles per chunk

    Yields:
        Lists of Article objects
    """
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        articles = []
        for _, row in chunk.iterrows():
            try:
                article_create = ArticleCreate(**row.to_dict())
                article = article_create.to_article()
                articles.append(article)
            except Exception as e:
                print(f"Error loading article {row.get('id', 'unknown')}: {e}")
                continue
        yield articles


def load_default_dataset() -> list[Article]:
    """Load the default synthetic news dataset."""
    default_path = settings.RAW_DATA_DIR / "synthetic_news.csv"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default dataset not found at {default_path}. "
            "Please place synthetic_news.csv in data/raw/"
        )
    return load_articles_from_csv(default_path)


def get_dataset_stats(articles: list[Article]) -> dict:
    """
    Get summary statistics for a loaded dataset.

    Args:
        articles: List of Article objects

    Returns:
        Dictionary of statistics
    """
    if not articles:
        return {
            "count": 0,
            "unique_outlets": 0,
            "outlets": [],
            "outlet_types": [],
            "biases": [],
            "sections": [],
            "topics": [],
            "pattern_types": [],
            "date_range": {"min": "", "max": ""},
            "wire_origin_count": 0,
            "pr_origin_count": 0,
            "with_coordinated_frame": 0,
            "with_convergent_frame": 0,
        }

    outlets = set(a.outlet for a in articles)
    outlet_types = set(a.outlet_type.value for a in articles)
    biases = set(a.outlet_bias.value for a in articles)
    sections = set(a.section for a in articles if a.section)
    topics = set(a.pattern_topic for a in articles if a.pattern_topic)
    patterns = set(a.pattern_type.value for a in articles if a.pattern_type)

    timestamps = [a.published_at for a in articles]

    return {
        "count": len(articles),
        "unique_outlets": len(outlets),
        "outlets": sorted(outlets),
        "outlet_types": sorted(outlet_types),
        "biases": sorted(biases),
        "sections": sorted(sections),
        "topics": sorted(topics),
        "pattern_types": sorted(patterns),
        "date_range": {
            "min": min(timestamps).isoformat(),
            "max": max(timestamps).isoformat(),
        },
        "wire_origin_count": sum(1 for a in articles if a.is_wire_origin),
        "pr_origin_count": sum(1 for a in articles if a.is_pr_origin),
        "with_coordinated_frame": sum(1 for a in articles if a.has_coordinated_frame),
        "with_convergent_frame": sum(1 for a in articles if a.has_convergent_frame),
    }