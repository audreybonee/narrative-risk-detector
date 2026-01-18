"""
Clustering module for the Emergent Narrative Detection System.

Uses HDBSCAN to cluster articles into narrative groups.
"""

from typing import Optional
import numpy as np
import uuid

from src.models import Article, NarrativeCluster
from src.config import settings
from .embeddings import compute_centroid, compute_cluster_coherence


class NarrativeClusterer:
    """
    Clusters articles into narrative groups using HDBSCAN.
    """

    def __init__(
        self,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        cluster_selection_epsilon: Optional[float] = None,
        metric: Optional[str] = None,
    ):
        """
        Initialize the clusterer.

        Args:
            min_cluster_size: Minimum articles to form a cluster
            min_samples: Minimum samples for core points
            cluster_selection_epsilon: Distance threshold for cluster merging
            metric: Distance metric to use
        """
        self.min_cluster_size = min_cluster_size or settings.HDBSCAN_MIN_CLUSTER_SIZE
        self.min_samples = min_samples or settings.HDBSCAN_MIN_SAMPLES
        self.cluster_selection_epsilon = cluster_selection_epsilon or settings.HDBSCAN_CLUSTER_SELECTION_EPSILON
        self.metric = metric or settings.HDBSCAN_METRIC

        self._clusterer = None

    def _get_clusterer(self):
        """Create HDBSCAN clusterer instance."""
        import hdbscan
        return hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            core_dist_n_jobs=-1,
        )

    def cluster_articles(
        self,
        articles: list[Article],
    ) -> tuple[list[NarrativeCluster], list[Article]]:
        """
        Cluster articles into narrative groups.

        Args:
            articles: List of articles with embeddings

        Returns:
            Tuple of (clusters, noise_articles)
        """
        if not articles:
            return [], []

        # Verify all articles have embeddings
        for article in articles:
            if article.embedding is None:
                raise ValueError(f"Article {article.id} missing embedding")

        # Extract embeddings
        embeddings = np.array([a.embedding for a in articles])

        # Perform clustering
        clusterer = self._get_clusterer()
        labels = clusterer.fit_predict(embeddings)

        # Group articles by cluster
        cluster_articles: dict[int, list[Article]] = {}
        noise_articles: list[Article] = []

        for article, label in zip(articles, labels):
            if label == -1:
                noise_articles.append(article)
            else:
                if label not in cluster_articles:
                    cluster_articles[label] = []
                cluster_articles[label].append(article)

        # Create NarrativeCluster objects
        clusters = []
        for label, articles_in_cluster in cluster_articles.items():
            cluster = self._create_cluster(articles_in_cluster)
            clusters.append(cluster)

        # Sort by size descending
        clusters.sort(key=lambda c: c.size, reverse=True)

        return clusters, noise_articles

    def _create_cluster(self, articles: list[Article]) -> NarrativeCluster:
        """
        Create a NarrativeCluster from a group of articles.

        Args:
            articles: Articles in the cluster

        Returns:
            NarrativeCluster object
        """
        # Sort by time
        sorted_articles = sorted(articles, key=lambda a: a.published_at)

        # Determine topic from most common pattern_topic
        topics = [a.pattern_topic for a in articles if a.pattern_topic]
        topic = None
        if topics:
            from collections import Counter
            topic = Counter(topics).most_common(1)[0][0]

        # Select representative headline (from earliest wire or first article)
        wire_articles = [a for a in sorted_articles if a.is_wire_origin]
        if wire_articles:
            representative = wire_articles[0].title
        else:
            representative = sorted_articles[0].title

        # Compute centroid
        embeddings = [a.embedding for a in articles if a.embedding]
        centroid = compute_centroid(embeddings) if embeddings else None

        cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"

        return NarrativeCluster(
            id=cluster_id,
            topic=topic,
            representative_headline=representative,
            articles=sorted_articles,
            first_seen=sorted_articles[0].published_at,
            last_seen=sorted_articles[-1].published_at,
            centroid_embedding=centroid,
        )

    def update_clusters(
        self,
        existing_clusters: list[NarrativeCluster],
        new_articles: list[Article],
        similarity_threshold: float = 0.85,
    ) -> tuple[list[NarrativeCluster], list[NarrativeCluster], list[Article]]:
        """
        Update existing clusters with new articles.

        Args:
            existing_clusters: Current clusters
            new_articles: New articles to incorporate
            similarity_threshold: Threshold for assigning to existing cluster

        Returns:
            Tuple of (updated_clusters, new_clusters, unassigned_articles)
        """
        from .embeddings import compute_similarity

        if not new_articles:
            return existing_clusters, [], []

        # Ensure new articles have embeddings
        for article in new_articles:
            if article.embedding is None:
                raise ValueError(f"Article {article.id} missing embedding")

        unassigned = []
        updated_cluster_ids = set()

        # Try to assign each new article to existing clusters
        for article in new_articles:
            best_cluster = None
            best_similarity = 0.0

            for cluster in existing_clusters:
                if cluster.centroid_embedding is None:
                    continue

                sim = compute_similarity(article.embedding, cluster.centroid_embedding)
                if sim > best_similarity and sim >= similarity_threshold:
                    best_similarity = sim
                    best_cluster = cluster

            if best_cluster:
                best_cluster.articles.append(article)
                best_cluster.articles.sort(key=lambda a: a.published_at)
                best_cluster.last_seen = max(best_cluster.last_seen, article.published_at)
                updated_cluster_ids.add(best_cluster.id)

                # Recompute centroid
                embeddings = [a.embedding for a in best_cluster.articles if a.embedding]
                best_cluster.centroid_embedding = compute_centroid(embeddings)
            else:
                unassigned.append(article)

        # Cluster unassigned articles
        new_clusters = []
        if unassigned:
            new_clusters, remaining_noise = self.cluster_articles(unassigned)
            unassigned = remaining_noise

        # Separate updated and unchanged clusters
        updated_clusters = [c for c in existing_clusters if c.id in updated_cluster_ids]

        return existing_clusters, new_clusters, unassigned


def get_cluster_coherence(cluster: NarrativeCluster) -> float:
    """
    Compute the semantic coherence of a cluster.

    Args:
        cluster: NarrativeCluster to analyze

    Returns:
        Coherence score (0 to 1)
    """
    embeddings = [a.embedding for a in cluster.articles if a.embedding]
    if len(embeddings) < 2:
        return 1.0
    return compute_cluster_coherence(embeddings)


def find_similar_clusters(
    clusters: list[NarrativeCluster],
    threshold: float = 0.8,
) -> list[tuple[NarrativeCluster, NarrativeCluster, float]]:
    """
    Find pairs of clusters that might be related.

    Args:
        clusters: List of clusters to compare
        threshold: Minimum similarity to report

    Returns:
        List of (cluster1, cluster2, similarity) tuples
    """
    from .embeddings import compute_similarity

    similar_pairs = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            c1, c2 = clusters[i], clusters[j]

            if c1.centroid_embedding is None or c2.centroid_embedding is None:
                continue

            sim = compute_similarity(c1.centroid_embedding, c2.centroid_embedding)
            if sim >= threshold:
                similar_pairs.append((c1, c2, sim))

    # Sort by similarity descending
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    return similar_pairs