"""
Processing module for the Emergent Narrative Detection System.
"""

from .embeddings import (
    EmbeddingGenerator,
    get_embedding_generator,
    compute_similarity,
    compute_centroid,
    compute_cluster_coherence,
)

from .clustering import (
    NarrativeClusterer,
    get_cluster_coherence,
    find_similar_clusters,
)

from .narrative_detector import (
    NarrativeDetector,
    detect_stage_transitions,
)

__all__ = [
    # Embeddings
    "EmbeddingGenerator",
    "get_embedding_generator",
    "compute_similarity",
    "compute_centroid",
    "compute_cluster_coherence",
    # Clustering
    "NarrativeClusterer",
    "get_cluster_coherence",
    "find_similar_clusters",
    # Detection
    "NarrativeDetector",
    "detect_stage_transitions",
]