"""
Embedding generation for the Emergent Narrative Detection System.

Uses sentence-transformers to generate embeddings for articles.
"""

from typing import Optional
import numpy as np

from src.models import Article
from src.config import settings


class EmbeddingGenerator:
    """
    Generates embeddings for news articles using sentence-transformers.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence-transformer model
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str], batch_size: Optional[int] = None) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding

        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def embed_article(self, article: Article) -> Article:
        """
        Generate embedding for an article and attach it.

        Args:
            article: Article to embed

        Returns:
            Article with embedding attached
        """
        embedding = self.embed_text(article.combined_text)
        article.embedding = embedding
        return article

    def embed_articles(
        self,
        articles: list[Article],
        batch_size: Optional[int] = None,
    ) -> list[Article]:
        """
        Generate embeddings for multiple articles.

        Args:
            articles: List of articles to embed
            batch_size: Batch size for encoding

        Returns:
            Articles with embeddings attached
        """
        texts = [a.combined_text for a in articles]
        embeddings = self.embed_texts(texts, batch_size)

        for article, embedding in zip(articles, embeddings):
            article.embedding = embedding

        return articles


def compute_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity (0 to 1)
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """
    Compute the centroid of a set of embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Centroid vector
    """
    if not embeddings:
        raise ValueError("Cannot compute centroid of empty list")

    arr = np.array(embeddings)
    centroid = np.mean(arr, axis=0)
    return centroid.tolist()


def compute_cluster_coherence(embeddings: list[list[float]]) -> float:
    """
    Compute the coherence (average pairwise similarity) of a cluster.

    Args:
        embeddings: List of embedding vectors in the cluster

    Returns:
        Average pairwise cosine similarity
    """
    if len(embeddings) < 2:
        return 1.0  # Single item is perfectly coherent with itself

    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = compute_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)

    return float(np.mean(similarities))


# Global instance for convenience
_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the global embedding generator."""
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator