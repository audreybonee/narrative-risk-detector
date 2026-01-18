"""
Cluster exploration endpoints.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from src.models.cluster import Cluster, ClusterSummary


router = APIRouter()


# In-memory storage for demo (soon it will be replaced with database)
_clusters_store: List[Cluster] = []


class ClusterDocument(BaseModel):
    """Document info within a cluster."""
    document_id: str
    text_snippet: str
    similarity_to_centroid: float
    timestamp: datetime
    score: int


@router.get("/clusters", response_model=List[Cluster])
async def list_clusters(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    min_size: int = Query(5, ge=1),
    coordination_level: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
) -> List[Cluster]:
    """
    List clusters with optional filtering.

    - **start_date**: Filter clusters after this date
    - **end_date**: Filter clusters before this date
    - **min_size**: Minimum cluster size
    - **coordination_level**: Filter by level (organic, suspicious, likely_coordinated)
    - **limit**: Maximum results
    - **offset**: Pagination offset
    """
    filtered = _clusters_store.copy()

    # Apply filters
    if start_date:
        filtered = [c for c in filtered if c.window_start >= start_date]
    if end_date:
        filtered = [c for c in filtered if c.window_end <= end_date]
    if min_size:
        filtered = [c for c in filtered if c.size >= min_size]
    if coordination_level:
        filtered = [c for c in filtered if c.coordination_level == coordination_level]

    # Sort by coordination ratio descending
    filtered.sort(key=lambda c: c.coordination_ratio, reverse=True)

    return filtered[offset:offset + limit]


@router.get("/clusters/{cluster_id}", response_model=Cluster)
async def get_cluster(cluster_id: str) -> Cluster:
    """
    Get specific cluster by ID.

    Returns full cluster details including members.
    """
    for cluster in _clusters_store:
        if cluster.id == cluster_id:
            return cluster

    raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")


@router.get("/clusters/{cluster_id}/documents")
async def get_cluster_documents(
    cluster_id: str,
    limit: int = Query(20, le=100),
) -> List[ClusterDocument]:
    """
    Get documents in a cluster.

    Returns document snippets sorted by relevance.
    """
    for cluster in _clusters_store:
        if cluster.id == cluster_id:
            sorted_members = sorted(
                cluster.members,
                key=lambda m: m.similarity_to_centroid,
                reverse=True
            )[:limit]

            # Would fetch actual documents from store
            return [
                ClusterDocument(
                    document_id=m.document_id,
                    text_snippet=f"Document {m.document_id}",  # Placeholder update coming soon
                    similarity_to_centroid=m.similarity_to_centroid,
                    timestamp=cluster.window_start,  # Placeholder update coming soon
                    score=0  # Placeholder update coming soon
                )
                for m in sorted_members
            ]

    raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")


@router.get("/clusters/summary/window", response_model=ClusterSummary)
async def get_window_summary(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
) -> ClusterSummary:
    """
    Get cluster summary for a time window.

    Returns aggregate statistics for clusters in the window.
    """
    window_clusters = [
        c for c in _clusters_store
        if c.window_start >= start_date and c.window_end <= end_date
    ]

    if not window_clusters:
        return ClusterSummary(
            window_start=start_date,
            window_end=end_date,
            total_clusters=0,
            total_documents=0,
        )

    return ClusterSummary(
        window_start=start_date,
        window_end=end_date,
        total_clusters=len(window_clusters),
        total_documents=sum(c.size for c in window_clusters),
        organic_count=sum(1 for c in window_clusters if c.coordination_level == "organic"),
        suspicious_count=sum(1 for c in window_clusters if c.coordination_level == "suspicious"),
        likely_coordinated_count=sum(1 for c in window_clusters if c.coordination_level == "likely_coordinated"),
        avg_coordination_ratio=sum(c.coordination_ratio for c in window_clusters) / len(window_clusters),
        max_coordination_ratio=max(c.coordination_ratio for c in window_clusters),
        avg_semantic_coherence=sum(c.semantic_coherence for c in window_clusters) / len(window_clusters),
    )


@router.get("/clusters/top/coordinated")
async def get_top_coordinated(
    limit: int = Query(10, le=50),
    min_ratio: float = Query(2.0),
) -> List[Cluster]:
    """
    Get top clusters by coordination ratio.

    Returns clusters with highest coordination signals.
    """
    filtered = [c for c in _clusters_store if c.coordination_ratio >= min_ratio]
    filtered.sort(key=lambda c: c.coordination_ratio, reverse=True)
    return filtered[:limit]


# Helper function to add clusters (used by other modules)
def add_cluster(cluster: Cluster) -> None:
    """Add a cluster to the store."""
    _clusters_store.append(cluster)


def clear_clusters() -> None:
    """Clear all clusters (for testing)."""
    _clusters_store.clear()
