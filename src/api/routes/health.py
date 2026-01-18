"""
Health check endpoints.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

from src.config import settings


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    components: dict


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns system status and component health.
    """
    # Check components
    components = {
        "api": "healthy",
        "config": "healthy",
    }

    # Check ChromaDB (if available)
    try:
        from src.processing.vector_store import VectorStore
        store = VectorStore()
        count = store.count()
        components["chromadb"] = f"healthy ({count} documents)"
    except Exception as e:
        components["chromadb"] = f"error: {str(e)}"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        components=components
    )


@router.get("/health/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}
