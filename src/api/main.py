"""
FastAPI application for the Emergent Narrative Detection System.

Provides REST API endpoints for narrative analysis, signals, and clusters.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime
import json

from src.config import settings
from src.models import (
    Article,
    NarrativeCluster,
    NarrativeSignal,
    NarrativeStage,
    PatternType,
)
from src.ingestion import load_articles_from_csv, get_dataset_stats
from src.processing import (
    EmbeddingGenerator,
    NarrativeClusterer,
    NarrativeDetector,
)

# Initialize FastAPI app
app = FastAPI(
    title="Emergent Narrative Detection API",
    description="API for detecting and tracking emergent narratives across news media",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use proper database)
_articles: list[Article] = []
_clusters: list[NarrativeCluster] = []
_signals: list[NarrativeSignal] = []
_embedding_generator: Optional[EmbeddingGenerator] = None
_clusterer: Optional[NarrativeClusterer] = None
_detector: Optional[NarrativeDetector] = None


def get_embedding_generator() -> EmbeddingGenerator:
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def get_clusterer() -> NarrativeClusterer:
    global _clusterer
    if _clusterer is None:
        _clusterer = NarrativeClusterer()
    return _clusterer


def get_detector() -> NarrativeDetector:
    global _detector
    if _detector is None:
        _detector = NarrativeDetector()
    return _detector


# ============== Health & Info Endpoints ==============

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Emergent Narrative Detection API",
        "version": "1.0.0",
        "status": "running",
        "articles_loaded": len(_articles),
        "clusters_count": len(_clusters),
        "signals_count": len(_signals),
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============== Data Loading Endpoints ==============

@app.post("/data/load")
async def load_data(filepath: str = Query(default=None, description="Path to CSV file")):
    """
    Load articles from a CSV file.

    If no filepath provided, loads the default synthetic_news.csv.
    """
    global _articles, _clusters, _signals

    try:
        if filepath:
            _articles = load_articles_from_csv(filepath)
        else:
            default_path = settings.RAW_DATA_DIR / "synthetic_news.csv"
            _articles = load_articles_from_csv(default_path)

        # Reset clusters and signals
        _clusters = []
        _signals = []

        stats = get_dataset_stats(_articles)
        return {
            "status": "success",
            "message": f"Loaded {len(_articles)} articles",
            "stats": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/stats")
async def get_stats():
    """Get statistics about the loaded dataset."""
    if not _articles:
        raise HTTPException(status_code=404, detail="No data loaded. Call /data/load first.")

    return get_dataset_stats(_articles)


# ============== Processing Endpoints ==============

@app.post("/process/embed")
async def embed_articles():
    """Generate embeddings for all loaded articles."""
    global _articles

    if not _articles:
        raise HTTPException(status_code=404, detail="No data loaded. Call /data/load first.")

    try:
        generator = get_embedding_generator()
        _articles = generator.embed_articles(_articles)

        return {
            "status": "success",
            "message": f"Generated embeddings for {len(_articles)} articles",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/cluster")
async def cluster_articles():
    """Cluster articles into narrative groups."""
    global _articles, _clusters

    if not _articles:
        raise HTTPException(status_code=404, detail="No data loaded. Call /data/load first.")

    # Check if embeddings exist
    if not _articles[0].embedding:
        raise HTTPException(
            status_code=400,
            detail="Articles not embedded. Call /process/embed first."
        )

    try:
        clusterer = get_clusterer()
        _clusters, noise = clusterer.cluster_articles(_articles)

        return {
            "status": "success",
            "clusters_found": len(_clusters),
            "noise_articles": len(noise),
            "clusters": [c.to_summary_dict() for c in _clusters],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/detect")
async def detect_signals():
    """Run narrative detection on clusters to generate signals."""
    global _clusters, _signals

    if not _clusters:
        raise HTTPException(
            status_code=404,
            detail="No clusters found. Call /process/cluster first."
        )

    try:
        detector = get_detector()
        _signals = detector.analyze_clusters(_clusters)

        return {
            "status": "success",
            "signals_detected": len(_signals),
            "signals": [s.to_dict() for s in _signals],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/full")
async def run_full_pipeline():
    """Run the complete pipeline: embed → cluster → detect."""
    global _articles, _clusters, _signals

    if not _articles:
        raise HTTPException(status_code=404, detail="No data loaded. Call /data/load first.")

    try:
        # Embed
        generator = get_embedding_generator()
        _articles = generator.embed_articles(_articles)

        # Cluster
        clusterer = get_clusterer()
        _clusters, noise = clusterer.cluster_articles(_articles)

        # Detect
        detector = get_detector()
        _signals = detector.analyze_clusters(_clusters)

        return {
            "status": "success",
            "articles_processed": len(_articles),
            "clusters_found": len(_clusters),
            "noise_articles": len(noise),
            "signals_detected": len(_signals),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== Cluster Endpoints ==============

@app.get("/clusters")
async def list_clusters(
    stage: Optional[str] = Query(default=None, description="Filter by lifecycle stage"),
    topic: Optional[str] = Query(default=None, description="Filter by topic"),
    min_size: int = Query(default=1, description="Minimum cluster size"),
):
    """List all narrative clusters with optional filtering."""
    if not _clusters:
        return {"clusters": [], "count": 0}

    filtered = _clusters

    if stage:
        try:
            stage_enum = NarrativeStage(stage)
            filtered = [c for c in filtered if c.stage == stage_enum]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {stage}")

    if topic:
        filtered = [c for c in filtered if c.topic and topic.lower() in c.topic.lower()]

    filtered = [c for c in filtered if c.size >= min_size]

    return {
        "clusters": [c.to_summary_dict() for c in filtered],
        "count": len(filtered),
    }


@app.get("/clusters/{cluster_id}")
async def get_cluster(cluster_id: str):
    """Get detailed information about a specific cluster."""
    cluster = next((c for c in _clusters if c.id == cluster_id), None)

    if not cluster:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

    return {
        **cluster.to_summary_dict(),
        "timeline": cluster.get_timeline(),
        "articles": [
            {
                "id": a.id,
                "outlet": a.outlet,
                "outlet_type": a.outlet_type.value,
                "outlet_bias": a.outlet_bias.value,
                "title": a.title,
                "published_at": a.published_at.isoformat(),
                "wire_source": a.wire_source.value,
            }
            for a in cluster.articles
        ],
    }


# ============== Signal Endpoints ==============

@app.get("/signals")
async def list_signals(
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    signal_type: Optional[str] = Query(default=None, description="Filter by signal type"),
    reviewed: Optional[bool] = Query(default=None, description="Filter by review status"),
):
    """List all detected signals with optional filtering."""
    if not _signals:
        return {"signals": [], "count": 0}

    filtered = _signals

    if severity:
        filtered = [s for s in filtered if s.severity.value == severity]

    if signal_type:
        filtered = [s for s in filtered if s.signal_type.value == signal_type]

    if reviewed is not None:
        filtered = [s for s in filtered if s.reviewed == reviewed]

    return {
        "signals": [s.to_dict() for s in filtered],
        "count": len(filtered),
    }


@app.get("/signals/{signal_id}")
async def get_signal(signal_id: str):
    """Get detailed information about a specific signal."""
    signal = next((s for s in _signals if s.id == signal_id), None)

    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")

    # Include associated cluster info
    cluster = next((c for c in _clusters if c.id == signal.cluster_id), None)

    return {
        **signal.to_dict(),
        "cluster": cluster.to_summary_dict() if cluster else None,
    }


@app.patch("/signals/{signal_id}/review")
async def review_signal(
    signal_id: str,
    reviewed: bool = Query(..., description="Mark as reviewed"),
    false_positive: Optional[bool] = Query(default=None, description="Mark as false positive"),
    notes: Optional[str] = Query(default=None, description="Analyst notes"),
):
    """Update the review status of a signal."""
    signal = next((s for s in _signals if s.id == signal_id), None)

    if not signal:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")

    signal.reviewed = reviewed
    if false_positive is not None:
        signal.false_positive = false_positive
    if notes:
        signal.analyst_notes = notes

    return signal.to_dict()


# ============== Article Endpoints ==============

@app.get("/articles")
async def list_articles(
    outlet: Optional[str] = Query(default=None, description="Filter by outlet"),
    outlet_type: Optional[str] = Query(default=None, description="Filter by outlet type"),
    topic: Optional[str] = Query(default=None, description="Filter by topic"),
    limit: int = Query(default=100, description="Maximum articles to return"),
):
    """List articles with optional filtering."""
    if not _articles:
        return {"articles": [], "count": 0}

    filtered = _articles

    if outlet:
        filtered = [a for a in filtered if outlet.lower() in a.outlet.lower()]

    if outlet_type:
        filtered = [a for a in filtered if a.outlet_type.value == outlet_type]

    if topic:
        filtered = [a for a in filtered if a.pattern_topic and topic.lower() in a.pattern_topic.lower()]

    filtered = filtered[:limit]

    return {
        "articles": [
            {
                "id": a.id,
                "outlet": a.outlet,
                "outlet_type": a.outlet_type.value,
                "outlet_bias": a.outlet_bias.value,
                "title": a.title,
                "published_at": a.published_at.isoformat(),
                "section": a.section,
                "pattern_type": a.pattern_type.value if a.pattern_type else None,
                "pattern_topic": a.pattern_topic,
            }
            for a in filtered
        ],
        "count": len(filtered),
    }


# ============== Run server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)