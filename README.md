# 📰 Emergent Narrative Detection System

An ML pipeline for detecting, tracking, and classifying how narratives emerge and propagate across the news media ecosystem. Designed for researchers in computational social science to differentiate between organic viral trends and coordinated influence operations.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange.svg)

## Overview

The Emergent Narrative Detection System shifts the unit of analysis from individual user posts to institutional behavior. By modeling news outlets as agents in a network, it identifies distinct patterns of information propagation:

- **Wire cascades**: How wire service stories propagate through the media ecosystem
- **PR amplification**: Corporate/institutional messaging gaining media traction
- **Frame convergence**: Multiple outlets organically converging on similar framing
- **Cross-spectrum spread**: Stories crossing political boundaries
- **Synchronized messaging**: Potential coordinated framing patterns

Rather than detecting "misinformation," this system focuses on understanding the *dynamics* of narrative emergence—how stories gain traction, which outlets pick them up, and how framing evolves over time.

## Features

-  **Semantic Clustering**: Groups articles by narrative similarity using sentence embeddings and HDBSCAN
-  **Narrative Lifecycle Tracking**: Monitors stories from nascent emergence through establishment
-  **Real-time Signals**: Detects noteworthy patterns like velocity spikes and cross-spectrum spread
-  **Interactive Dashboard**: Streamlit UI for exploring narratives and reviewing signals
-  **REST API**: FastAPI endpoints for integration with other tools
-  **Configurable Thresholds**: Tune detection sensitivity via environment variables

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/narrative-detection.git
cd narrative-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Then in your browser:
1. Click **"Load Data"** in the sidebar
2. Click **"Run Full Pipeline"** to process articles
3. Explore the detected narratives and signals

### Run via CLI

```bash
python scripts/batch_pipeline.py --input data/raw/synthetic_news.csv --verbose
```

Results are saved to `data/processed/`.

## Project Structure

```
narrative-detection/
├── README.md
├── CLAUDE.md              # Development guidance for AI assistants
├── requirements.txt
├── .env.example           # Configuration template
│
├── data/
│   ├── raw/               # Input datasets
│   ├── processed/         # Pipeline outputs (JSON)
│   └── chroma/            # Vector store persistence
│
├── src/
│   ├── models/            # Pydantic data models
│   │   ├── enums.py       # Taxonomy enums (OutletType, PatternType, etc.)
│   │   ├── article.py     # Article model
│   │   ├── cluster.py     # NarrativeCluster model
│   │   └── signal.py      # NarrativeSignal model
│   │
│   ├── ingestion/         # Data loading
│   │   ├── loader.py      # CSV ingestion
│   │   └── time_window.py # Temporal iteration utilities
│   │
│   ├── processing/        # ML pipeline
│   │   ├── embeddings.py  # Sentence-transformer embeddings
│   │   ├── clustering.py  # HDBSCAN narrative clustering
│   │   └── narrative_detector.py  # Signal detection
│   │
│   ├── api/               # REST API
│   │   └── main.py        # FastAPI application
│   │
│   ├── labeling/          # Weak supervision (planned)
│   ├── training/          # Classifier training (planned)
│   ├── monitoring/        # Drift detection (planned)
│   └── db/                # Database persistence (planned)
│
├── dashboard/
│   └── app.py             # Streamlit dashboard
│
├── scripts/
│   └── batch_pipeline.py  # CLI batch processing
│
└── tests/
    └── test_core.py       # Unit tests
```

## Data Format

The system expects a CSV with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique article identifier |
| `outlet` | string | Publication name (e.g., "Reuters", "CNN") |
| `outlet_type` | enum | `wire_service`, `major_national`, `local_regional`, `independent` |
| `outlet_bias` | enum | `left`, `center-left`, `center`, `center-right`, `right` |
| `title` | string | Article headline |
| `body` | string | Article content |
| `author` | string | Byline |
| `published_at` | datetime | Publication timestamp (ISO format) |
| `section` | string | News section (Politics, Business, etc.) |
| `wire_source` | string | Origin: `Reuters`, `AP`, `Original`, `Press Release` |

**Optional ground-truth columns** (for labeled datasets):
- `pattern_type`: `wire_echo`, `pr_amplification`, `narrative_convergence`, `synchronized_messaging`, `independent_reporting`
- `pattern_topic`: Story topic identifier
- `keywords`: List of keywords (as string)
- `pr_source`: PR origin if applicable
- `convergent_frame`: Convergent framing text
- `coordinated_frame`: Coordinated framing text

## Key Concepts

### Narrative Lifecycle Stages

Stories progress through a lifecycle as they spread:

```
NASCENT → EMERGING → SPREADING → ESTABLISHED → DECLINING → DORMANT
```

| Stage | Criteria |
|-------|----------|
| **Nascent** | 1-2 sources, < 6 hours old |
| **Emerging** | 3-5 sources, cross-outlet spread beginning |
| **Spreading** | 6+ sources, clear velocity increase |
| **Established** | Wide coverage, consistent framing |
| **Declining** | Velocity decreasing |
| **Dormant** | Coverage stopped |

### Core Metrics

| Metric | Description |
|--------|-------------|
| **Velocity** | Articles per hour covering the narrative |
| **Source Diversity** | Number of distinct outlet types (wire, national, local, independent) |
| **Bias Spread** | Standard deviation of outlet political bias (higher = cross-spectrum) |
| **Frame Coherence** | Semantic similarity within the narrative cluster |

### Signal Types

The system generates signals when it detects noteworthy patterns:

| Signal | Severity | Description |
|--------|----------|-------------|
| `rapid_emergence` | Medium-High | Fast initial spread of a new narrative |
| `cross_spectrum` | Medium-High | Story crossing political boundaries |
| `wire_cascade` | Low | Wire story triggering broad pickup |
| `pr_amplification` | Medium-High | PR-sourced narrative gaining traction |
| `frame_convergence` | Low | Organic convergence on similar framing |
| `synchronized_framing` | High-Critical | Potential coordinated messaging |
| `velocity_spike` | Medium | Sudden increase in coverage rate |

## Configuration

Copy `.env.example` to `.env` and customize:

```bash
# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Clustering parameters
HDBSCAN_MIN_CLUSTER_SIZE=3
HDBSCAN_MIN_SAMPLES=2

# Detection thresholds
VELOCITY_EMERGING_THRESHOLD=0.5
VELOCITY_SPREADING_THRESHOLD=2.0
SOURCE_DIVERSITY_THRESHOLD=3
BIAS_SPREAD_CROSS_SPECTRUM=0.7
FRAME_COHERENCE_THRESHOLD=0.85

# API settings
API_HOST=0.0.0.0
API_PORT=8000
```

## API Endpoints

Start the API server:

```bash
uvicorn src.api.main:app --reload --port 8000
```

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/data/load` | Load articles from CSV |
| `POST` | `/process/full` | Run complete pipeline |
| `GET` | `/clusters` | List narrative clusters |
| `GET` | `/clusters/{id}` | Get cluster details |
| `GET` | `/signals` | List detected signals |
| `PATCH` | `/signals/{id}/review` | Mark signal as reviewed |
| `GET` | `/articles` | List articles |

Full API documentation available at `http://localhost:8000/docs` when running.

## Technical Details

### Embedding Model

Uses [sentence-transformers](https://www.sbert.net/) with the `all-MiniLM-L6-v2` model by default:
- 384-dimensional embeddings
- Fast inference (~14k sentences/second on GPU)
- Good balance of speed and quality for news text

### Clustering Algorithm

[HDBSCAN](https://hdbscan.readthedocs.io/) (Hierarchical Density-Based Spatial Clustering):
- No need to specify number of clusters
- Handles noise (outlier articles)
- Finds clusters of varying densities

### Pipeline Flow

```
Articles (CSV)
     │
     ▼
┌─────────────┐
│  Ingestion  │  Load & validate data
└─────────────┘
     │
     ▼
┌─────────────┐
│  Embedding  │  Generate semantic vectors
└─────────────┘
     │
     ▼
┌─────────────┐
│ Clustering  │  Group into narrative clusters
└─────────────┘
     │
     ▼
┌─────────────┐
│ Detection   │  Analyze patterns, generate signals
└─────────────┘
     │
     ▼
  Signals + Clusters
```

## Roadmap

- [ ] **Weak supervision**: Labeling functions for training data generation
- [ ] **Classifier training**: Predict narrative patterns from features
- [ ] **Drift detection**: Monitor for embedding/topic drift over time
- [ ] **Database persistence**: SQLite/PostgreSQL for production use
- [ ] **Historical analysis**: Compare current narratives to past patterns
- [ ] **Entity extraction**: Track people, organizations, and claims

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for embedding models
- [HDBSCAN](https://hdbscan.readthedocs.io/) for clustering
- [FastAPI](https://fastapi.tiangolo.com/) and [Streamlit](https://streamlit.io/) for interfaces

---

Built with ❤️ for media research