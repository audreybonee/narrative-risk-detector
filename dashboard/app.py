"""
Streamlit dashboard for the Emergent Narrative Detection System.

Provides interactive visualization and analysis of emergent narratives.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.models import NarrativeStage, PatternType, SignalSeverity
from src.ingestion import load_articles_from_csv, get_dataset_stats, get_time_series_counts
from src.processing import EmbeddingGenerator, NarrativeClusterer, NarrativeDetector

# Page config
st.set_page_config(
    page_title="Emergent Narrative Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .signal-critical { color: #ff4b4b; font-weight: bold; }
    .signal-high { color: #ffa600; font-weight: bold; }
    .signal-medium { color: #00cc96; }
    .signal-low { color: #636efa; }
    .stage-nascent { color: #ab63fa; }
    .stage-emerging { color: #19d3f3; }
    .stage-spreading { color: #00cc96; }
    .stage-established { color: #636efa; }
</style>
""", unsafe_allow_html=True)


# ============== Session State ==============

if "articles" not in st.session_state:
    st.session_state.articles = []
if "clusters" not in st.session_state:
    st.session_state.clusters = []
if "signals" not in st.session_state:
    st.session_state.signals = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False


# ============== Sidebar ==============

st.sidebar.title("📰 Narrative Detection")
st.sidebar.markdown("---")

# Data loading
st.sidebar.header("Data")

data_source = st.sidebar.radio(
    "Data source",
    ["Default dataset", "Upload CSV"],
)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload news CSV", type=["csv"])
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = settings.RAW_DATA_DIR / "uploaded_news.csv"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_path = temp_path
    else:
        data_path = None
else:
    data_path = settings.RAW_DATA_DIR / "synthetic_news.csv"

if st.sidebar.button("Load Data", disabled=data_path is None):
    with st.spinner("Loading articles..."):
        try:
            st.session_state.articles = load_articles_from_csv(data_path)
            st.session_state.data_loaded = True
            st.session_state.pipeline_run = False
            st.session_state.clusters = []
            st.session_state.signals = []
            st.sidebar.success(f"Loaded {len(st.session_state.articles)} articles")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")

st.sidebar.markdown("---")

# Pipeline controls
st.sidebar.header("Pipeline")

if st.sidebar.button("Run Full Pipeline", disabled=not st.session_state.data_loaded):
    with st.spinner("Running pipeline..."):
        try:
            # Embed
            generator = EmbeddingGenerator()
            st.session_state.articles = generator.embed_articles(st.session_state.articles)

            # Cluster
            clusterer = NarrativeClusterer()
            st.session_state.clusters, noise = clusterer.cluster_articles(st.session_state.articles)

            # Detect
            detector = NarrativeDetector()
            st.session_state.signals = detector.analyze_clusters(st.session_state.clusters)

            st.session_state.pipeline_run = True
            st.sidebar.success("Pipeline complete!")
        except Exception as e:
            st.sidebar.error(f"Pipeline error: {e}")

st.sidebar.markdown("---")

# Filters
if st.session_state.pipeline_run:
    st.sidebar.header("Filters")

    # Stage filter
    stages = ["All"] + [s.value for s in NarrativeStage]
    selected_stage = st.sidebar.selectbox("Narrative Stage", stages)

    # Severity filter
    severities = ["All"] + [s.value for s in SignalSeverity]
    selected_severity = st.sidebar.selectbox("Signal Severity", severities)


# ============== Main Content ==============

st.title("🔍 Emergent Narrative Detection")
st.markdown("Tracking how stories spread and frames converge across the news ecosystem")

if not st.session_state.data_loaded:
    st.info("👈 Load a dataset from the sidebar to get started")

    st.markdown("""
    ## About This System

    The Emergent Narrative Detection System identifies and tracks how narratives propagate
    across news media. It detects:

    - **Wire cascades**: How wire stories spread through the ecosystem
    - **PR amplification**: Corporate messaging gaining media traction
    - **Frame convergence**: Multiple outlets converging on similar framing
    - **Cross-spectrum spread**: Stories crossing political boundaries
    - **Synchronized messaging**: Potential coordinated framing patterns

    ## Pattern Types

    | Pattern | Signal Strength | Description |
    |---------|----------------|-------------|
    | Wire Echo | Low | Standard wire story propagation |
    | Independent | Low | Original investigative journalism |
    | PR Amplification | Medium | Corporate/institutional PR pickup |
    | Narrative Convergence | Medium | Organic frame alignment |
    | Synchronized Messaging | High | Coordinated framing |
    """)

elif not st.session_state.pipeline_run:
    # Show data stats
    stats = get_dataset_stats(st.session_state.articles)

    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Articles", stats["count"])
    col2.metric("Outlets", stats["unique_outlets"])
    col3.metric("Topics", len(stats["topics"]))
    col4.metric("Date Range", f"{stats['date_range']['min'][:10]} to {stats['date_range']['max'][:10]}")

    st.markdown("---")

    # Distribution charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Outlet Types")
        df = pd.DataFrame(st.session_state.articles)
        outlet_counts = df["outlet_type"].apply(lambda x: x.value).value_counts()
        fig = px.pie(values=outlet_counts.values, names=outlet_counts.index, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Political Bias Distribution")
        bias_counts = df["outlet_bias"].apply(lambda x: x.value).value_counts()
        bias_order = ["left", "center-left", "center", "center-right", "right"]
        bias_counts = bias_counts.reindex(bias_order).fillna(0)
        fig = px.bar(x=bias_counts.index, y=bias_counts.values, color=bias_counts.index,
                    color_discrete_map={
                        "left": "#3366cc",
                        "center-left": "#6699cc",
                        "center": "#999999",
                        "center-right": "#cc6666",
                        "right": "#cc3333",
                    })
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Articles")
        st.plotly_chart(fig, use_container_width=True)

    # Pattern types
    st.subheader("Pattern Types (Ground Truth)")
    pattern_counts = df["pattern_type"].apply(lambda x: x.value if x else "unknown").value_counts()
    fig = px.bar(x=pattern_counts.index, y=pattern_counts.values, color=pattern_counts.index)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Articles")
    st.plotly_chart(fig, use_container_width=True)

    st.info("👈 Click 'Run Full Pipeline' in the sidebar to analyze narratives")

else:
    # Full dashboard with results

    # Apply filters
    filtered_clusters = st.session_state.clusters
    filtered_signals = st.session_state.signals

    if selected_stage != "All":
        filtered_clusters = [c for c in filtered_clusters if c.stage.value == selected_stage]

    if selected_severity != "All":
        filtered_signals = [s for s in filtered_signals if s.severity.value == selected_severity]

    # Metrics row
    st.header("📈 Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Articles", len(st.session_state.articles))
    col2.metric("Clusters", len(st.session_state.clusters))
    col3.metric("Active Signals", len([s for s in filtered_signals if not s.reviewed]))
    col4.metric("High Priority", len([s for s in filtered_signals if s.severity.value in ["high", "critical"]]))
    col5.metric("Cross-Spectrum", len([s for s in filtered_signals if s.signal_type.value == "cross_spectrum"]))

    st.markdown("---")

    # Signals section
    st.header("🚨 Detected Signals")

    if not filtered_signals:
        st.info("No signals detected with current filters")
    else:
        for signal in filtered_signals[:10]:  # Show top 10
            severity_class = f"signal-{signal.severity.value}"

            with st.expander(f"{'🔴' if signal.severity.value == 'critical' else '🟠' if signal.severity.value == 'high' else '🟢' if signal.severity.value == 'medium' else '🔵'} {signal.title}", expanded=signal.severity.value in ["critical", "high"]):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**{signal.description}**")
                    st.markdown(f"📰 *{signal.headline}*")
                    st.markdown(f"**Stage:** {signal.narrative_stage.value} | **Type:** {signal.signal_type.value}")

                with col2:
                    st.metric("Outlets", signal.unique_outlets)
                    st.metric("Velocity", f"{signal.velocity:.1f}/hr")
                    st.metric("Bias Spread", f"{signal.bias_spread:.2f}")

                if signal.frame_text:
                    st.info(f"💬 Frame: *{signal.frame_text}*")

    st.markdown("---")

    # Clusters section
    st.header("📚 Narrative Clusters")

    # Cluster visualization
    if filtered_clusters:
        cluster_data = []
        for c in filtered_clusters:
            cluster_data.append({
                "Topic": c.topic or "Unknown",
                "Size": c.size,
                "Outlets": c.unique_outlets,
                "Stage": c.stage.value,
                "Velocity": c.velocity,
                "Bias Spread": c.bias_spread,
                "Duration (hrs)": c.duration_hours,
            })

        cluster_df = pd.DataFrame(cluster_data)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cluster Size vs Velocity")
            fig = px.scatter(
                cluster_df,
                x="Velocity",
                y="Size",
                color="Stage",
                size="Outlets",
                hover_data=["Topic", "Bias Spread"],
                title="",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("By Lifecycle Stage")
            stage_counts = cluster_df["Stage"].value_counts()
            fig = px.pie(values=stage_counts.values, names=stage_counts.index, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        # Cluster table
        st.subheader("Cluster Details")
        st.dataframe(
            cluster_df.style.background_gradient(subset=["Size", "Velocity"], cmap="YlOrRd"),
            use_container_width=True,
        )

        # Detailed cluster view
        st.subheader("Explore Cluster")
        cluster_options = {f"{c.topic or 'Unknown'} ({c.size} articles)": c for c in filtered_clusters}
        selected_cluster_name = st.selectbox("Select cluster", list(cluster_options.keys()))
        selected_cluster = cluster_options[selected_cluster_name]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Representative Headline:** {selected_cluster.representative_headline}")
            st.markdown(f"**Keywords:** {', '.join([k[0] for k in selected_cluster.get_keywords_frequency(5)])}")

            # Timeline
            timeline_data = []
            for article in selected_cluster.articles:
                timeline_data.append({
                    "Time": article.published_at,
                    "Outlet": article.outlet,
                    "Type": article.outlet_type.value,
                    "Bias": article.outlet_bias.value,
                })

            timeline_df = pd.DataFrame(timeline_data)
            fig = px.scatter(
                timeline_df,
                x="Time",
                y="Outlet",
                color="Bias",
                symbol="Type",
                title="Publication Timeline",
                color_discrete_map={
                    "left": "#3366cc",
                    "center-left": "#6699cc",
                    "center": "#999999",
                    "center-right": "#cc6666",
                    "right": "#cc3333",
                },
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Size", selected_cluster.size)
            st.metric("Unique Outlets", selected_cluster.unique_outlets)
            st.metric("Source Diversity", selected_cluster.source_diversity)
            st.metric("Velocity", f"{selected_cluster.velocity:.2f}/hr")
            st.metric("Bias Spread", f"{selected_cluster.bias_spread:.2f}")
            st.metric("Stage", selected_cluster.stage.value)

            if selected_cluster.common_frame:
                st.info(f"**Frame:** {selected_cluster.common_frame}")


# Footer
st.markdown("---")
st.markdown("Built with ❤️ for computational social science research")