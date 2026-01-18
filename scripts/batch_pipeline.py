"""
Batch processing pipeline for the Emergent Narrative Detection System.

Processes news articles through the full pipeline and outputs results.

Usage:
    python scripts/batch_pipeline.py --input data/raw/synthetic_news.csv --window 4h
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.ingestion import (
    load_articles_from_csv,
    get_dataset_stats,
    TimeWindowIterator,
    group_articles_by_topic,
)
from src.processing import (
    EmbeddingGenerator,
    NarrativeClusterer,
    NarrativeDetector,
    detect_stage_transitions,
)
from src.models import NarrativeCluster, NarrativeSignal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the narrative detection pipeline on a news dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(settings.RAW_DATA_DIR / "synthetic_news.csv"),
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.PROCESSED_DATA_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="4h",
        help="Time window size (e.g., 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Run in streaming mode (process window by window)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    return parser.parse_args()


def run_batch_pipeline(
    input_path: str,
    output_dir: str,
    verbose: bool = False,
) -> dict:
    """
    Run the full batch pipeline on all articles at once.

    Args:
        input_path: Path to input CSV
        output_dir: Directory for output files
        verbose: Print progress

    Returns:
        Results dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    if verbose:
        print(f"Loading articles from {input_path}...")
    articles = load_articles_from_csv(input_path)
    stats = get_dataset_stats(articles)

    if verbose:
        print(f"Loaded {len(articles)} articles from {stats['unique_outlets']} outlets")

    # Generate embeddings
    if verbose:
        print("Generating embeddings...")
    generator = EmbeddingGenerator()
    articles = generator.embed_articles(articles)

    if verbose:
        print(f"Generated embeddings for {len(articles)} articles")

    # Cluster articles
    if verbose:
        print("Clustering articles...")
    clusterer = NarrativeClusterer()
    clusters, noise = clusterer.cluster_articles(articles)

    if verbose:
        print(f"Found {len(clusters)} clusters, {len(noise)} noise articles")

    # Detect signals
    if verbose:
        print("Detecting narrative signals...")
    detector = NarrativeDetector()
    signals = detector.analyze_clusters(clusters)

    if verbose:
        print(f"Detected {len(signals)} signals")

    # Prepare results
    results = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "input_file": input_path,
        "stats": stats,
        "clusters": [c.to_summary_dict() for c in clusters],
        "signals": [s.to_dict() for s in signals],
        "summary": {
            "total_articles": len(articles),
            "total_clusters": len(clusters),
            "noise_articles": len(noise),
            "total_signals": len(signals),
            "signals_by_severity": {},
            "signals_by_type": {},
        },
    }

    # Count signals by severity and type
    for signal in signals:
        sev = signal.severity.value
        results["summary"]["signals_by_severity"][sev] = \
            results["summary"]["signals_by_severity"].get(sev, 0) + 1

        sig_type = signal.signal_type.value
        results["summary"]["signals_by_type"][sig_type] = \
            results["summary"]["signals_by_type"].get(sig_type, 0) + 1

    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Full results
    results_file = output_path / f"results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"Results saved to {results_file}")

    # Signals only (for quick review)
    signals_file = output_path / f"signals_{timestamp}.json"
    with open(signals_file, "w") as f:
        json.dump({
            "run_timestamp": results["run_timestamp"],
            "total_signals": len(signals),
            "signals": results["signals"],
        }, f, indent=2, default=str)

    if verbose:
        print(f"Signals saved to {signals_file}")

    # Clusters only
    clusters_file = output_path / f"clusters_{timestamp}.json"
    with open(clusters_file, "w") as f:
        json.dump({
            "run_timestamp": results["run_timestamp"],
            "total_clusters": len(clusters),
            "clusters": results["clusters"],
        }, f, indent=2, default=str)

    if verbose:
        print(f"Clusters saved to {clusters_file}")

    return results


def run_streaming_pipeline(
    input_path: str,
    output_dir: str,
    window_size: str = "4h",
    verbose: bool = False,
) -> dict:
    """
    Run the pipeline in streaming mode, processing window by window.

    Args:
        input_path: Path to input CSV
        output_dir: Directory for output files
        window_size: Time window size
        verbose: Print progress

    Returns:
        Results dictionary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    if verbose:
        print(f"Loading articles from {input_path}...")
    articles = load_articles_from_csv(input_path)

    # Generate embeddings for all (can also be done incrementally)
    if verbose:
        print("Generating embeddings...")
    generator = EmbeddingGenerator()
    articles = generator.embed_articles(articles)

    # Initialize components
    clusterer = NarrativeClusterer()
    detector = NarrativeDetector()

    # Track state across windows
    all_clusters: list[NarrativeCluster] = []
    all_signals: list[NarrativeSignal] = []
    previous_clusters: list[NarrativeCluster] = []

    # Process windows
    iterator = TimeWindowIterator(articles, window_size=window_size)

    window_results = []

    for window_start, window_end, window_articles in iterator:
        if verbose:
            print(f"\nProcessing window: {window_start} to {window_end}")
            print(f"  Articles in window: {len(window_articles)}")

        if not window_articles:
            continue

        # Cluster this window
        clusters, noise = clusterer.cluster_articles(window_articles)

        if verbose:
            print(f"  Clusters found: {len(clusters)}")

        # Detect signals
        signals = detector.analyze_clusters(clusters)

        # Check for stage transitions
        if previous_clusters:
            transition_signals = detect_stage_transitions(clusters, previous_clusters)
            signals.extend(transition_signals)

        if verbose:
            print(f"  Signals detected: {len(signals)}")

        # Store results
        all_clusters.extend(clusters)
        all_signals.extend(signals)

        window_results.append({
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "articles": len(window_articles),
            "clusters": len(clusters),
            "signals": len(signals),
        })

        previous_clusters = clusters

    # Compile final results
    results = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "mode": "streaming",
        "window_size": window_size,
        "input_file": input_path,
        "windows_processed": len(window_results),
        "window_details": window_results,
        "summary": {
            "total_articles": len(articles),
            "total_clusters": len(all_clusters),
            "total_signals": len(all_signals),
        },
        "signals": [s.to_dict() for s in all_signals],
    }

    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"streaming_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\nStreaming results saved to {results_file}")

    return results


def print_summary(results: dict):
    """Print a formatted summary of results."""
    print("\n" + "=" * 60)
    print("NARRATIVE DETECTION RESULTS")
    print("=" * 60)

    summary = results.get("summary", {})

    print(f"\nArticles processed: {summary.get('total_articles', 'N/A')}")
    print(f"Clusters found: {summary.get('total_clusters', 'N/A')}")
    print(f"Signals detected: {summary.get('total_signals', 'N/A')}")

    if "signals_by_severity" in summary:
        print("\nSignals by Severity:")
        for severity, count in sorted(summary["signals_by_severity"].items()):
            print(f"  {severity}: {count}")

    if "signals_by_type" in summary:
        print("\nSignals by Type:")
        for sig_type, count in sorted(summary["signals_by_type"].items()):
            print(f"  {sig_type}: {count}")

    # Print top signals
    signals = results.get("signals", [])
    high_priority = [s for s in signals if s.get("severity") in ["high", "critical"]]

    if high_priority:
        print("\n" + "-" * 60)
        print("HIGH PRIORITY SIGNALS")
        print("-" * 60)
        for signal in high_priority[:5]:
            print(f"\n[{signal['severity'].upper()}] {signal['title']}")
            print(f"  {signal['description']}")
            print(f"  Headline: {signal['headline'][:80]}...")

    print("\n" + "=" * 60)


def main():
    args = parse_args()

    print("Emergent Narrative Detection Pipeline")
    print("=" * 40)

    if args.streaming:
        results = run_streaming_pipeline(
            input_path=args.input,
            output_dir=args.output,
            window_size=args.window,
            verbose=args.verbose,
        )
    else:
        results = run_batch_pipeline(
            input_path=args.input,
            output_dir=args.output,
            verbose=args.verbose,
        )

    print_summary(results)


if __name__ == "__main__":
    main()