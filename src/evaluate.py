"""Evaluation and reporting utilities.

This module aggregates per-seed metrics, computes summary statistics,
and generates plots and tables for presentation.

Outputs:
- summary.json (mean/std)
- results.csv (per-seed metrics)
- plots (bar charts with optional error bars)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import json
import csv

import numpy as np
import matplotlib.pyplot as plt
import argparse
from config_utils import load_config


@dataclass(frozen=True)
class SeedResult:
    """Container for a single seed result."""

    seed: int
    accuracy: float
    precision: float
    recall: float
    f1: float


def load_seed_results(results_dir: Path) -> List[SeedResult]:
    """Load metrics.json from all seed directories.

    Args:
        results_dir: Path to experiment directory

    Returns:
        List of SeedResult
    """
    results: List[SeedResult] = []

    for seed_dir in sorted(results_dir.glob("seed_*")):
        metrics_path = seed_dir / "metrics.json"

        if not metrics_path.exists():
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            data: Dict[str, float] = json.load(f)

        seed_id: int = int(seed_dir.name.split("_")[1])

        results.append(
            SeedResult(
                seed=seed_id,
                accuracy=float(data["accuracy"]),
                precision=float(data["precision"]),
                recall=float(data["recall"]),
                f1=float(data["f1"]),
            )
        )

    return results


def compute_summary(results: Sequence[SeedResult]) -> Dict[str, float]:
    """Compute mean and std for all metrics."""
    accuracies: np.ndarray = np.array([r.accuracy for r in results])
    precisions: np.ndarray = np.array([r.precision for r in results])
    recalls: np.ndarray = np.array([r.recall for r in results])
    f1s: np.ndarray = np.array([r.f1 for r in results])

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }


def save_summary(summary: Dict[str, float], output_path: Path) -> None:
    """Save summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_csv(results: Sequence[SeedResult], output_path: Path) -> None:
    """Save per-seed results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "accuracy", "precision", "recall", "f1"])

        for r in results:
            writer.writerow([r.seed, r.accuracy, r.precision, r.recall, r.f1])


def plot_results(
    results: Sequence[SeedResult],
    title: str,
    output_path: Path,
) -> None:
    """Generate bar chart of per-seed results.

    Args:
        results: Per-seed results
        title: Plot title
        output_path: Where to save plot
    """
    seeds: List[int] = [r.seed for r in results]
    accuracies: List[float] = [r.accuracy for r in results]

    plt.figure()
    plt.bar(seeds, accuracies)

    plt.xlabel("Seed")
    plt.ylabel("Accuracy")
    plt.title(title)

    for i, acc in enumerate(accuracies):
        plt.text(seeds[i], acc, f"{acc:.2f}", ha="center", va="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_summary_bar(
    summary: Dict[str, float],
    label: str,
    output_path: Path,
) -> None:
    """Plot single bar with error bar (mean ± std).

    Args:
        summary: Summary metrics
        label: Label for bar
        output_path: Save path
    """
    mean_val: float = summary["accuracy_mean"]
    std_val: float = summary["accuracy_std"]

    plt.figure()
    plt.bar([label], [mean_val], yerr=[std_val], capsize=5)

    plt.ylabel("Accuracy")
    plt.title("Mean Performance with Std")

    plt.text(0, mean_val, f"{mean_val:.2f}", ha="center", va="bottom")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment results.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    config = load_config(args.config)
    evaluate_cfg = config.get("evaluate", {})

    results_dir_value = args.results_dir or str(
        evaluate_cfg.get("results_dir", "results/baseline")
    )
    output_dir_value = args.output_dir or str(
        evaluate_cfg.get("output_dir", "results/baseline")
    )
    per_seed_title = str(evaluate_cfg.get("per_seed_plot_title", "Per-seed Accuracy"))
    summary_label = str(evaluate_cfg.get("summary_bar_label", "Model"))

    results_dir = Path(results_dir_value)
    output_dir = Path(output_dir_value)

    results = load_seed_results(results_dir)

    summary = compute_summary(results)

    save_summary(summary, output_dir / "results_summary.json")
    save_csv(results, output_dir / "results.csv")

    plot_results(results, per_seed_title, output_dir / "per_seed.png")
    plot_summary_bar(summary, summary_label, output_dir / "summary.png")

    print("Evaluation complete")
    print(summary)


if __name__ == "__main__":
    _cli()
