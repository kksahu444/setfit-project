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
from typing import Any, Dict, List, Optional, Sequence

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
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: Optional[List[List[float]]]


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
            data: Dict[str, Any] = json.load(f)

        seed_id: int = int(seed_dir.name.split("_")[1])

        raw_per_class = data.get("per_class", {})
        per_class: Dict[str, Dict[str, float]] = {}
        if isinstance(raw_per_class, dict):
            for label, stats in raw_per_class.items():
                if isinstance(stats, dict):
                    per_class[str(label)] = {
                        "precision": float(stats.get("precision", 0.0)),
                        "recall": float(stats.get("recall", 0.0)),
                        "f1": float(stats.get("f1", 0.0)),
                        "support": float(stats.get("support", 0.0)),
                    }

        raw_confusion = data.get("confusion_matrix")
        confusion: Optional[List[List[float]]] = None
        if isinstance(raw_confusion, list):
            confusion = [
                [float(v) for v in row]
                for row in raw_confusion
                if isinstance(row, list)
            ]

        results.append(
            SeedResult(
                seed=seed_id,
                accuracy=float(data["accuracy"]),
                precision=float(data["precision"]),
                recall=float(data["recall"]),
                f1=float(data["f1"]),
                per_class=per_class,
                confusion_matrix=confusion,
            )
        )

    return results


def compute_summary(results: Sequence[SeedResult]) -> Dict[str, float]:
    """Compute mean and std for all metrics."""
    if not results:
        raise ValueError("No seed results found; cannot compute summary.")

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


def compute_per_class_summary(
    results: Sequence[SeedResult],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-class metrics across seeds."""
    labels = sorted({label for r in results for label in r.per_class.keys()})
    summary: Dict[str, Dict[str, float]] = {}

    for label in labels:
        p_vals = [r.per_class[label]["precision"] for r in results if label in r.per_class]
        r_vals = [r.per_class[label]["recall"] for r in results if label in r.per_class]
        f_vals = [r.per_class[label]["f1"] for r in results if label in r.per_class]
        s_vals = [r.per_class[label]["support"] for r in results if label in r.per_class]

        summary[label] = {
            "precision_mean": float(np.mean(p_vals)) if p_vals else 0.0,
            "precision_std": float(np.std(p_vals)) if p_vals else 0.0,
            "recall_mean": float(np.mean(r_vals)) if r_vals else 0.0,
            "recall_std": float(np.std(r_vals)) if r_vals else 0.0,
            "f1_mean": float(np.mean(f_vals)) if f_vals else 0.0,
            "f1_std": float(np.std(f_vals)) if f_vals else 0.0,
            "support_mean": float(np.mean(s_vals)) if s_vals else 0.0,
        }

    return summary


def compute_confusion_summary(results: Sequence[SeedResult]) -> Dict[str, List[List[float]]]:
    """Aggregate confusion matrices across seeds (sum + row-normalized)."""
    matrices = [np.array(r.confusion_matrix, dtype=float) for r in results if r.confusion_matrix]
    if not matrices:
        return {"sum": [], "row_normalized": []}

    cm_sum = np.sum(matrices, axis=0)
    row_sums = cm_sum.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_sum / row_sums

    return {
        "sum": cm_sum.tolist(),
        "row_normalized": cm_norm.tolist(),
    }


def save_summary(summary: Dict[str, float], output_path: Path) -> None:
    """Save summary JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_json_dict(obj: Dict[str, Any], output_path: Path) -> None:
    """Save any dictionary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(results: Sequence[SeedResult], output_path: Path) -> None:
    """Save per-seed results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "accuracy", "precision", "recall", "f1"])

        for r in results:
            writer.writerow([r.seed, r.accuracy, r.precision, r.recall, r.f1])


def save_per_class_csv(
    per_class_summary: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Save aggregated per-class metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "f1_mean",
                "f1_std",
                "support_mean",
            ]
        )

        for label, metrics in sorted(per_class_summary.items()):
            writer.writerow(
                [
                    label,
                    metrics.get("precision_mean", 0.0),
                    metrics.get("precision_std", 0.0),
                    metrics.get("recall_mean", 0.0),
                    metrics.get("recall_std", 0.0),
                    metrics.get("f1_mean", 0.0),
                    metrics.get("f1_std", 0.0),
                    metrics.get("support_mean", 0.0),
                ]
            )


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
    if not results:
        raise ValueError(f"No metrics.json files found in {results_dir}")

    summary = compute_summary(results)
    per_class_summary = compute_per_class_summary(results)
    confusion_summary = compute_confusion_summary(results)

    save_summary(summary, output_dir / "results_summary.json")
    save_csv(results, output_dir / "results.csv")
    save_json_dict(per_class_summary, output_dir / "per_class_summary.json")
    save_per_class_csv(per_class_summary, output_dir / "per_class_summary.csv")
    save_json_dict(confusion_summary, output_dir / "confusion_summary.json")

    plot_results(results, per_seed_title, output_dir / "per_seed.png")
    plot_summary_bar(summary, summary_label, output_dir / "summary.png")

    print("Evaluation complete")
    print(summary)


if __name__ == "__main__":
    _cli()
