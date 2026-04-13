"""Run research matrix experiments for SetFit-HNM hypotheses and ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config_utils import load_config
from evaluate import (
    compute_confusion_summary,
    compute_per_class_summary,
    compute_summary,
    load_seed_results,
    plot_results,
    plot_summary_bar,
    save_csv,
    save_json_dict,
    save_per_class_csv,
    save_summary,
)
from hard_negative import HardNegativeConfig
from train import run_experiment


def _parse_list_override(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values if values else None


def _resolve_dataset_keys(
    config: Dict[str, Any],
    override_keys: Optional[Sequence[str]],
) -> List[str]:
    datasets_cfg = config.get("datasets", {})
    if not isinstance(datasets_cfg, dict) or not datasets_cfg:
        raise ValueError("Config must define a non-empty datasets mapping.")

    matrix_cfg = config.get("research", {}).get("experiment_matrix", {})
    default_keys = matrix_cfg.get("datasets", list(datasets_cfg.keys()))

    keys = list(override_keys) if override_keys is not None else list(default_keys)
    for key in keys:
        if key not in datasets_cfg:
            raise ValueError(f"Unknown dataset key '{key}'. Available: {list(datasets_cfg.keys())}")
    return keys


def _resolve_int_values(
    default_values: Sequence[int],
    override: Optional[str],
    name: str,
) -> List[int]:
    if override is None:
        return [int(v) for v in default_values]

    parsed: List[int] = []
    for token in override.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid integer in --{name}: {token}") from exc

    if not parsed:
        raise ValueError(f"--{name} must provide at least one integer value")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hypothesis and ablation experiment matrix.")
    parser.add_argument("--config", type=str, default="configs/research.yaml")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset keys")
    parser.add_argument("--k_values", type=str, default=None, help="Comma-separated K values")
    parser.add_argument("--pair_strategies", type=str, default=None, help="Comma-separated strategies")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated integer seeds")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    config = load_config(args.config)
    matrix_cfg = config.get("research", {}).get("experiment_matrix", {})
    train_defaults = config.get("train_defaults", {})

    dataset_keys = _resolve_dataset_keys(config, _parse_list_override(args.datasets))
    k_values = _resolve_int_values(matrix_cfg.get("k_values", [8]), args.k_values, "k_values")
    pair_strategies = (
        _parse_list_override(args.pair_strategies)
        or [str(v) for v in matrix_cfg.get("pair_strategies", ["random", "hard"])]
    )
    seeds = _resolve_int_values(matrix_cfg.get("seeds", [0, 1, 2, 3, 4]), args.seeds, "seeds")

    output_root = Path(str(matrix_cfg.get("output_root", "results/research")))
    output_root.mkdir(parents=True, exist_ok=True)

    num_pos_per_anchor = int(train_defaults.get("pair_sampling", {}).get("num_pos_per_anchor", 1))
    num_neg_per_anchor = int(train_defaults.get("pair_sampling", {}).get("num_neg_per_anchor", 1))

    hard_negative_cfg = HardNegativeConfig(
        k_hard=int(train_defaults.get("hard_negative", {}).get("k_hard", 2)),
        k_easy=int(train_defaults.get("hard_negative", {}).get("k_easy", 1)),
    )

    rows: List[Dict[str, Any]] = []

    for dataset_key in dataset_keys:
        dataset_cfg = config["datasets"][dataset_key]
        dataset_name = str(dataset_cfg["name"])
        text_column = str(dataset_cfg.get("text_column", "text"))
        label_column = str(dataset_cfg.get("label_column", "label"))

        label_names_raw = dataset_cfg.get("label_names")
        label_names: Optional[List[str]] = None
        if isinstance(label_names_raw, list):
            label_names = [str(v) for v in label_names_raw]

        for k in k_values:
            for strategy in pair_strategies:
                run_dir = output_root / dataset_key / f"k_{k}" / strategy
                run_desc = f"dataset={dataset_key} | K={k} | strategy={strategy}"
                print(f"\n[RUN] {run_desc}")

                if args.dry_run:
                    continue

                run_experiment(
                    dataset_name=dataset_name,
                    text_column=text_column,
                    label_column=label_column,
                    label_names=label_names,
                    seeds=seeds,
                    k_per_class=k,
                    test_size=float(train_defaults.get("test_size", 0.2)),
                    output_dir=run_dir,
                    mode="baseline",
                    pair_strategy=strategy,
                    base_model_name=str(train_defaults.get("base_model_name", "sentence-transformers/paraphrase-mpnet-base-v2")),
                    batch_size=int(train_defaults.get("batch_size", 16)),
                    epochs=int(train_defaults.get("epochs", 1)),
                    lr_max_iter=int(train_defaults.get("lr_max_iter", 1000)),
                    num_pos_per_anchor=num_pos_per_anchor,
                    num_neg_per_anchor=num_neg_per_anchor,
                    hard_negative_config=hard_negative_cfg,
                )

                seed_results = load_seed_results(run_dir)
                if not seed_results:
                    raise ValueError(f"No seed metrics were produced for {run_desc}")
                summary = compute_summary(seed_results)
                per_class_summary = compute_per_class_summary(seed_results)
                confusion_summary = compute_confusion_summary(seed_results)

                summary_with_meta: Dict[str, Any] = dict(summary)
                summary_with_meta["dataset"] = dataset_key
                summary_with_meta["dataset_name"] = dataset_name
                summary_with_meta["k_per_class"] = k
                summary_with_meta["pair_strategy"] = strategy
                summary_with_meta["num_seeds"] = len(seeds)

                save_summary(summary, run_dir / "results_summary_metrics.json")
                save_json_dict(summary_with_meta, run_dir / "results_summary.json")
                save_csv(seed_results, run_dir / "results.csv")
                save_json_dict(per_class_summary, run_dir / "per_class_summary.json")
                save_per_class_csv(per_class_summary, run_dir / "per_class_summary.csv")
                save_json_dict(confusion_summary, run_dir / "confusion_summary.json")
                plot_results(seed_results, f"{dataset_key} | K={k} | {strategy}", run_dir / "per_seed.png")
                plot_summary_bar(summary, f"{dataset_key}-{strategy}-K{k}", run_dir / "summary.png")

                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "dataset_name": dataset_name,
                        "k_per_class": k,
                        "pair_strategy": strategy,
                        "num_seeds": len(seeds),
                        "accuracy_mean": summary["accuracy_mean"],
                        "accuracy_std": summary["accuracy_std"],
                        "precision_mean": summary["precision_mean"],
                        "recall_mean": summary["recall_mean"],
                        "f1_mean": summary["f1_mean"],
                        "results_dir": str(run_dir),
                    }
                )

    if args.dry_run:
        print("\nDry-run complete. No training jobs executed.")
        return

    manifest_csv = output_root / "manifest.csv"
    manifest_json = output_root / "manifest.json"

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_key",
                "dataset_name",
                "k_per_class",
                "pair_strategy",
                "num_seeds",
                "accuracy_mean",
                "accuracy_std",
                "precision_mean",
                "recall_mean",
                "f1_mean",
                "results_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nCompleted {len(rows)} experiment cells.")
    print(f"Manifest written to: {manifest_csv}")


if __name__ == "__main__":
    main()
