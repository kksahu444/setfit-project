"""Threshold analysis for uncertainty-aware/open-set behavior on saved model runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.metrics import accuracy_score

from config_utils import load_config
from data_loader import load_local_jsonl
from demo import load_classifier, load_encoder


def _parse_thresholds(raw: Optional[str], defaults: List[float]) -> List[float]:
    if raw is None:
        return defaults

    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid threshold value: {token}") from exc

    if not values:
        raise ValueError("At least one threshold is required.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confidence-threshold analysis on a trained run.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path like results/.../seed_0")
    parser.add_argument("--config", type=str, default="configs/research.yaml")
    parser.add_argument("--thresholds", type=str, default=None, help="Comma-separated threshold list")
    parser.add_argument("--test_jsonl", type=str, default=None)
    parser.add_argument("--unknown_jsonl", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()

    config = load_config(args.config)
    threshold_cfg = config.get("threshold_analysis", {})
    default_thresholds = [float(v) for v in threshold_cfg.get("thresholds", [0.1, 0.3, 0.5, 0.7, 0.9])]
    thresholds = _parse_thresholds(args.thresholds, default_thresholds)

    model_dir = Path(args.model_dir)
    test_jsonl = Path(args.test_jsonl) if args.test_jsonl else (model_dir / "test.jsonl")

    if not test_jsonl.exists():
        raise FileNotFoundError(f"Test JSONL not found: {test_jsonl}")

    encoder = load_encoder(model_dir)
    clf = load_classifier(model_dir)

    test_samples = load_local_jsonl(test_jsonl)
    test_texts = [s.text for s in test_samples]
    test_labels = np.array([s.label for s in test_samples])

    test_embeddings = encoder.encode(
        test_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    test_probs = clf.predict_proba(test_embeddings)
    test_preds = np.argmax(test_probs, axis=1)
    test_conf = np.max(test_probs, axis=1)

    unknown_conf: Optional[np.ndarray] = None
    if args.unknown_jsonl:
        unknown_path = Path(args.unknown_jsonl)
        unknown_samples = load_local_jsonl(unknown_path)
        unknown_texts = [s.text for s in unknown_samples]
        if unknown_texts:
            unknown_embeddings = encoder.encode(
                unknown_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            unknown_probs = clf.predict_proba(unknown_embeddings)
            unknown_conf = np.max(unknown_probs, axis=1)

    rows = []
    for thr in thresholds:
        accepted_mask = test_conf >= thr
        coverage = float(np.mean(accepted_mask))
        rejection_rate = 1.0 - coverage

        if np.any(accepted_mask):
            accepted_acc = float(accuracy_score(test_labels[accepted_mask], test_preds[accepted_mask]))
        else:
            accepted_acc = 0.0

        row = {
            "threshold": thr,
            "coverage": coverage,
            "rejection_rate": rejection_rate,
            "accepted_accuracy": accepted_acc,
        }

        if unknown_conf is not None and unknown_conf.size > 0:
            false_accept_rate = float(np.mean(unknown_conf >= thr))
            unknown_reject_rate = 1.0 - false_accept_rate
            row["unknown_false_accept_rate"] = false_accept_rate
            row["unknown_reject_rate"] = unknown_reject_rate

        rows.append(row)

    output_csv = Path(args.output_csv) if args.output_csv else (model_dir / str(threshold_cfg.get("output_filename", "threshold_analysis.csv")))
    output_json = Path(args.output_json) if args.output_json else (model_dir / "threshold_analysis.json")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Threshold analysis saved to: {output_csv}")


if __name__ == "__main__":
    main()
