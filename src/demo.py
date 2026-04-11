"""Simple CLI demo for inference using a trained encoder + logistic regression head.

Usage:
    python src/demo.py --model_dir results/baseline/seed_0

Expects the following structure inside --model_dir:
    encoder/                # saved SentenceTransformer
    classifier.json         # serialized LogisticRegression params

The script loads both, encodes input text, and prints predicted class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

THRESHOLD = 0.1

LABEL_MAP = {
    0: "BUSINESS",
    1: "ENTERTAINMENT",
    2: "POLITICS & GOVERNANCE",
    3: "SPORTS",
    4: "EDUCATION & TECHNOLOGY",
}


# -----------------------------------------------------------------------------
# Loading utilities
# -----------------------------------------------------------------------------


def load_encoder(model_dir: Path) -> SentenceTransformer:
    """Load SentenceTransformer encoder from disk."""
    encoder_path: Path = model_dir / "encoder"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found at: {encoder_path}")
    return SentenceTransformer(str(encoder_path))


def load_classifier(model_dir: Path) -> LogisticRegression:
    """Reconstruct LogisticRegression from saved JSON."""
    clf_path: Path = model_dir / "classifier.json"
    if not clf_path.exists():
        raise FileNotFoundError(f"Classifier not found at: {clf_path}")

    with clf_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    clf = LogisticRegression()
    clf.classes_ = np.array(data["classes"])  
    clf.coef_ = np.array(data["coef"])  
    clf.intercept_ = np.array(data["intercept"])
    return clf


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------


def predict(
    model: SentenceTransformer,
    clf: LogisticRegression,
    texts: List[str],
) -> List[str]:
    """Predict labels for a list of texts with confidence threshold."""
    embeddings: np.ndarray = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    probs: np.ndarray = clf.predict_proba(embeddings)

    results: List[str] = []

    for prob_vec in probs:
        max_idx: int = int(np.argmax(prob_vec))
        max_prob: float = float(np.max(prob_vec))

        if max_prob < THRESHOLD:
            results.append("OTHER")
        else:
            label_name = LABEL_MAP[max_idx]
            results.append(f"{label_name} ({max_prob:.2f})")

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SetFit Hard-Negative Demo")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory (e.g., results/baseline/seed_0)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive mode",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text input (non-interactive mode)",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    model = load_encoder(model_dir)
    clf = load_classifier(model_dir)

    if args.interactive:
        print("\nEnter text (type 'exit' to quit):")
        while True:
            user_input = input("> ")
            if user_input.lower() in {"exit", "quit"}:
                break

            preds = predict(model, clf, [user_input])
            print(f"Prediction: {preds[0]}")

    else:
        if args.text is None:
            raise ValueError("Provide --text or use --interactive mode")

        preds = predict(model, clf, [args.text])
        print(f"Prediction: {preds[0]}")


if __name__ == "__main__":
    main()
