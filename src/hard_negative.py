"""Hard negative mining utilities for contrastive pair construction.

This module builds on top of baseline pair construction by selecting
semantically similar but differently labeled examples as negatives.

These pairs provide stronger training signals compared to random negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence
import random

import numpy as np
from sentence_transformers import SentenceTransformer

from data_loader import Sample
from pair_builder import Pair


@dataclass(frozen=True)
class HardNegativeConfig:
    """Configuration for hard negative mining.

    Attributes:
        k_hard: Number of hard negatives per anchor
        k_easy: Number of random negatives per anchor (optional mix)
    """

    k_hard: int = 2
    k_easy: int = 0


def encode_samples(
    model: SentenceTransformer,
    samples: Sequence[Sample],
) -> np.ndarray:
    """Encode samples into normalized embeddings.

    Args:
        model: SentenceTransformer model
        samples: Input samples

    Returns:
        Embedding matrix of shape (N, D)
    """
    texts: List[str] = [s.text for s in samples]

    embeddings: np.ndarray = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings


def build_hard_negative_pairs(
    samples: Sequence[Sample],
    embeddings: np.ndarray,
    config: HardNegativeConfig,
) -> List[Pair]:
    """Construct contrastive pairs using hard-negative mining.

    Args:
        samples: Input dataset
        embeddings: Precomputed embeddings (normalized)
        config: Hard-negative configuration

    Returns:
        List of Pair objects
    """
    num_samples: int = len(samples)
    pairs: List[Pair] = []

    # Cosine similarity matrix (since embeddings are normalized)
    sim_matrix: np.ndarray = embeddings @ embeddings.T

    for i in range(num_samples):
        anchor: Sample = samples[i]
        anchor_label: int = anchor.label

        # Positive pool
        pos_indices: List[int] = [
            j for j in range(num_samples) if samples[j].label == anchor_label and j != i
        ]

        # Negative pool
        neg_indices: List[int] = [
            j for j in range(num_samples) if samples[j].label != anchor_label
        ]

        if not pos_indices or not neg_indices:
            continue

        # Select one positive at random for better pair diversity.
        pos_j: int = random.choice(pos_indices)
        pairs.append(Pair(anchor.text, samples[pos_j].text, 1))

        # Sort negatives by similarity (descending)
        neg_sorted: List[int] = sorted(
            neg_indices,
            key=lambda j: float(sim_matrix[i, j]),
            reverse=True,
        )

        # Hard negatives (most similar wrong-label examples)
        hard_negatives: List[int] = neg_sorted[: config.k_hard]

        for j in hard_negatives:
            pairs.append(Pair(anchor.text, samples[j].text, 0))

        # Optional easy negatives (random tail)
        if config.k_easy > 0:
            easy_candidates: List[int] = neg_sorted[-config.k_easy :]
            for j in easy_candidates:
                pairs.append(Pair(anchor.text, samples[j].text, 0))

    return pairs


def _demo() -> None:
    """Simple demo for hard negative mining."""
    from data_loader import load_hf_dataset, few_shot_train_test_setup

    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")

    samples = load_hf_dataset("ag_news", split="train")
    train, _ = few_shot_train_test_setup(samples, k_per_class=8)

    embeddings = encode_samples(model, train)

    config = HardNegativeConfig(k_hard=2, k_easy=1)
    pairs = build_hard_negative_pairs(train, embeddings, config)

    print(f"Generated {len(pairs)} hard-negative pairs")
    for p in pairs[:5]:
        print(p)


if __name__ == "__main__":
    _demo()
