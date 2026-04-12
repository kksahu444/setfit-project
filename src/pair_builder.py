"""Pair construction utilities for SetFit-style contrastive training.

This module provides:
1. Baseline random pair construction (reproducing SetFit behavior)
2. Structured interfaces to extend into hard-negative mining later

All functions are written with strict typing and are compatible with mypy and ruff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import random

from data_loader import Sample


@dataclass(frozen=True)
class Pair:
    """Represents a contrastive pair.

    Attributes:
        text_a: First sentence
        text_b: Second sentence
        label: 1 for positive pair, 0 for negative pair
    """

    text_a: str
    text_b: str
    label: int


def build_all_pairs(samples: Sequence[Sample]) -> List[Pair]:
    """Construct all possible pairs (quadratic) from dataset.

    Args:
        samples: Input dataset

    Returns:
        List of Pair objects
    """
    pairs: List[Pair] = []

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            label: int = int(samples[i].label == samples[j].label)

            pairs.append(
                Pair(
                    text_a=samples[i].text,
                    text_b=samples[j].text,
                    label=label,
                )
            )

    return pairs


def build_sampled_pairs(
    samples: Sequence[Sample],
    num_pos_per_anchor: int = 1,
    num_neg_per_anchor: int = 1,
    seed: int = 42,
) -> List[Pair]:
    """Construct sampled pairs per anchor (more efficient than O(N^2)).

    This mimics practical SetFit training behavior where we limit
    number of pairs instead of using all combinations.

    Args:
        samples: Input dataset
        num_pos_per_anchor: Number of positive pairs per anchor
        num_neg_per_anchor: Number of negative pairs per anchor
        seed: Random seed

    Returns:
        List of Pair objects
    """
    random.seed(seed)

    pairs: List[Pair] = []

    for i, anchor in enumerate(samples):
        pos_pool: List[int] = [
            j for j, s in enumerate(samples) if s.label == anchor.label and j != i
        ]

        neg_pool: List[int] = [
            j for j, s in enumerate(samples) if s.label != anchor.label
        ]

        # Positive pairs
        for _ in range(min(num_pos_per_anchor, len(pos_pool))):
            j: int = random.choice(pos_pool)
            pairs.append(Pair(anchor.text, samples[j].text, 1))

        # Negative pairs
        for _ in range(min(num_neg_per_anchor, len(neg_pool))):
            j = random.choice(neg_pool)
            pairs.append(Pair(anchor.text, samples[j].text, 0))

    return pairs


def pairs_to_tuples(pairs: Sequence[Pair]) -> List[Tuple[str, str, int]]:
    """Convert Pair objects into tuple format.

    Useful for downstream training pipelines.

    Args:
        pairs: List of Pair objects

    Returns:
        List of (text_a, text_b, label)
    """
    return [(p.text_a, p.text_b, p.label) for p in pairs]


def _demo() -> None:
    """Simple demo to validate pair construction."""
    from data_loader import load_hf_dataset, few_shot_train_test_setup

    samples = load_hf_dataset("ag_news", split="train")
    train, _ = few_shot_train_test_setup(samples, k_per_class=8)

    pairs = build_sampled_pairs(train)

    print(f"Generated {len(pairs)} pairs")
    for p in pairs[:5]:
        print(p)


if __name__ == "__main__":
    _demo()
