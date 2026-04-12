"""Data loading and reproducible few-shot sampling utilities.

This module is intentionally lightweight and generic so we can:
1) reproduce the vanilla SetFit baseline first,
2) save the exact sampled splits,
3) reuse the same splits later for hard-negative mining experiments.

Expected input schema after normalization:
    {
        "text": str,
        "label": int,
        "label_name": Optional[str]
    }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Mapping

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
from config_utils import load_config


# -----------------------------
# Data containers
# -----------------------------


@dataclass
class Sample:
    text: str
    label: int
    label_name: Optional[str] = None


@dataclass
class SplitMetadata:
    dataset_name: str
    seed: int
    k_per_class: int
    train_size: int
    test_size: int
    labels: List[int]


# -----------------------------
# Reproducibility helpers
# -----------------------------


def set_global_seed(seed: int) -> None:
    """Set Python and NumPy seeds for reproducible sampling."""
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------
# Loading utilities
# -----------------------------


def _dataset_to_samples(
    ds: Dataset,
    text_column: str,
    label_column: str,
    label_names: Optional[Sequence[str]] = None,
) -> List[Sample]:
    """Convert a Hugging Face Dataset into a list of normalized samples."""
    samples: List[Sample] = []

    for row in ds:
        if not isinstance(row, Mapping):
            raise TypeError(f"Expected Mapping, got {type(row)}")

        text = row[text_column]
        label = int(row[label_column])

        label_name: Optional[str] = None
        if label_names is not None and 0 <= label < len(label_names):
            label_name = str(label_names[label])

        samples.append(Sample(text=text, label=label, label_name=label_name))

    return samples


def load_hf_dataset(
    dataset_name: str,
    split: str,
    text_column: str = "text",
    label_column: str = "label",
    label_names: Optional[Sequence[str]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> List[Sample]:
    """Load a Hugging Face dataset split and normalize it into Sample objects."""
    ds = load_dataset(
        dataset_name, split=split, cache_dir=str(cache_dir) if cache_dir else None
    )
    return _dataset_to_samples(
        ds, text_column=text_column, label_column=label_column, label_names=label_names
    )


def load_local_jsonl(
    path: Union[str, Path],
    text_key: str = "text",
    label_key: str = "label",
    label_name_key: Optional[str] = None,
) -> List[Sample]:
    """Load a local JSONL file into Sample objects."""
    path = Path(path)
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(
                Sample(
                    text=str(obj[text_key]),
                    label=int(obj[label_key]),
                    label_name=(
                        str(obj[label_name_key])
                        if label_name_key and label_name_key in obj
                        else None
                    ),
                )
            )
    return samples


# -----------------------------
# Sampling utilities
# -----------------------------


def stratified_train_test_split(
    samples: Sequence[Sample],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Stratified train/test split for a list of samples."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")

    y = np.array([s.label for s in samples])
    idx = np.arange(len(samples))

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(idx, y))

    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, test_samples


def few_shot_sample_per_class(
    samples: Sequence[Sample],
    k_per_class: int,
    seed: int = 42,
) -> List[Sample]:
    """Sample exactly k examples per class, stratified by label.

    This is the key function for reproducing the SetFit baseline before any
    hard-negative mining is added.
    """
    if k_per_class <= 0:
        raise ValueError("k_per_class must be positive")

    set_global_seed(seed)
    by_label: Dict[int, List[Sample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    chosen: List[Sample] = []
    for label, group in sorted(by_label.items(), key=lambda x: x[0]):
        if len(group) < k_per_class:
            raise ValueError(
                f"Not enough samples for label {label}: have {len(group)}, need {k_per_class}."
            )
        chosen.extend(random.sample(group, k_per_class))

    random.shuffle(chosen)
    return chosen


def few_shot_train_test_setup(
    samples: Sequence[Sample],
    k_per_class: int,
    seed: int = 42,
    test_size: float = 0.2,
) -> Tuple[List[Sample], List[Sample]]:
    """Convenience wrapper: stratified split first, then few-shot sample train set.

    Recommended workflow:
    1. Split full dataset into train/test once per seed.
    2. Sample k examples per class from train only.
    3. Keep test fixed for fair comparisons.
    """
    train_full, test = stratified_train_test_split(
        samples, test_size=test_size, seed=seed
    )
    train_few_shot = few_shot_sample_per_class(
        train_full, k_per_class=k_per_class, seed=seed
    )
    return train_few_shot, test


# -----------------------------
# Serialization helpers
# -----------------------------


def samples_to_dicts(samples: Sequence[Sample]) -> List[dict]:
    return [asdict(s) for s in samples]


def save_json(path: Union[str, Path], obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_samples_jsonl(path: Union[str, Path], samples: Sequence[Sample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


def save_samples_txt(
    path: Union[str, Path],
    samples: Sequence[Sample],
) -> None:
    """Save samples as readable .txt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            label_name = s.label_name if s.label_name else str(s.label)
            f.write(f"[{label_name}] {s.text}\n\n")


def build_split_metadata(
    dataset_name: str,
    seed: int,
    k_per_class: int,
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
) -> SplitMetadata:
    labels = sorted({s.label for s in train_samples} | {s.label for s in test_samples})
    return SplitMetadata(
        dataset_name=dataset_name,
        seed=seed,
        k_per_class=k_per_class,
        train_size=len(train_samples),
        test_size=len(test_samples),
        labels=labels,
    )


# -----------------------------
# Simple CLI for sanity checks
# -----------------------------


def _demo() -> None:
    parser = argparse.ArgumentParser(description="Load and sample a few-shot dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--k_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    loader_cfg = config.get("data_loader", {})

    dataset_name = str(dataset_cfg.get("name", "SetFit/bbc-news"))
    text_column = str(dataset_cfg.get("text_column", "text"))
    label_column = str(dataset_cfg.get("label_column", "label"))
    labels_raw = dataset_cfg.get("label_names")
    label_names: Optional[List[str]] = None
    if isinstance(labels_raw, list):
        label_names = [str(v) for v in labels_raw]

    k_per_class = (
        args.k_per_class
        if args.k_per_class is not None
        else int(loader_cfg.get("k_per_class", 8))
    )
    seed = args.seed if args.seed is not None else int(loader_cfg.get("seed", 42))
    test_size = (
        args.test_size
        if args.test_size is not None
        else float(loader_cfg.get("test_size", 0.2))
    )

    full_samples = load_hf_dataset(
        dataset_name=dataset_name,
        split="train",
        text_column=text_column,
        label_column=label_column,
        label_names=label_names,
    )

    train_fs, test = few_shot_train_test_setup(
        full_samples,
        k_per_class=k_per_class,
        seed=seed,
        test_size=test_size,
    )

    save_samples_txt("data/train.txt", train_fs)
    save_samples_txt("data/test.txt", test)

    meta = build_split_metadata(
        dataset_name=dataset_name,
        seed=seed,
        k_per_class=k_per_class,
        train_samples=train_fs,
        test_samples=test,
    )

    print(meta)
    print(f"Few-shot train size: {len(train_fs)}")
    print(f"Test size: {len(test)}")
    print("Sample preview:")
    for s in train_fs[:3]:
        print(s)


if __name__ == "__main__":
    _demo()
