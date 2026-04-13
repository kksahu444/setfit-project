"""Training script for baseline SetFit-style pipeline using sentence-transformers.

Pipeline:
1) Load data and create reproducible few-shot split
2) Build contrastive pairs (baseline: random sampling)
3) Train encoder with CosineSimilarityLoss
4) Train Logistic Regression head on embeddings
5) Evaluate on test set
6) Save metrics (JSON) and model artifacts

"""

from __future__ import annotations

import os
import json
import importlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import argparse

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
)

try:
    _loss_module = importlib.import_module("sentence_transformers.losses")
except ModuleNotFoundError:
    _loss_module = importlib.import_module("sentence_transformers.sentence_transformer.losses")

CosineSimilarityLoss = getattr(_loss_module, "CosineSimilarityLoss")
from hard_negative import build_hard_negative_pairs, encode_samples, HardNegativeConfig
from config_utils import load_config

from data_loader import (
    Sample,
    build_split_metadata,
    few_shot_train_test_setup,
    load_hf_dataset,
    save_json,
    save_samples_jsonl,
)
from pair_builder import Pair, build_sampled_pairs


class PairDataset(Dataset[InputExample]):
    """Dataset wrapper for InputExample list."""

    def __init__(self, data: List[InputExample]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> InputExample:
        return self.data[idx]


def _pairs_to_input_examples(pairs: Sequence[Pair]) -> List[InputExample]:
    """Convert Pair objects to sentence-transformers InputExample."""
    examples: List[InputExample] = []
    for p in pairs:
        # label must be float for CosineSimilarityLoss
        examples.append(InputExample(texts=[p.text_a, p.text_b], label=float(p.label)))
    return examples


def _encode_texts(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    """Encode texts into normalized embeddings."""
    embeddings: np.ndarray = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


def _train_encoder(
    model: SentenceTransformer,
    pairs: Sequence[Pair],
    batch_size: int,
    epochs: int,
) -> None:
    """Train encoder with cosine similarity loss on provided pairs."""
    train_examples: List[InputExample] = _pairs_to_input_examples(pairs)
    train_dataset = PairDataset(train_examples)
    train_loader: DataLoader[InputExample] = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )

    loss = CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        show_progress_bar=True,
    )


def _train_head(
    model: SentenceTransformer,
    train_samples: Sequence[Sample],
    lr_max_iter: int,
) -> LogisticRegression:
    """Train Logistic Regression classifier on embeddings."""
    texts: List[str] = [s.text for s in train_samples]
    labels: List[int] = [s.label for s in train_samples]

    X: np.ndarray = _encode_texts(model, texts)

    clf = LogisticRegression(max_iter=lr_max_iter)
    clf.fit(X, labels)
    return clf


def _evaluate(
    model: SentenceTransformer,
    clf: LogisticRegression,
    test_samples: Sequence[Sample],
) -> Dict[str, Any]:
    """Evaluate classifier on test set."""
    texts: List[str] = [s.text for s in test_samples]
    labels: List[int] = [s.label for s in test_samples]

    X_test: np.ndarray = _encode_texts(model, texts)
    preds: np.ndarray = clf.predict(X_test)

    acc: float = float(accuracy_score(labels, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    label_order: List[int] = sorted(set(labels))
    per_class_precision_raw, per_class_recall_raw, per_class_f1_raw, per_class_support_raw = (
        precision_recall_fscore_support(
            labels,
            preds,
            labels=label_order,
            average=None,
            zero_division=0,
        )
    )

    per_class_precision = np.asarray(per_class_precision_raw, dtype=float)
    per_class_recall = np.asarray(per_class_recall_raw, dtype=float)
    per_class_f1 = np.asarray(per_class_f1_raw, dtype=float)
    per_class_support = np.asarray(per_class_support_raw, dtype=float)

    per_class: Dict[str, Dict[str, float]] = {}
    for i, label_id in enumerate(label_order):
        per_class[str(label_id)] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": float(per_class_support[i]),
        }

    cm = confusion_matrix(labels, preds, labels=label_order)

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "label_order": label_order,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _resolve_pair_strategy(mode: str, pair_strategy: Optional[str]) -> str:
    """Resolve strategy while keeping backward compatibility with old mode-only flow."""
    if pair_strategy:
        return pair_strategy

    if mode == "baseline":
        return "random"
    if mode == "hard_negative":
        return "hard"

    raise ValueError(f"Unsupported mode: {mode}")


def run_single_seed(
    dataset_name: str,
    text_column: str,
    label_column: str,
    label_names: Optional[Sequence[str]],
    seed: int,
    k_per_class: int,
    test_size: float,
    output_dir: Path,
    mode: str,
    pair_strategy: Optional[str],
    base_model_name: str,
    batch_size: int,
    epochs: int,
    lr_max_iter: int,
    num_pos_per_anchor: int,
    num_neg_per_anchor: int,
    hard_negative_config: HardNegativeConfig,
) -> Dict[str, Any]:
    """Run one seed: data -> pairs -> train -> eval -> save artifacts."""

    print(f"\n========== Running seed {seed} ==========")

    # 1) Load full dataset (train split only; we create our own split)
    full_samples: List[Sample] = load_hf_dataset(
        dataset_name,
        split="train",
        text_column=text_column,
        label_column=label_column,
        label_names=label_names,
    )

    # 2) Few-shot setup (stratified split + k per class sampling)
    train_fs, test = few_shot_train_test_setup(
        full_samples,
        k_per_class=k_per_class,
        seed=seed,
        test_size=test_size,
    )

    # 3) Initialize encoder
    model = SentenceTransformer(base_model_name)

    # 4) Build contrastive pairs using selected strategy
    strategy = _resolve_pair_strategy(mode=mode, pair_strategy=pair_strategy)

    if strategy == "random":
        pairs: List[Pair] = build_sampled_pairs(
            train_fs,
            num_pos_per_anchor=num_pos_per_anchor,
            num_neg_per_anchor=num_neg_per_anchor,
            seed=seed,
        )

    elif strategy in {"easy", "hard", "mixed"}:
        print(f"[Seed {seed}] Generating embeddings...")

        embeddings = encode_samples(model, train_fs)

        print(f"[Seed {seed}] Building {strategy} pairs...")

        if strategy == "easy":
            mining_config = HardNegativeConfig(k_hard=0, k_easy=max(1, num_neg_per_anchor))
        elif strategy == "hard":
            mining_config = HardNegativeConfig(k_hard=max(1, hard_negative_config.k_hard), k_easy=0)
        else:
            mining_config = HardNegativeConfig(
                k_hard=max(1, hard_negative_config.k_hard),
                k_easy=max(1, hard_negative_config.k_easy),
            )

        pairs = build_hard_negative_pairs(
            train_fs,
            embeddings,
            mining_config,
            num_pos_per_anchor=num_pos_per_anchor,
            seed=seed,
        )

        print(f"[Seed {seed}] Generated {len(pairs)} pairs")

    else:
        raise ValueError(f"Unsupported pair strategy: {strategy}")

    # 5) Train encoder
    _train_encoder(model, pairs, batch_size=batch_size, epochs=epochs)

    # 6) Train classifier head
    clf: LogisticRegression = _train_head(
        model,
        train_fs,
        lr_max_iter=lr_max_iter,
    )

    # 7) Evaluate
    metrics: Dict[str, float] = _evaluate(model, clf, test)

    # 8) Save artifacts
    seed_dir: Path = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Save splits for reproducibility
    save_samples_jsonl(seed_dir / "train_fs.jsonl", train_fs)
    save_samples_jsonl(seed_dir / "test.jsonl", test)

    # Save metadata
    meta = build_split_metadata(
        dataset_name=dataset_name,
        seed=seed,
        k_per_class=k_per_class,
        train_samples=train_fs,
        test_samples=test,
    )
    save_json(seed_dir / "metadata.json", asdict(meta))

    # Save metrics
    save_json(seed_dir / "metrics.json", metrics)

    # Save model (encoder)
    model.save(str(seed_dir / "encoder"))

    # Save classifier
    with (seed_dir / "classifier.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "coef": clf.coef_.tolist(),
                "intercept": clf.intercept_.tolist(),
                "classes": clf.classes_.tolist(),
            },
            f,
            indent=2,
        )

    return metrics


def run_experiment(
    dataset_name: str,
    text_column: str,
    label_column: str,
    label_names: Optional[Sequence[str]],
    seeds: Sequence[int],
    k_per_class: int,
    test_size: float,
    output_dir: Path,
    mode: str,
    pair_strategy: Optional[str],
    base_model_name: str,
    batch_size: int,
    epochs: int,
    lr_max_iter: int,
    num_pos_per_anchor: int,
    num_neg_per_anchor: int,
    hard_negative_config: HardNegativeConfig,
) -> Dict[str, Any]:
    """Run multiple seeds and aggregate mean/std metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[float] = []
    strategy = _resolve_pair_strategy(mode=mode, pair_strategy=pair_strategy)

    for seed in seeds:
        metrics = run_single_seed(
            dataset_name=dataset_name,
            text_column=text_column,
            label_column=label_column,
            label_names=label_names,
            seed=seed,
            k_per_class=k_per_class,
            test_size=test_size,
            output_dir=output_dir,
            mode=mode,
            pair_strategy=strategy,
            base_model_name=base_model_name,
            batch_size=batch_size,
            epochs=epochs,
            lr_max_iter=lr_max_iter,
            num_pos_per_anchor=num_pos_per_anchor,
            num_neg_per_anchor=num_neg_per_anchor,
            hard_negative_config=hard_negative_config,
        )
        all_metrics.append(metrics["accuracy"])

    mean_acc: float = float(np.mean(all_metrics))
    std_acc: float = float(np.std(all_metrics))

    summary: Dict[str, Any] = {
        "accuracy_mean": mean_acc,
        "accuracy_std": std_acc,
        "pair_strategy": strategy,
        "mode": mode,
        "k_per_class": k_per_class,
    }

    save_json(output_dir / "summary.json", summary)
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train baseline SetFit-style model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument("--k_per_class", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--test_size", type=float, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "hard_negative"],
        default=None,
    )
    parser.add_argument(
        "--pair_strategy",
        type=str,
        choices=["random", "easy", "hard", "mixed"],
        default=None,
        help="Pair construction strategy for ablation studies.",
    )
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    config = load_config(args.config)
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("train", {})
    pair_cfg = train_cfg.get("pair_sampling", {})
    hard_negative_cfg = train_cfg.get("hard_negative", {})

    dataset_name = args.dataset_name or str(dataset_cfg.get("name", "SetFit/bbc-news"))
    text_column = str(dataset_cfg.get("text_column", "text"))
    label_column = str(dataset_cfg.get("label_column", "label"))

    labels_raw = dataset_cfg.get("label_names")
    label_names: Optional[List[str]] = None
    if isinstance(labels_raw, list):
        label_names = [str(v) for v in labels_raw]

    mode = args.mode or str(train_cfg.get("mode", "baseline"))
    pair_strategy = args.pair_strategy or train_cfg.get("pair_strategy")
    k_per_class = (
        args.k_per_class
        if args.k_per_class is not None
        else int(train_cfg.get("k_per_class", 8))
    )
    test_size = (
        args.test_size
        if args.test_size is not None
        else float(train_cfg.get("test_size", 0.2))
    )

    seeds = args.seeds
    if seeds is None:
        cfg_seeds = train_cfg.get("seeds", [0, 1, 2, 3])
        if not isinstance(cfg_seeds, list):
            raise ValueError("train.seeds must be a list in config.")
        seeds = [int(seed) for seed in cfg_seeds]

    base_model_name = str(
        train_cfg.get(
            "base_model_name", "sentence-transformers/paraphrase-mpnet-base-v2"
        )
    )
    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 1))
    lr_max_iter = int(train_cfg.get("lr_max_iter", 1000))
    num_pos_per_anchor = int(pair_cfg.get("num_pos_per_anchor", 1))
    num_neg_per_anchor = int(pair_cfg.get("num_neg_per_anchor", 1))

    hard_negative_config = HardNegativeConfig(
        k_hard=int(hard_negative_cfg.get("k_hard", 2)),
        k_easy=int(hard_negative_cfg.get("k_easy", 1)),
    )

    run_label = pair_strategy if pair_strategy else mode
    output_path = Path(args.output_dir) if args.output_dir else (Path("results") / str(run_label))
    summary = run_experiment(
        dataset_name=dataset_name,
        text_column=text_column,
        label_column=label_column,
        label_names=label_names,
        seeds=seeds,
        k_per_class=k_per_class,
        test_size=test_size,
        output_dir=output_path,
        mode=mode,
        pair_strategy=str(pair_strategy) if pair_strategy else None,
        base_model_name=base_model_name,
        batch_size=batch_size,
        epochs=epochs,
        lr_max_iter=lr_max_iter,
        num_pos_per_anchor=num_pos_per_anchor,
        num_neg_per_anchor=num_neg_per_anchor,
        hard_negative_config=hard_negative_config,
    )

    print("Summary:")
    print(summary)


if __name__ == "__main__":
    _cli()
