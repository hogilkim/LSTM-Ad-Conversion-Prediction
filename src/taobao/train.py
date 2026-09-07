"""Reproducible training and evaluation for :class:`taobao.model.ConversionLSTM`.

Usage from the repository root, after creating the tensor artifacts::

    uv run python -m taobao.train

The test split is not opened until training and validation-based checkpoint selection
have finished. It is then evaluated exactly once using the restored best checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from taobao.data.dataset import TensorSplitDataset
from taobao.data.tensors import DEFAULT_OUT_DIR
from taobao.model import ConversionLSTM

DEFAULT_VOCAB_SIZES = Path("data/processed/vocab_sizes.json")
DEFAULT_CHECKPOINT = Path("models/conversion_lstm_best.pt")
MAX_EPOCHS = 15


@dataclass(frozen=True)
class ModelConfig:
    """Arguments needed to reconstruct a :class:`ConversionLSTM`."""

    num_categories: int
    num_behaviors: int
    category_embedding_dim: int = 32
    behavior_embedding_dim: int = 4
    hidden_dim: int = 64


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings and artifact locations."""

    tensors_dir: Path = DEFAULT_OUT_DIR
    vocab_sizes_path: Path = DEFAULT_VOCAB_SIZES
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    seed: int = 42
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = MAX_EPOCHS
    patience: int = 3
    device: str = "auto"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.max_epochs <= MAX_EPOCHS:
            raise ValueError(f"max_epochs must be between 1 and {MAX_EPOCHS}")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")


@dataclass(frozen=True)
class EvaluationMetrics:
    auc: float
    log_loss: float


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_auc: float
    validation_log_loss: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingResult:
    history: list[EpochMetrics]
    best_epoch: int
    best_validation: EvaluationMetrics
    test: EvaluationMetrics
    elapsed_seconds: float
    checkpoint_path: Path
    device: str


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA, then MPS, then CPU, or validate an explicit device request."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(requested)


def load_vocab_sizes(path: Path) -> dict[str, Any]:
    """Load and validate the vocabulary metadata used to size embeddings."""
    with path.open() as f:
        vocab_sizes = json.load(f)
    for name in ("cats", "behaviors"):
        if not isinstance(vocab_sizes.get(name), int) or vocab_sizes[name] < 1:
            raise ValueError(f"{path} must contain a positive integer {name!r} size")
    return vocab_sizes


def make_data_loader(
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int,
    *,
    shuffle: bool,
    device: torch.device,
    seed: int,
    num_workers: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a loader, using a seeded generator only for shuffled training data."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device, non_blocking=device.type == "cuda")
        for name, tensor in batch.items()
    }


def train_one_epoch(
    model: ConversionLSTM,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: Adam,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
) -> float:
    """Train for one pass and return mean binary cross-entropy."""
    model.train()
    loss_sum = 0.0
    n_examples = 0
    for cpu_batch in data_loader:
        batch = _move_batch(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["cats"], batch["behs"], batch["hours"], batch["lengths"])
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        batch_size = batch["labels"].numel()
        loss_sum += loss.item() * batch_size
        n_examples += batch_size
    if not n_examples:
        raise ValueError("cannot train on an empty dataset")
    return loss_sum / n_examples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> EvaluationMetrics:
    """Calculate AUC and probabilistic log loss for one complete split."""
    was_training = model.training
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for cpu_batch in data_loader:
            batch = _move_batch(cpu_batch, device)
            logits = model(
                batch["cats"], batch["behs"], batch["hours"], batch["lengths"]
            )
            labels.append(batch["labels"].detach().cpu().numpy())
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    if was_training:
        model.train()
    if not labels:
        raise ValueError("cannot evaluate an empty dataset")

    all_labels = np.concatenate(labels)
    all_probabilities = np.concatenate(probabilities)
    return EvaluationMetrics(
        auc=float(roc_auc_score(all_labels, all_probabilities)),
        log_loss=float(log_loss(all_labels, all_probabilities, labels=[0.0, 1.0])),
    )


def _training_config_for_checkpoint(config: TrainingConfig) -> dict[str, Any]:
    saved = asdict(config)
    for name in ("tensors_dir", "vocab_sizes_path", "checkpoint_path"):
        saved[name] = str(saved[name])
    return saved


def save_checkpoint(
    path: Path,
    model: ConversionLSTM,
    optimizer: Adam,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    vocab_sizes: dict[str, Any],
    history: list[EpochMetrics],
    validation: EvaluationMetrics,
    selected_device: torch.device,
) -> None:
    """Atomically save the current best model and everything needed to rebuild it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": history[-1].epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "training_config": _training_config_for_checkpoint(training_config),
        "vocab_sizes": vocab_sizes,
        "validation_metrics": asdict(validation),
        "history": [asdict(metrics) for metrics in history],
        "selected_device": str(selected_device),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint using PyTorch's restricted weights-only unpickler."""
    return torch.load(path, map_location=device, weights_only=True)


def run_training(
    config: TrainingConfig,
    model_config: ModelConfig | None = None,
) -> TrainingResult:
    """Train on train/validation, restore the best model, then test exactly once."""
    started = time.perf_counter()
    set_seed(config.seed)
    device = select_device(config.device)
    vocab_sizes = load_vocab_sizes(config.vocab_sizes_path)
    if model_config is None:
        model_config = ModelConfig(
            num_categories=vocab_sizes["cats"],
            num_behaviors=vocab_sizes["behaviors"],
        )
    elif (
        model_config.num_categories != vocab_sizes["cats"]
        or model_config.num_behaviors != vocab_sizes["behaviors"]
    ):
        raise ValueError("model category and behavior sizes must match the vocabulary metadata")

    train_dataset = TensorSplitDataset("train", config.tensors_dir)
    validation_dataset = TensorSplitDataset("val", config.tensors_dir)
    train_loader = make_data_loader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        device=device,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    validation_loader = make_data_loader(
        validation_dataset,
        config.batch_size,
        shuffle=False,
        device=device,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    model = ConversionLSTM(**asdict(model_config)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    history: list[EpochMetrics] = []
    best_epoch = 0
    best_validation = EvaluationMetrics(auc=float("-inf"), log_loss=float("inf"))
    epochs_without_improvement = 0

    print(
        f"device={device} seed={config.seed} batch_size={config.batch_size} "
        f"train={len(train_dataset):,} val={len(validation_dataset):,}",
        flush=True,
    )
    print(f"checkpoint={config.checkpoint_path}", flush=True)

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation = evaluate(model, validation_loader, device)
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            validation_auc=validation.auc,
            validation_log_loss=validation.log_loss,
            elapsed_seconds=time.perf_counter() - epoch_started,
        )
        history.append(epoch_metrics)

        improved = validation.auc > best_validation.auc
        if improved:
            best_epoch = epoch
            best_validation = validation
            epochs_without_improvement = 0
            save_checkpoint(
                config.checkpoint_path,
                model,
                optimizer,
                model_config,
                config,
                vocab_sizes,
                history,
                validation,
                device,
            )
        else:
            epochs_without_improvement += 1

        marker = " best" if improved else ""
        print(
            f"epoch={epoch:02d}/{config.max_epochs:02d} "
            f"train_loss={train_loss:.6f} val_auc={validation.auc:.6f} "
            f"val_log_loss={validation.log_loss:.6f} "
            f"seconds={epoch_metrics.elapsed_seconds:.1f}{marker}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            print(
                f"early_stopping epoch={epoch} patience={config.patience}",
                flush=True,
            )
            break

    # The test artifact is deliberately not opened until all model selection is over.
    checkpoint = load_checkpoint(config.checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_dataset = TensorSplitDataset("test", config.tensors_dir)
    test_loader = make_data_loader(
        test_dataset,
        config.batch_size,
        shuffle=False,
        device=device,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    test_metrics = evaluate(model, test_loader, device)
    elapsed_seconds = time.perf_counter() - started

    print(
        f"best_epoch={best_epoch} best_val_auc={best_validation.auc:.6f} "
        f"best_val_log_loss={best_validation.log_loss:.6f}",
        flush=True,
    )
    print(
        f"test_auc={test_metrics.auc:.6f} test_log_loss={test_metrics.log_loss:.6f}",
        flush=True,
    )
    print(
        f"elapsed_seconds={elapsed_seconds:.1f} checkpoint={config.checkpoint_path}",
        flush=True,
    )
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation=best_validation,
        test=test_metrics,
        elapsed_seconds=elapsed_seconds,
        checkpoint_path=config.checkpoint_path,
        device=str(device),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vocab-sizes", type=Path, default=DEFAULT_VOCAB_SIZES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--category-embedding-dim", type=int, default=32)
    parser.add_argument("--behavior-embedding-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        tensors_dir=args.tensors_dir,
        vocab_sizes_path=args.vocab_sizes,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        num_workers=args.num_workers,
    )
    vocab_sizes = load_vocab_sizes(config.vocab_sizes_path)
    model_config = ModelConfig(
        num_categories=vocab_sizes["cats"],
        num_behaviors=vocab_sizes["behaviors"],
        category_embedding_dim=args.category_embedding_dim,
        behavior_embedding_dim=args.behavior_embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    run_training(config, model_config)


if __name__ == "__main__":
    main()
