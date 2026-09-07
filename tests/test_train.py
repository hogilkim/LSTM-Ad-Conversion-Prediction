"""Fast tests for LSTM training, model selection, and evaluation."""

import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import taobao.train as training
from taobao.model import ConversionLSTM
from taobao.train import (
    EvaluationMetrics,
    ModelConfig,
    TrainingConfig,
    evaluate,
    run_training,
    train_one_epoch,
)


class DictDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self.cats = torch.tensor([[1, 0], [2, 0], [3, 0], [4, 0]])
        self.behs = torch.tensor([[1, 0], [1, 0], [2, 0], [2, 0]])
        self.hours = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "cats": self.cats[index],
            "behs": self.behs[index],
            "hours": self.hours[index],
            "lengths": torch.tensor(1),
            "labels": self.labels[index],
        }


class FirstHourModel(nn.Module):
    def forward(self, cats, behs, hours, lengths):  # noqa: ANN001
        return hours[:, 0]


def test_evaluate_calculates_auc_and_log_loss_and_restores_mode() -> None:
    model = FirstHourModel()
    model.train()
    loader = DataLoader(DictDataset(), batch_size=2, shuffle=False)

    metrics = evaluate(model, loader, torch.device("cpu"))

    assert metrics.auc == 1.0
    probabilities = torch.sigmoid(torch.tensor([-2.0, -1.0, 1.0, 2.0])).numpy()
    expected = -np.mean(
        np.log([1 - probabilities[0], 1 - probabilities[1], probabilities[2], probabilities[3]])
    )
    assert metrics.log_loss == pytest.approx(expected)
    assert model.training


def test_train_one_epoch_updates_the_model() -> None:
    torch.manual_seed(0)
    model = ConversionLSTM(
        num_categories=6,
        num_behaviors=3,
        category_embedding_dim=2,
        behavior_embedding_dim=2,
        hidden_dim=3,
    )
    loader = DataLoader(DictDataset(), batch_size=4, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    original = model.classifier.weight.detach().clone()

    loss = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))

    assert np.isfinite(loss)
    assert not torch.equal(original, model.classifier.weight)


def _write_split(tensors_dir, split: str) -> None:  # noqa: ANN001
    arrays = {
        "cats": np.array([[1, 2, 0], [2, 3, 0], [3, 4, 1], [4, 1, 2]], dtype=np.int64),
        "behs": np.array([[1, 2, 0], [2, 1, 0], [1, 2, 1], [2, 1, 2]], dtype=np.int64),
        "hours": np.array(
            [[0.8, 0.2, 0.0], [0.7, 0.1, 0.0], [0.9, 0.5, 0.1], [0.6, 0.3, 0.1]],
            dtype=np.float32,
        ),
        "lengths": np.array([2, 2, 3, 3], dtype=np.int64),
        "labels": np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    }
    for name, array in arrays.items():
        np.save(tensors_dir / f"{split}_{name}.npy", array)


def test_run_training_stops_on_validation_auc_and_tests_once(tmp_path, monkeypatch) -> None:
    for split in ("train", "val", "test"):
        _write_split(tmp_path, split)
    vocab_sizes_path = tmp_path / "vocab_sizes.json"
    vocab_sizes_path.write_text(json.dumps({"cats": 6, "behaviors": 3, "pad_index": 0}))
    checkpoint_path = tmp_path / "best.pt"
    config = TrainingConfig(
        tensors_dir=tmp_path,
        vocab_sizes_path=vocab_sizes_path,
        checkpoint_path=checkpoint_path,
        batch_size=2,
        max_epochs=15,
        patience=3,
    )
    model_config = ModelConfig(
        num_categories=6,
        num_behaviors=3,
        category_embedding_dim=2,
        behavior_embedding_dim=2,
        hidden_dim=3,
    )

    loader_settings = []
    real_make_data_loader = training.make_data_loader

    def recording_loader(dataset, batch_size, *, shuffle, device, seed, num_workers):  # noqa: ANN001
        loader_settings.append((dataset.split, shuffle))
        return real_make_data_loader(
            dataset,
            batch_size,
            shuffle=shuffle,
            device=device,
            seed=seed,
            num_workers=num_workers,
        )

    validation_metrics = iter(
        [
            EvaluationMetrics(0.80, 0.50),
            EvaluationMetrics(0.79, 0.49),
            EvaluationMetrics(0.78, 0.48),
            EvaluationMetrics(0.77, 0.47),
        ]
    )
    evaluation_calls = []

    def fixed_evaluate(model, data_loader, device):  # noqa: ANN001
        split = data_loader.dataset.split
        evaluation_calls.append(split)
        if split == "val":
            return next(validation_metrics)
        return EvaluationMetrics(0.70, 0.60)

    monkeypatch.setattr(training, "make_data_loader", recording_loader)
    monkeypatch.setattr(training, "train_one_epoch", lambda *args: 0.4)
    monkeypatch.setattr(training, "evaluate", fixed_evaluate)

    result = run_training(config, model_config)

    assert result.best_epoch == 1
    assert len(result.history) == 4
    assert evaluation_calls == ["val", "val", "val", "val", "test"]
    assert loader_settings == [("train", True), ("val", False), ("test", False)]
    assert result.test == EvaluationMetrics(0.70, 0.60)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["validation_metrics"]["auc"] == 0.80
    assert checkpoint["model_config"] == {
        "num_categories": 6,
        "num_behaviors": 3,
        "category_embedding_dim": 2,
        "behavior_embedding_dim": 2,
        "hidden_dim": 3,
    }
    assert checkpoint["vocab_sizes"] == {"cats": 6, "behaviors": 3, "pad_index": 0}
