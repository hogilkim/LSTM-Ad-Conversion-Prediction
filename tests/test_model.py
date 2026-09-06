"""Tests for the conversion LSTM."""

import torch

from taobao.model import ConversionLSTM


def test_conversion_lstm_returns_one_logit_per_example() -> None:
    model = ConversionLSTM(
        num_categories=8,
        num_behaviors=5,
        category_embedding_dim=3,
        behavior_embedding_dim=2,
        hidden_dim=4,
    )
    cats = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]])
    behs = torch.tensor([[1, 2, 0, 0], [1, 2, 3, 4]])
    hours = torch.tensor([[0.8, 0.4, 0.0, 0.0], [1.0, 0.7, 0.3, 0.1]])
    lengths = torch.tensor([2, 4])

    logits = model(cats, behs, hours, lengths)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()

    logits.sum().backward()
    assert model.classifier.weight.grad is not None


def test_conversion_lstm_ignores_positions_after_each_length() -> None:
    torch.manual_seed(0)
    model = ConversionLSTM(
        num_categories=8,
        num_behaviors=5,
        category_embedding_dim=3,
        behavior_embedding_dim=2,
        hidden_dim=4,
    )
    model.eval()
    lengths = torch.tensor([2])

    padded_logit = model(
        cats=torch.tensor([[1, 2, 0, 0]]),
        behs=torch.tensor([[1, 2, 0, 0]]),
        hours=torch.tensor([[0.8, 0.4, 0.0, 0.0]]),
        lengths=lengths,
    )
    changed_padding_logit = model(
        cats=torch.tensor([[1, 2, 6, 7]]),
        behs=torch.tensor([[1, 2, 3, 4]]),
        hours=torch.tensor([[0.8, 0.4, 0.9, 0.2]]),
        lengths=lengths,
    )

    torch.testing.assert_close(padded_logit, changed_padding_logit, rtol=0, atol=0)
