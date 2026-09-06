"""LSTM model for user-level conversion prediction."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ConversionLSTM(nn.Module):
    """Predict one conversion logit from a padded behavior sequence."""

    def __init__(
        self,
        num_categories: int,
        num_behaviors: int,
        category_embedding_dim: int = 32,
        behavior_embedding_dim: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(
            num_categories, category_embedding_dim, padding_idx=0
        )
        self.behavior_embedding = nn.Embedding(
            num_behaviors, behavior_embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            category_embedding_dim + behavior_embedding_dim + 1,
            hidden_dim,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        cats: torch.Tensor,
        behs: torch.Tensor,
        hours: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return one raw logit per example.

        ``lengths`` identifies the real prefix of each right-padded sequence. Packing
        prevents the LSTM from updating its state on padded positions, so ``hidden`` is
        the state after each user's final real event rather than the final array slot.
        """
        features = torch.cat(
            (
                self.category_embedding(cats),
                self.behavior_embedding(behs),
                hours.unsqueeze(-1),
            ),
            dim=-1,
        )
        packed = pack_padded_sequence(
            features,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(hidden[-1]).squeeze(-1)
