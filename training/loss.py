import torch
import torch.nn as nn


class LanguageModelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        if logits.ndim != 3:
            raise ValueError(
                "Logits must have shape [batch_size, seq_len, vocab_size]."
            )

        if targets.ndim != 2:
            raise ValueError(
                "Targets must have shape [batch_size, seq_len]."
            )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "Logits and targets must have matching batch and sequence dimensions."
            )

        batch_size, seq_len, vocab_size = logits.shape

        logits = logits.reshape(
            batch_size * seq_len,
            vocab_size,
        )

        targets = targets.reshape(
            batch_size * seq_len,
        )

        return self.loss(logits, targets)