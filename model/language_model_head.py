
import torch
import torch.nn as nn


class LanguageModelHead(nn.Module):
    """
    Projects Transformer representations into vocabulary logits.
    """

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.projection = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape [batch_size, sequence_length, embedding_dim]"
            )

        if x.shape[1] == 0:
            raise ValueError("sequence_length must be positive")

        if x.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {x.shape[2]}"
            )

        return self.projection(x)
