
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network
    used inside a Transformer block.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.linear2 = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape [batch, sequence, embedding]."
            )

        if x.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {x.size(-1)}."
            )

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x
