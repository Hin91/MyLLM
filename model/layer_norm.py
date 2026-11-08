
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Implements Layer Normalization over the last tensor dimension.
    """

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.embedding_dim = embedding_dim
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1:
            raise ValueError("Input must have at least one dimension.")

        if x.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected last dimension {self.embedding_dim}, "
                f"got {x.size(-1)}."
            )

        mean = x.mean(dim=-1, keepdim=True)

        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        normalized = (x - mean) / torch.sqrt(
            variance + self.eps
        )

        return self.gamma * normalized + self.beta
