
import torch
from torch import nn

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """
    A single Pre-LayerNorm Transformer block.

    The block consists of:
    1. LayerNorm
    2. Multi-Head Self-Attention
    3. Residual Connection
    4. LayerNorm
    5. Feed-Forward Network
    6. Residual Connection
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        max_seq_len: int,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            max_seq_len,
        )

        self.norm2 = LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embedding_dim].

        Returns:
            Tensor of shape [batch_size, seq_len, embedding_dim].
        """

        if x.ndim != 3:
            raise ValueError(
                "Input must have shape [batch_size, seq_len, embedding_dim]"
            )

        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {x.shape[-1]}"
            )

        if x.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds "
                f"maximum sequence length {self.max_seq_len}"
            )

        # First sub-layer: Multi-Head Self-Attention
        normalized_x = self.norm1(x)
        attention_output = self.attention(normalized_x)
        x = x + attention_output

        # Second sub-layer: Feed-Forward Network
        normalized_x = self.norm2(x)
        feed_forward_output = self.feed_forward(normalized_x)
        x = x + feed_forward_output

        return x
