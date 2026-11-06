
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Implements single-head scaled dot-product causal self-attention.
    """

    def __init__(self, embedding_dim: int, max_seq_len: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        # Store the causal mask as a non-trainable model buffer.
        causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape
                (batch_size, sequence_length, embedding_dim)

        Returns:
            Tensor of shape
                (batch_size, sequence_length, embedding_dim)
        """

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch_size, sequence_length, embedding_dim)."
            )

        batch_size, sequence_length, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {embedding_dim}."
            )

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"maximum supported length ({self.max_seq_len})."
            )

        # Project input vectors into queries, keys, and values.
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        # Calculate scaled attention scores.
        scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.embedding_dim)

        # Prevent each position from attending to future positions.
        mask = self.causal_mask[:sequence_length, :sequence_length]

        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        # Convert attention scores into probability distributions.
        attention_weights = F.softmax(scores, dim=-1)

        # Combine value vectors using the attention weights.
        attention_output = torch.matmul(
            attention_weights,
            values,
        )

        return self.output_projection(attention_output)
