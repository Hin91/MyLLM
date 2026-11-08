import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head causal self-attention.

    The input embedding dimension is divided across multiple
    attention heads. Each head independently computes attention,
    and their outputs are concatenated and projected.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_seq_len: int,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.max_seq_len = max_seq_len

        # Independent learned projections for Query, Key, and Value.
        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Combines information from all attention heads.
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Lower-triangular mask prevents access to future positions.
        causal_mask = torch.tril(
            torch.ones(
                max_seq_len,
                max_seq_len,
                dtype=torch.bool,
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape [batch, sequence, embedding]."
            )

        batch_size, sequence_length, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {embedding_dim}."
            )

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum length {self.max_seq_len}."
            )

        # Project input into Query, Key, and Value representations.
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        # Split the embedding dimension into multiple attention heads.
        queries = queries.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        keys = keys.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        values = values.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Calculate scaled attention scores.
        scores = torch.matmul(
            queries,
            keys.transpose(-2, -1),
        )

        scores = scores / math.sqrt(self.head_dim)

        # Apply causal masking.
        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]

        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        # Convert scores into attention probabilities.
        attention_weights = F.softmax(scores, dim=-1)

        # Aggregate Value vectors using attention probabilities.
        attention_output = torch.matmul(
            attention_weights,
            values,
        )

        # Merge attention heads back into one embedding dimension.
        attention_output = attention_output.transpose(1, 2).contiguous()

        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        # Combine information from all attention heads.
        return self.output_projection(attention_output)