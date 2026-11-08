
import torch
from torch import nn

from model.input_embeddings import InputEmbedding
from model.layer_norm import LayerNorm
from model.transformer_block import TransformerBlock


class TransformerModel(nn.Module):
    """
    Decoder-only Transformer model.

    Converts token IDs into contextual representations
    using multiple Transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        max_seq_len: int,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Convert token IDs into vector representations.
        self.input_embedding = InputEmbedding(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
        )

        # Create a stack of Transformer blocks.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

        # Normalize the final hidden representations.
        self.final_norm = LayerNorm(embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of shape
                [batch_size, sequence_length].

        Returns:
            Contextual representations of shape
                [batch_size, sequence_length, embedding_dim].
        """

        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape "
                "[batch_size, sequence_length]"
            )

        if token_ids.shape[0] == 0:
            raise ValueError("batch_size must be positive")

        if token_ids.shape[1] == 0:
            raise ValueError("sequence_length must be positive")

        if token_ids.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Sequence length {token_ids.shape[1]} exceeds "
                f"maximum sequence length {self.max_seq_len}"
            )

        # Step 1: Token and positional embeddings.
        x = self.input_embedding(token_ids)

        # Step 2: Process representations through Transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Step 3: Final normalization.
        x = self.final_norm(x)

        return x
