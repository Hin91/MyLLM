import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.positional_embeddings import PositionalEmbedding


class InputEmbedding(nn.Module):
    """
    Combines token embeddings and positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.positional_embedding = PositionalEmbedding(
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of shape
                (batch_size, sequence_length)

        Returns:
            Combined embeddings of shape
                (batch_size, sequence_length, embedding_dim)
        """
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.positional_embedding(token_ids)

        return token_vectors + position_vectors