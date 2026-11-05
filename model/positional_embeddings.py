import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Converts token positions into trainable vector representations.
    """

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of shape
                (batch_size, sequence_length)

        Returns:
            Positional embeddings of shape
                (1, sequence_length, embedding_dim)
        """
        if token_ids.ndim != 2:
            raise ValueError(
                "Expected token_ids to have shape "
                "(batch_size, sequence_length)."
            )

        sequence_length = token_ids.size(1)

        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"maximum supported length ({self.max_seq_len})."
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        position_vectors = self.embedding(positions)

        return position_vectors.unsqueeze(0)