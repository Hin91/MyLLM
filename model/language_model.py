
import torch
import torch.nn as nn

from model.transformer_model import TransformerModel
from model.language_model_head import LanguageModelHead


class LanguageModel(nn.Module):
    """
    Decoder-only autoregressive language model.

    Combines a Transformer backbone with a vocabulary projection head.
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

        self.transformer = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )

        self.language_model_head = LanguageModelHead(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
        )

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transformer(token_ids)

        logits = self.language_model_head(hidden_states)

        return logits
