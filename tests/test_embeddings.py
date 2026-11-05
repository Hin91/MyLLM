import torch

from model.embeddings import TokenEmbedding


def test_embedding_output_shape():
    vocab_size = 8000
    embedding_dim = 256
    batch_size = 16
    sequence_length = 512

    model = TokenEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length),
    )

    output = model(token_ids)

    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )


def test_embedding_parameter_count():
    vocab_size = 8000
    embedding_dim = 256

    model = TokenEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == vocab_size * embedding_dim


def test_embedding_supports_gradients():
    model = TokenEmbedding(
        vocab_size=8000,
        embedding_dim=256,
    )

    token_ids = torch.tensor([[10, 20, 30]])

    output = model(token_ids)
    loss = output.sum()

    loss.backward()

    assert model.embedding.weight.grad is not None


def test_embedding_rejects_out_of_range_ids():
    model = TokenEmbedding(
        vocab_size=8000,
        embedding_dim=256,
    )

    invalid_ids = torch.tensor([[8000]])

    try:
        model(invalid_ids)
    except (IndexError, RuntimeError):
        pass
    else:
        raise AssertionError(
            "Embedding should reject token IDs outside vocabulary"
        )