import pytest
import torch

from model.positional_embeddings import PositionalEmbedding


def test_output_shape():
    layer = PositionalEmbedding(
        max_seq_len=512,
        embedding_dim=256,
    )

    token_ids = torch.randint(0, 8000, (16, 512))
    output = layer(token_ids)

    assert output.shape == (1, 512, 256)


def test_parameter_count():
    layer = PositionalEmbedding(
        max_seq_len=512,
        embedding_dim=256,
    )

    expected_params = 512 * 256
    actual_params = sum(
        parameter.numel()
        for parameter in layer.parameters()
    )

    assert actual_params == expected_params


def test_gradient_propagation():
    layer = PositionalEmbedding(
        max_seq_len=512,
        embedding_dim=256,
    )

    token_ids = torch.randint(0, 8000, (2, 32))
    output = layer(token_ids)

    loss = output.sum()
    loss.backward()

    assert layer.embedding.weight.grad is not None


def test_sequence_length_limit():
    layer = PositionalEmbedding(
        max_seq_len=512,
        embedding_dim=256,
    )

    token_ids = torch.randint(0, 8000, (2, 513))

    with pytest.raises(ValueError):
        layer(token_ids)


def test_invalid_input_dimensions():
    layer = PositionalEmbedding(
        max_seq_len=512,
        embedding_dim=256,
    )

    token_ids = torch.randint(0, 8000, (16, 512, 1))

    with pytest.raises(ValueError):
        layer(token_ids)