import pytest
import torch

from model.input_embeddings import InputEmbedding


@pytest.fixture
def input_embedding():
    return InputEmbedding(
        vocab_size=8000,
        max_seq_len=512,
        embedding_dim=256,
    )


def test_output_shape(input_embedding):
    token_ids = torch.randint(0, 8000, (16, 512))

    output = input_embedding(token_ids)

    assert output.shape == (16, 512, 256)


def test_output_dtype(input_embedding):
    token_ids = torch.randint(0, 8000, (2, 32))

    output = input_embedding(token_ids)

    assert output.dtype == torch.float32


def test_gradient_propagation(input_embedding):
    token_ids = torch.randint(0, 8000, (2, 32))

    output = input_embedding(token_ids)
    loss = output.sum()
    loss.backward()

    assert input_embedding.token_embedding.embedding.weight.grad is not None

    assert (
        input_embedding.positional_embedding.embedding.weight.grad
        is not None
    )


def test_sequence_length_limit(input_embedding):
    token_ids = torch.randint(0, 8000, (2, 513))

    with pytest.raises(ValueError):
        input_embedding(token_ids)


def test_invalid_input_dimensions(input_embedding):
    token_ids = torch.randint(0, 8000, (2, 32, 1))

    with pytest.raises(ValueError):
        input_embedding(token_ids)