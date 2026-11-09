
import pytest
import torch

from model.language_model import LanguageModel


@pytest.fixture
def model():
    return LanguageModel(
        vocab_size=100,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        max_seq_len=16,
    )


def test_output_shape(model):
    token_ids = torch.randint(0, 100, (4, 10))

    logits = model(token_ids)

    assert logits.shape == (4, 10, 100)


def test_output_dtype(model):
    token_ids = torch.randint(0, 100, (2, 8))

    logits = model(token_ids)

    assert logits.dtype == torch.float32


def test_output_is_finite(model):
    token_ids = torch.randint(0, 100, (2, 8))

    logits = model(token_ids)

    assert torch.isfinite(logits).all()


def test_gradient_propagation(model):
    token_ids = torch.randint(0, 100, (2, 8))

    logits = model(token_ids)
    loss = logits.mean()

    loss.backward()

    assert (
        model.transformer.input_embedding
        .token_embedding.embedding.weight.grad is not None
    )

    assert (
        model.transformer.blocks[0]
        .attention.query_projection.weight.grad is not None
    )

    assert (
        model.language_model_head
        .projection.weight.grad is not None
    )


def test_parameter_registration(model):
    parameters = list(model.parameters())

    assert len(parameters) > 0

    assert all(parameter.requires_grad for parameter in parameters)


def test_sequence_too_long(model):
    token_ids = torch.randint(0, 100, (2, 17))

    with pytest.raises(ValueError):
        model(token_ids)


def test_invalid_input_dimensions(model):
    token_ids = torch.randint(0, 100, (2, 3, 4))

    with pytest.raises(ValueError):
        model(token_ids)
