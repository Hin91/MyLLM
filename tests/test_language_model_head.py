
import pytest
import torch

from model.language_model_head import LanguageModelHead


@pytest.fixture
def head():
    return LanguageModelHead(
        embedding_dim=32,
        vocab_size=100,
    )


def test_output_shape(head):
    x = torch.randn(4, 10, 32)

    output = head(x)

    assert output.shape == (4, 10, 100)


def test_output_dtype(head):
    x = torch.randn(2, 8, 32)

    output = head(x)

    assert output.dtype == torch.float32


def test_output_is_finite(head):
    x = torch.randn(2, 8, 32)

    output = head(x)

    assert torch.isfinite(output).all()


def test_parameter_count(head):
    total_params = sum(
        parameter.numel()
        for parameter in head.parameters()
    )

    expected_params = 32 * 100 + 100

    assert total_params == expected_params


def test_gradient_propagation(head):
    x = torch.randn(2, 8, 32, requires_grad=True)

    output = head(x)
    loss = output.mean()

    loss.backward()

    assert x.grad is not None
    assert head.projection.weight.grad is not None
    assert head.projection.bias.grad is not None


def test_invalid_input_dimensions(head):
    x = torch.randn(2, 3)

    with pytest.raises(ValueError):
        head(x)


def test_invalid_embedding_dimension(head):
    x = torch.randn(2, 8, 64)

    with pytest.raises(ValueError):
        head(x)


def test_empty_sequence(head):
    x = torch.empty(2, 0, 32)

    with pytest.raises(ValueError):
        head(x)


@pytest.mark.parametrize(
    "embedding_dim, vocab_size",
    [
        (0, 100),
        (-1, 100),
        (32, 0),
        (32, -1),
    ],
)
def test_invalid_configuration(embedding_dim, vocab_size):
    with pytest.raises(ValueError):
        LanguageModelHead(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
        )
