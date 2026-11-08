
import pytest
import torch

from model.transformer_block import TransformerBlock


@pytest.fixture
def block():
    return TransformerBlock(
        embedding_dim=256,
        num_heads=4,
        hidden_dim=1024,
        max_seq_len=512,
    )


def test_output_shape(block):
    x = torch.randn(2, 128, 256)

    output = block(x)

    assert output.shape == (2, 128, 256)


def test_output_dtype(block):
    x = torch.randn(2, 128, 256)

    output = block(x)

    assert output.dtype == x.dtype


def test_output_is_finite(block):
    x = torch.randn(2, 128, 256)

    output = block(x)

    assert torch.isfinite(output).all()


def test_gradient_flow(block):
    x = torch.randn(2, 128, 256, requires_grad=True)

    output = block(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_parameter_count(block):
    parameter_count = sum(
        parameter.numel()
        for parameter in block.parameters()
    )

    assert parameter_count == 789760


def test_invalid_input_dimensions(block):
    x = torch.randn(2, 256)

    with pytest.raises(ValueError):
        block(x)


def test_invalid_embedding_dimension(block):
    x = torch.randn(2, 128, 128)

    with pytest.raises(ValueError):
        block(x)


def test_sequence_too_long(block):
    x = torch.randn(2, 513, 256)

    with pytest.raises(ValueError):
        block(x)


@pytest.mark.parametrize(
    "embedding_dim, num_heads, hidden_dim, max_seq_len",
    [
        (0, 4, 1024, 512),
        (256, 0, 1024, 512),
        (256, 4, 0, 512),
        (256, 4, 1024, 0),
        (250, 4, 1024, 512),
    ],
)
def test_invalid_configuration(
    embedding_dim,
    num_heads,
    hidden_dim,
    max_seq_len,
):
    with pytest.raises(ValueError):
        TransformerBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )
