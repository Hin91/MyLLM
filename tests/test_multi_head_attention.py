import pytest
import torch

from model.multi_head_attention import MultiHeadAttention


@pytest.fixture
def attention():
    return MultiHeadAttention(
        embedding_dim=256,
        num_heads=4,
        max_seq_len=512,
    )


def test_output_shape(attention):
    x = torch.randn(2, 32, 256)

    output = attention(x)

    assert output.shape == (2, 32, 256)


def test_output_dtype(attention):
    x = torch.randn(2, 32, 256)

    output = attention(x)

    assert output.dtype == x.dtype


def test_head_configuration(attention):
    assert attention.num_heads == 4
    assert attention.head_dim == 64


def test_parameter_count(attention):
    expected_parameters = 4 * (256 * 256 + 256)

    actual_parameters = sum(
        parameter.numel()
        for parameter in attention.parameters()
    )

    assert actual_parameters == expected_parameters


def test_gradient_propagation(attention):
    x = torch.randn(2, 16, 256, requires_grad=True)

    output = attention(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    assert attention.query_projection.weight.grad is not None
    assert attention.key_projection.weight.grad is not None
    assert attention.value_projection.weight.grad is not None
    assert attention.output_projection.weight.grad is not None


def test_causal_property(attention):
    attention.eval()

    x1 = torch.randn(1, 8, 256)
    x2 = x1.clone()

    # Modify only future positions.
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])

    with torch.no_grad():
        output1 = attention(x1)
        output2 = attention(x2)

    # Earlier positions must remain unaffected.
    assert torch.allclose(
        output1[:, :5, :],
        output2[:, :5, :],
        atol=1e-6,
    )


def test_sequence_length_limit(attention):
    x = torch.randn(1, 513, 256)

    with pytest.raises(ValueError):
        attention(x)


def test_invalid_embedding_dimension():
    with pytest.raises(ValueError):
        MultiHeadAttention(
            embedding_dim=255,
            num_heads=4,
            max_seq_len=512,
        )


def test_invalid_input_dimensions(attention):
    x = torch.randn(2, 32, 256, 1)

    with pytest.raises(ValueError):
        attention(x)


def test_invalid_number_of_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(
            embedding_dim=256,
            num_heads=0,
            max_seq_len=512,
        )