
import pytest
import torch

from model.attention import CausalSelfAttention


def create_attention():
    return CausalSelfAttention(
        embedding_dim=256,
        max_seq_len=512,
    )


def test_output_shape():
    layer = create_attention()
    x = torch.randn(16, 512, 256)

    output = layer(x)

    assert output.shape == (16, 512, 256)


def test_output_dtype():
    layer = create_attention()
    x = torch.randn(2, 32, 256)

    output = layer(x)

    assert output.dtype == x.dtype


def test_gradient_propagation():
    layer = create_attention()
    x = torch.randn(2, 32, 256, requires_grad=True)

    output = layer(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.query_projection.weight.grad is not None
    assert layer.key_projection.weight.grad is not None
    assert layer.value_projection.weight.grad is not None
    assert layer.output_projection.weight.grad is not None


def test_causal_property():
    torch.manual_seed(42)

    layer = create_attention()
    layer.eval()

    x1 = torch.randn(1, 8, 256)
    x2 = x1.clone()

    # Modify only future positions.
    x2[:, 5:, :] = torch.randn(1, 3, 256)

    with torch.no_grad():
        output1 = layer(x1)
        output2 = layer(x2)

    # Earlier outputs must not depend on future inputs.
    assert torch.allclose(
        output1[:, :5, :],
        output2[:, :5, :],
        atol=1e-6,
    )


def test_sequence_length_limit():
    layer = create_attention()
    x = torch.randn(1, 513, 256)

    with pytest.raises(ValueError):
        layer(x)


def test_invalid_input_dimensions():
    layer = create_attention()
    x = torch.randn(2, 32, 256, 1)

    with pytest.raises(ValueError):
        layer(x)


def test_invalid_embedding_dimension():
    layer = create_attention()
    x = torch.randn(2, 32, 128)

    with pytest.raises(ValueError):
        layer(x)
