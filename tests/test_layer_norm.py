
import pytest
import torch

from model.layer_norm import LayerNorm


def test_output_shape():
    model = LayerNorm(embedding_dim=256)

    x = torch.randn(16, 512, 256)
    output = model(x)

    assert output.shape == x.shape


def test_normalized_mean_and_variance():
    model = LayerNorm(embedding_dim=256)

    x = torch.randn(2, 8, 256)
    output = model(x)

    mean = output.mean(dim=-1)
    variance = output.var(dim=-1, unbiased=False)

    assert torch.allclose(
        mean,
        torch.zeros_like(mean),
        atol=1e-5,
    )

    assert torch.allclose(
        variance,
        torch.ones_like(variance),
        atol=1e-4,
    )


def test_parameter_count():
    model = LayerNorm(embedding_dim=256)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == 512


def test_gradient_propagation():
    model = LayerNorm(embedding_dim=256)

    x = torch.randn(2, 8, 256, requires_grad=True)

    output = model(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert model.gamma.grad is not None
    assert model.beta.grad is not None


def test_invalid_embedding_dimension():
    model = LayerNorm(embedding_dim=256)

    x = torch.randn(2, 8, 128)

    with pytest.raises(ValueError):
        model(x)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        LayerNorm(embedding_dim=0)

    with pytest.raises(ValueError):
        LayerNorm(embedding_dim=256, eps=0)


def test_single_feature_dimension():
    model = LayerNorm(embedding_dim=1)

    x = torch.tensor([[[5.0]]])
    output = model(x)

    assert torch.allclose(
        output,
        torch.zeros_like(output),
        atol=1e-5,
    )
