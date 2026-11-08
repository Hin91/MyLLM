
import pytest
import torch

from model.feed_forward import FeedForward


def test_output_shape():
    model = FeedForward(
        embedding_dim=256,
        hidden_dim=1024,
    )

    x = torch.randn(16, 512, 256)
    output = model(x)

    assert output.shape == x.shape


def test_parameter_count():
    model = FeedForward(
        embedding_dim=256,
        hidden_dim=1024,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == 525568


def test_gradient_propagation():
    model = FeedForward(
        embedding_dim=256,
        hidden_dim=1024,
    )

    x = torch.randn(2, 8, 256, requires_grad=True)

    output = model(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    for parameter in model.parameters():
        assert parameter.grad is not None


def test_invalid_input_dimensions():
    model = FeedForward(
        embedding_dim=256,
        hidden_dim=1024,
    )

    x = torch.randn(2, 256)

    with pytest.raises(ValueError):
        model(x)


def test_invalid_embedding_dimension():
    model = FeedForward(
        embedding_dim=256,
        hidden_dim=1024,
    )

    x = torch.randn(2, 8, 128)

    with pytest.raises(ValueError):
        model(x)


def test_invalid_configuration():
    with pytest.raises(ValueError):
        FeedForward(embedding_dim=0, hidden_dim=1024)

    with pytest.raises(ValueError):
        FeedForward(embedding_dim=256, hidden_dim=0)
