import pytest
import torch
import torch.nn as nn

from training.optimizer import ModelOptimizer


def test_default_initialization():
    model = nn.Linear(10, 5)
    optimizer = ModelOptimizer(model)

    assert isinstance(optimizer.optimizer, torch.optim.AdamW)

    param_group = optimizer.optimizer.param_groups[0]

    assert param_group["lr"] == 3e-4
    assert param_group["betas"] == (0.9, 0.999)
    assert param_group["eps"] == 1e-8
    assert param_group["weight_decay"] == 0.01


def test_custom_parameters():
    model = nn.Linear(10, 5)

    optimizer = ModelOptimizer(
        model,
        learning_rate=1e-3,
        betas=(0.8, 0.99),
        eps=1e-7,
        weight_decay=0.05,
    )

    param_group = optimizer.optimizer.param_groups[0]

    assert param_group["lr"] == 1e-3
    assert param_group["betas"] == (0.8, 0.99)
    assert param_group["eps"] == 1e-7
    assert param_group["weight_decay"] == 0.05


def test_parameter_update():
    model = nn.Linear(10, 5)
    optimizer = ModelOptimizer(model)

    initial_weights = model.weight.detach().clone()

    x = torch.randn(4, 10)
    targets = torch.randn(4, 5)

    criterion = nn.MSELoss()

    optimizer.optimizer.zero_grad()

    predictions = model(x)
    loss = criterion(predictions, targets)

    loss.backward()
    optimizer.optimizer.step()

    assert not torch.equal(initial_weights, model.weight)


def test_invalid_model():
    with pytest.raises(TypeError):
        ModelOptimizer("not a model")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": -1},
        {"betas": (-0.1, 0.999)},
        {"betas": (0.9, 1.0)},
        {"eps": 0},
        {"weight_decay": -0.01},
    ],
)
def test_invalid_parameters(kwargs):
    model = nn.Linear(10, 5)

    with pytest.raises(ValueError):
        ModelOptimizer(model, **kwargs)