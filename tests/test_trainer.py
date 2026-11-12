import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest


from training.loss import LanguageModelLoss
from training.optimizer import ModelOptimizer
from training.trainer import Trainer


def create_trainer():
    model = nn.Sequential(
        nn.Embedding(100, 16),
        nn.Linear(16, 100),
    )

    loss_fn = LanguageModelLoss()
    optimizer = ModelOptimizer(model)
    device = torch.device("cpu")

    return Trainer(model, loss_fn, optimizer, device)


def test_train_epoch_returns_loss():
    trainer = create_trainer()

    inputs = torch.randint(0, 100, (8, 16))
    targets = torch.randint(0, 100, (8, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    average_loss = trainer.train_epoch(loader)

    assert isinstance(average_loss, float)
    assert average_loss > 0
    assert torch.isfinite(torch.tensor(average_loss))


def test_train_epoch_updates_parameters():
    trainer = create_trainer()

    initial_parameters = [
        parameter.detach().clone()
        for parameter in trainer.model.parameters()
    ]

    inputs = torch.randint(0, 100, (8, 16))
    targets = torch.randint(0, 100, (8, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    trainer.train_epoch(loader)

    for initial, updated in zip(
        initial_parameters,
        trainer.model.parameters(),
    ):
        assert not torch.equal(initial, updated)


def test_train_epoch_sets_training_mode():
    trainer = create_trainer()
    trainer.model.eval()

    inputs = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    trainer.train_epoch(loader)

    assert trainer.model.training is True

def test_validate_returns_loss():
    trainer = create_trainer()

    inputs = torch.randint(0, 100, (8, 16))
    targets = torch.randint(0, 100, (8, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    average_loss = trainer.validate(loader)

    assert isinstance(average_loss, float)
    assert average_loss > 0
    assert torch.isfinite(torch.tensor(average_loss))


def test_validate_does_not_update_parameters():
    trainer = create_trainer()

    initial_parameters = [
        parameter.detach().clone()
        for parameter in trainer.model.parameters()
    ]

    inputs = torch.randint(0, 100, (8, 16))
    targets = torch.randint(0, 100, (8, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    trainer.validate(loader)

    for initial, updated in zip(
        initial_parameters,
        trainer.model.parameters(),
    ):
        assert torch.equal(initial, updated)


def test_validate_sets_evaluation_mode():
    trainer = create_trainer()

    inputs = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    trainer.model.train()
    trainer.validate(loader)

    assert trainer.model.training is False


def test_validate_does_not_calculate_gradients():
    trainer = create_trainer()

    inputs = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    trainer.validate(loader)

    for parameter in trainer.model.parameters():
        assert parameter.grad is None


def test_validate_empty_loader():
    trainer = create_trainer()

    empty_dataset = TensorDataset(
        torch.empty((0, 16), dtype=torch.long),
        torch.empty((0, 16), dtype=torch.long),
    )
    loader = DataLoader(empty_dataset, batch_size=4)

    with pytest.raises(ValueError, match="Validation DataLoader is empty"):
        trainer.validate(loader)

def test_train_returns_history():
    trainer = create_trainer()

    inputs = torch.randint(0, 100, (8, 16))
    targets = torch.randint(0, 100, (8, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    history = trainer.train(
        train_loader=loader,
        validation_loader=loader,
        num_epochs=3,
    )

    assert isinstance(history, dict)
    assert "train_loss" in history
    assert "validation_loss" in history

    assert len(history["train_loss"]) == 3
    assert len(history["validation_loss"]) == 3

    assert all(isinstance(loss, float) for loss in history["train_loss"])
    assert all(isinstance(loss, float) for loss in history["validation_loss"])


def test_train_rejects_invalid_num_epochs():
    trainer = create_trainer()

    inputs = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))

    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4)

    with pytest.raises(ValueError, match="Number of epochs must be positive"):
        trainer.train(
            train_loader=loader,
            validation_loader=loader,
            num_epochs=0,
        )