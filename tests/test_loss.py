import pytest
import torch

from training.loss import LanguageModelLoss


def test_output_shape():
    loss_fn = LanguageModelLoss()

    logits = torch.randn(2, 4, 100)
    targets = torch.randint(0, 100, (2, 4))

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0

def test_loss_values():
    loss_fn = LanguageModelLoss()

    logits = torch.randn(2, 4, 100)
    targets = torch.randint(0, 100, (2, 4))

    loss = loss_fn(logits, targets)

    assert loss >= 0

def test_grad_requirement():
    loss_fn = LanguageModelLoss()
    logits = torch.randn(2, 4, 100, requires_grad=True)
    targets = torch.randint(0, 100, (2, 4))

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None

@pytest.mark.parametrize(
    "batch_size, seq_len, vocab_size",
    [
        (1, 4, 100),
        (2, 4, 100),
        (8, 16, 100),
        (4, 32, 500),
    ],
)
def test_batch_processing(batch_size, seq_len, vocab_size):
    loss_fn = LanguageModelLoss()

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

def test_logit_shape_error():
    loss_fn = LanguageModelLoss()

    logits = torch.randn(2, 4, 100,10)
    targets = torch.randint(0, 100, (2, 4))
    with pytest.raises(ValueError, match="Logits must have shape"):
        loss_fn(logits, targets)

def test_target_shape_error():
    loss_fn = LanguageModelLoss()

    logits = torch.randn(2, 4, 100)
    targets = torch.randint(0, 100, (2, 4,10))

    with pytest.raises(ValueError, match="Targets must have shape"):
        loss_fn(logits, targets)


def test_matching_shape_error():
    loss_fn = LanguageModelLoss()

    logits = torch.randn(2, 4, 100)
    targets = torch.randint(0, 100, (2, 5))
    with pytest.raises(ValueError, match="Logits and targets must have matching batch and sequence dimensions"):
        loss_fn(logits, targets)


