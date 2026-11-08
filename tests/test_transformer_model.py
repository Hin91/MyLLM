
import pytest
import torch

from model.transformer_model import TransformerModel


@pytest.fixture
def model():
    return TransformerModel(
        vocab_size=100,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        max_seq_len=64,
    )


def test_output_shape(model):
    token_ids = torch.randint(0, 100, (4, 32))

    output = model(token_ids)

    assert output.shape == (4, 32, 32)


def test_output_dtype(model):
    token_ids = torch.randint(0, 100, (4, 32))

    output = model(token_ids)

    assert output.dtype == torch.float32


def test_output_is_finite(model):
    token_ids = torch.randint(0, 100, (4, 32))

    output = model(token_ids)

    assert torch.isfinite(output).all()


def test_gradient_flow(model):
    token_ids = torch.randint(0, 100, (4, 32))

    output = model(token_ids)
    loss = output.mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_number_of_transformer_blocks(model):
    assert len(model.blocks) == 2


def test_parameters_are_registered(model):
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count > 0


def test_causal_property(model):
    model.eval()

    first_sequence = torch.tensor(
        [[1, 2, 3, 4, 5, 6]]
    )

    second_sequence = torch.tensor(
        [[1, 2, 3, 4, 90, 91]]
    )

    with torch.no_grad():
        first_output = model(first_sequence)
        second_output = model(second_sequence)

    # Changing future tokens must not affect earlier outputs.
    assert torch.allclose(
        first_output[:, :4, :],
        second_output[:, :4, :],
        atol=1e-6,
    )


def test_invalid_input_dimensions(model):
    token_ids = torch.randint(0, 100, (4,))

    with pytest.raises(ValueError):
        model(token_ids)


def test_empty_batch(model):
    token_ids = torch.empty((0, 32), dtype=torch.long)

    with pytest.raises(ValueError):
        model(token_ids)


def test_empty_sequence(model):
    token_ids = torch.empty((4, 0), dtype=torch.long)

    with pytest.raises(ValueError):
        model(token_ids)


def test_sequence_too_long(model):
    token_ids = torch.randint(0, 100, (2, 65))

    with pytest.raises(ValueError):
        model(token_ids)


@pytest.mark.parametrize(
    "config",
    [
        {
            "vocab_size": 0,
            "embedding_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 64,
            "max_seq_len": 64,
        },
        {
            "vocab_size": 100,
            "embedding_dim": 32,
            "num_layers": 0,
            "num_heads": 4,
            "hidden_dim": 64,
            "max_seq_len": 64,
        },
        {
            "vocab_size": 100,
            "embedding_dim": 30,
            "num_layers": 2,
            "num_heads": 4,
            "hidden_dim": 64,
            "max_seq_len": 64,
        },
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        TransformerModel(**config)
