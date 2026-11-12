import torch
import torch.nn as nn


class ModelOptimizer:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be an instance of nn.Module.")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")

        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("Betas must be in the range [0, 1).")

        if eps <= 0:
            raise ValueError("Epsilon must be positive.")

        if weight_decay < 0:
            raise ValueError("Weight decay must be non-negative.")

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )