import torch
import torch.nn as nn

from training.loss import LanguageModelLoss
from training.optimizer import ModelOptimizer


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: LanguageModelLoss,
        optimizer: ModelOptimizer,
        device: torch.device,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        self.model.to(self.device)

    def train_epoch(self, train_loader) -> float:
        self.model.train()

        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.loss_fn(logits, y)

            loss.backward()

            self.optimizer.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        return average_loss

    def validate(self, validation_loader) -> float:
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for x, y in validation_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.loss_fn(logits, y)

                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()

        if total_tokens == 0:
            raise ValueError("Validation DataLoader is empty.")

        average_loss = total_loss / total_tokens

        return average_loss

    def train(
            self,
            train_loader,
            validation_loader,
            num_epochs: int,
    ) -> dict[str, list[float]]:
        if num_epochs <= 0:
            raise ValueError("Number of epochs must be positive.")

        history = {
            "train_loss": [],
            "validation_loss": [],
        }

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            validation_loss = self.validate(validation_loader)

            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Validation Loss: {validation_loss:.4f}"
            )

        return history