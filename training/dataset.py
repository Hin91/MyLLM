from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# Data paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_ROOT / "data" / "tokenized" / "train.bin"
VALIDATION_FILE = PROJECT_ROOT / "data" / "tokenized" / "validation.bin"


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class TokenDataset(Dataset):

    def __init__(self, file_path: Path, block_size: int):

        self.block_size = block_size

        # Loading a binary array of tokens
        self.tokens = np.fromfile(
            file_path,
            dtype=np.uint16,
        )

        if len(self.tokens) <= block_size:
            raise ValueError(
                "Dataset is too small for the selected block_size"
            )

        # Number of non-overlapping blocks.
        self.num_samples = (len(self.tokens) - 1) // block_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        # Starting position of the current block.
        start = index * self.block_size

        # Taking block_size + 1 tokens.
        chunk = self.tokens[
            start : start + self.block_size + 1
        ]

        # Input: all tokens except the last one.
        x = torch.tensor(
            chunk[:-1].astype(np.int64),
            dtype=torch.long,
        )

        # Response: all tokens except the first one.
        y = torch.tensor(
            chunk[1:].astype(np.int64),
            dtype=torch.long,
        )

        return x, y


# --------------------------------------------------
# DataLoader creation
# --------------------------------------------------

def create_dataloaders(
    block_size: int = 512,
    batch_size: int = 16,
):

    train_dataset = TokenDataset(
        TRAIN_FILE,
        block_size,
    )

    validation_dataset = TokenDataset(
        VALIDATION_FILE,
        block_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, validation_loader


# --------------------------------------------------
# Testing
# --------------------------------------------------

def main():

    block_size = 512
    batch_size = 16

    train_loader, validation_loader = create_dataloaders(
        block_size=block_size,
        batch_size=batch_size,
    )

    print("Dataset loaded successfully!")

    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")

    x, y = next(iter(train_loader))

    print("\nBatch information:")

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    print(f"x dtype: {x.dtype}")
    print(f"y dtype: {y.dtype}")

    print("\nFirst training example:")

    print("x:", x[0][:20].tolist())
    print("y:", y[0][:20].tolist())

    # Checking that each response actually corresponds to the next input token.
    assert torch.equal(x[:, 1:], y[:, :-1])

    print("\nShift test passed!")


if __name__ == "__main__":
    main()