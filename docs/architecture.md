# Architecture Notes: Token Embedding

## Overview

The Token Embedding layer converts discrete token IDs into dense vector representations used by the Transformer architecture.

The tokenizer produces integer IDs from a fixed vocabulary. These IDs are not numerical measurements and do not encode semantic relationships by themselves.

The embedding layer maps each token ID to a trainable vector.

## Configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 8,000 |
| Embedding dimension | 256 |
| Embedding parameters | 2,048,000 |

## Mathematical Representation

The embedding matrix is defined as:

E ∈ ℝ^(V × D)

Where:

- \(V\) is the vocabulary size.
- \(D\) is the embedding dimension.

For the current configuration:

E ∈ ℝ^(8000 × 256)

Each token ID selects one row from this matrix.

## Input and Output

Input tensor:

`[batch_size, sequence_length]`

Output tensor:

`[batch_size, sequence_length, embedding_dim]`

With the initial training configuration:

- Batch size: 16
- Sequence length: 512
- Embedding dimension: 256

Expected output shape:

`[16, 512, 256]`

## Implementation

The layer is implemented using `torch.nn.Embedding`.

Embedding weights are initialized by PyTorch and updated during training through backpropagation and gradient-based optimization.

## Design Decisions

- Token IDs are stored in compact `uint16` binary datasets.
- IDs are converted to `torch.long` before embedding lookup.
- Positional information is handled by a separate component.
- The embedding layer is independent of the tokenizer implementation.

## Tests

Automated tests verify:

- Output tensor shape.
- Embedding parameter count.
- Gradient propagation.
- Rejection of token IDs outside the vocabulary.

## Current Limitations

This component represents token identity but does not encode token position.

Positional embeddings and Transformer attention will be implemented in subsequent stages.