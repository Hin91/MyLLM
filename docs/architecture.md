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

# Positional Embedding

## Overview

The Positional Embedding layer provides the Transformer with information about the position of each token in a sequence.

Token embeddings represent token identity, but they do not explicitly encode token order. Positional embeddings address this limitation by assigning a trainable vector to each sequence position.

## Mathematical Representation

The positional embedding matrix is defined as:

**P ∈ ℝ^(L × D)**

Where:

- **L** is the maximum supported sequence length.
- **D** is the embedding dimension.
- **P** is the trainable positional embedding matrix.

For the current project configuration:

**P ∈ ℝ^(512 × 256)**

Each row represents one sequence position.

## Configuration

| Parameter | Value |
|---|---:|
| Maximum sequence length | 512 |
| Embedding dimension | 256 |
| Number of parameters | 131,072 |
| Embedding type | Learned |
| Framework | PyTorch |

## Implementation

The layer is implemented using `torch.nn.Embedding`.

For an input sequence of length N, the layer generates position indices from 0 to N - 1 and retrieves their corresponding vectors.

The output has shape:

`[1, sequence_length, embedding_dim]`

The leading dimension is 1 because positional vectors are shared across all sequences in a batch.

## Design Decisions

- Learned positional embeddings are used instead of fixed sinusoidal encodings.
- The maximum sequence length is initially limited to 512 tokens.
- Positional vectors are trainable parameters.
- Position indices are generated on the same device as the input tensor.
- The layer validates input dimensions and maximum sequence length.
- Broadcasting allows positional vectors to be added to token embeddings without explicitly copying them across the batch.

## Integration

Token and positional embeddings are combined through element-wise addition:

**X = E_token + E_position**

Both tensors must have compatible dimensions.

For the initial training configuration:

- Token embeddings: `[16, 512, 256]`
- Positional embeddings: `[1, 512, 256]`
- Combined representation: `[16, 512, 256]`

## Tests

Automated tests verify:

- Output tensor shape.
- Number of trainable parameters.
- Gradient propagation.
- Rejection of sequences exceeding the maximum length.
- Rejection of inputs with invalid dimensions.

Test file: `tests/test_positional_embeddings.py`

## Current Limitations

- The model cannot process sequences longer than 512 tokens using this configuration.
- Learned positional embeddings are tied to absolute sequence positions.
- The positional embedding layer does not independently model relationships between tokens.

The next architectural component, Causal Self-Attention, will allow the model to incorporate information from preceding tokens.
# Input Embedding

## Overview

The Input Embedding component combines token embeddings and positional embeddings into a single representation that serves as the input to the Transformer network.

It acts as an integration layer between token IDs and subsequent Transformer components.

## Architecture

The component consists of three operations:

1. Token IDs are converted into token vectors.
2. Sequence positions are converted into positional vectors.
3. Both representations are combined through element-wise addition.

**X = E_token + E_position**

## Tensor Dimensions

For the initial model configuration:

| Component | Tensor shape |
|---|---|
| Input token IDs | `[16, 512]` |
| Token embeddings | `[16, 512, 256]` |
| Positional embeddings | `[1, 512, 256]` |
| Combined output | `[16, 512, 256]` |

Positional embeddings use broadcasting along the batch dimension.

## Implementation

Implemented in `model/input_embeddings.py` using:

- `TokenEmbedding`
- `PositionalEmbedding`
- Element-wise tensor addition

## Design Decisions

- Token and positional embeddings remain separate modules.
- InputEmbedding combines the modules without duplicating their internal logic.
- The output dimension remains unchanged.
- Both embedding components are trainable.
- The maximum sequence length is controlled by the positional embedding component.

## Tests

Automated tests verify:

- Output tensor shape.
- Output data type.
- Gradient propagation to both embedding matrices.
- Rejection of sequences exceeding the configured maximum length.
- Rejection of inputs with invalid dimensions.

Test file: `tests/test_input_embeddings.py`

## Architectural Role

InputEmbedding is the first major component of the Transformer model.

Its output will be passed to the Causal Self-Attention layer, where the model will begin learning relationships between tokens while respecting the autoregressive constraint.