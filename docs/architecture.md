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
# Causal Self-Attention

## Overview

Causal Self-Attention allows each token to incorporate information from previous tokens when constructing its contextual representation.

It is based on the Scaled Dot-Product Attention mechanism and is designed for autoregressive language modeling.

Unlike unrestricted self-attention, causal attention prevents tokens from accessing information from future positions.

## Architecture

The component performs the following operations:

1. Projects the input into Query, Key, and Value representations.
2. Calculates similarity scores between queries and keys.
3. Scales the scores by the square root of the head dimension.
4. Applies a causal mask to prevent access to future positions.
5. Applies Softmax to obtain attention weights.
6. Computes a weighted sum of Value vectors.
7. Applies an output linear projection.

## Mathematical Foundation

The attention mechanism is defined as:

**Attention(Q, K, V) = softmax(QKᵀ / √d_k) V**

Where:

- Q — Query representations.
- K — Key representations.
- V — Value representations.
- d_k — Dimension of the Key and Query vectors.

The causal mask replaces scores corresponding to future positions with negative infinity before Softmax.

As a result, future positions receive zero attention weight.

## Tensor Dimensions

## Tensor Dimensions

For the initial single-head configuration:

| Component | Tensor shape | Description |
|---|---|---|
| Input | `[16, 512, 256]` | Original token representations |
| Query | `[16, 512, 256]` | Query vectors produced by a learned linear projection |
| Key | `[16, 512, 256]` | Key vectors produced by a separate learned linear projection |
| Value | `[16, 512, 256]` | Value vectors produced by a separate learned linear projection |
| Transposed Key | `[16, 256, 512]` | Key tensor transposed for matrix multiplication |
| Attention scores | `[16, 512, 512]` | Pairwise attention scores between sequence positions |
| Attention weights | `[16, 512, 512]` | Normalized scores after causal masking and Softmax |
| Attention output | `[16, 512, 256]` | Weighted combinations of Value vectors |
| Final output | `[16, 512, 256]` | Attention output after the final linear projection |

**Note:** Query, Key, and Value have identical tensor shapes, but contain different values because they are generated using independent trainable linear projections. The attention mechanism transforms the content of the representations while preserving the sequence length and embedding dimension.

## Implementation

Implemented in `model/attention.py` using:

- PyTorch linear projections
- Scaled Dot-Product Attention
- Lower-triangular causal masking
- Softmax normalization
- Weighted Value aggregation
- Output linear projection

The current implementation uses a single attention head as an intermediate development stage.

## Design Decisions

- Query, Key, and Value are produced using separate trainable linear projections.
- Causal masking is applied before Softmax.
- The causal mask is registered as a non-trainable module buffer.
- The output dimension remains equal to the input embedding dimension.
- The maximum supported sequence length is configurable.
- Attention dropout is not included in this initial implementation.

## Tests

Automated tests verify:

- Output tensor shape.
- Output data type.
- Gradient propagation through all projections.
- Causal behavior: future inputs cannot affect earlier outputs.
- Rejection of sequences exceeding the configured maximum length.
- Rejection of invalid input dimensions.

Test file: `tests/test_attention.py`

## Computational Complexity

The attention score matrix has dimensions proportional to the square of the sequence length.

The main attention computation has time complexity proportional to:

**O(B × T² × D)**

Where:

- B — batch size.
- T — sequence length.
- D — embedding dimension.

The attention score matrix requires memory proportional to:

**O(B × T²)**

## Architectural Role

Causal Self-Attention is responsible for contextual information exchange between token positions.

Its output will be passed to subsequent Transformer components.

# Multi-Head Attention

## Overview

Multi-Head Attention extends single-head causal self-attention
by processing multiple attention heads in parallel.

Each head operates on a separate portion of the embedding
dimension. Their outputs are concatenated and transformed
using a final linear projection.

## Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 256 |
| Number of heads | 4 |
| Head dimension | 64 |
| Maximum sequence length | 512 |

The head dimension is calculated as:

head_dim = embedding_dim / num_heads = 256 / 4 = 64

## Tensor Dimensions

For batch size 16 and sequence length 512:

| Component | Tensor shape |
|---|---|
| Input | [16, 512, 256] |
| Query | [16, 4, 512, 64] |
| Key | [16, 4, 512, 64] |
| Value | [16, 4, 512, 64] |
| Attention scores | [16, 4, 512, 512] |
| Attention weights | [16, 4, 512, 512] |
| Attention output | [16, 4, 512, 64] |
| Concatenated output | [16, 512, 256] |
| Final output | [16, 512, 256] |

## Implementation

Implemented in `model/multi_head_attention.py` using:

- Independent Query, Key, and Value projections.
- Multiple parallel attention heads.
- Scaled dot-product attention.
- Causal masking.
- Softmax normalization.
- Attention head concatenation.
- Final output projection.

## Parameter Count

The implementation contains four linear projections.

Each projection has:

256 × 256 + 256 = 65,792 parameters.

Total:

4 × 65,792 = 263,168 parameters.

## Tests

Test file: `tests/test_multi_head_attention.py`

Automated tests verify:

- Output shape and data type.
- Attention head configuration.
- Parameter count.
- Gradient propagation.
- Causal behavior.
- Maximum sequence length.
- Input dimension validation.
- Attention head configuration validation.
## Feed-Forward Network

### Purpose

The Feed-Forward Network (FFN) is a position-wise neural network used inside each Transformer block.

While Multi-Head Attention allows tokens to exchange information across sequence positions, the FFN independently transforms the representation of each token.

### Architecture

The network consists of two linear layers and a GELU activation function:

`FFN(x) = Linear2(GELU(Linear1(x)))`

The first linear layer expands the embedding dimension, while the second projects it back to its original size.

### Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 256 |
| Hidden dimension | 1024 |
| Expansion factor | 4 |
| Activation | GELU |

### Tensor Dimensions

For an input tensor with shape `[batch, sequence, embedding]`:

- Input: `[16, 512, 256]`
- First linear layer: `[16, 512, 1024]`
- GELU: `[16, 512, 1024]`
- Second linear layer: `[16, 512, 256]`
- Output: `[16, 512, 256]`

### Parameter Count

The first linear layer contains 263,168 parameters.

The second linear layer contains 262,400 parameters.

Total: **525,568 trainable parameters**.

### Implementation

Implemented in `model/feed_forward.py` using PyTorch.

The implementation validates input dimensions and embedding size.

### Tests

The test suite verifies:

- Output shape preservation.
- Correct parameter count.
- Gradient propagation.
- Input dimension validation.
- Invalid configuration handling.

## Residual Connections

### Purpose

Residual connections provide a direct path for information and gradients to flow through the Transformer architecture.

Instead of replacing the input with a transformed representation, the network adds the transformed output to the original input.

### Mathematical Definition

`Output = x + F(x)`

Where:
- `x` is the original input.
- `F(x)` is the output of a neural network sublayer.

### Application in the Transformer

Residual connections are used after both Multi-Head Attention and the Feed-Forward Network.

The selected Pre-LayerNorm architecture uses the following equations:

`x1 = x + Attention(LayerNorm(x))`

`x2 = x1 + FFN(LayerNorm(x1))`

### Benefits

- Provides a direct gradient path.
- Helps preserve existing representations.
- Makes optimization of deeper neural networks easier.

### Implementation

Residual connections do not require a separate trainable module.

They will be implemented directly inside `TransformerBlock` using tensor addition.

### Architecture Decision

The project uses a Pre-LayerNorm Transformer block, where Layer Normalization is applied before each main sublayer.
