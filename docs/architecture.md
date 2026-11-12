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

## Layer Normalization

### Purpose

Layer Normalization stabilizes neural network activations by normalizing feature values independently for each token.

Unlike Batch Normalization, Layer Normalization does not depend on statistics from other samples in the batch.

### Mathematical Definition

First, calculate the mean:

`mean = sum(x) / N`

Then calculate the population variance:

`variance = sum((x - mean)^2) / N`

Normalize the input:

`x_normalized = (x - mean) / sqrt(variance + epsilon)`

Finally, apply learnable scale and shift parameters:

`y = gamma * x_normalized + beta`

### Learnable Parameters

- `gamma`: learnable scale parameter, initialized to ones.
- `beta`: learnable shift parameter, initialized to zeros.
- `epsilon`: small constant for numerical stability.

### Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 256 |
| Epsilon | 1e-5 |
| Trainable parameters | 512 |

### Tensor Dimensions

For an input tensor `[16, 512, 256]`, normalization is performed independently over the last dimension.

The output retains the original shape `[16, 512, 256]`.

### Architecture Decision

The project uses Pre-LayerNorm.

Normalization is applied before Multi-Head Attention and before the Feed-Forward Network:

`x1 = x + Attention(LayerNorm(x))`

`x2 = x1 + FFN(LayerNorm(x1))`

### Implementation

Implemented in `model/layer_norm.py` using PyTorch operations and trainable parameters.

Population variance (`unbiased=False`) is used to match the standard PyTorch LayerNorm convention.

### Tests

The test suite verifies:

- Output shape preservation.
- Normalized mean and variance.
- Correct parameter count.
- Gradient propagation.
- Invalid input dimensions.
- Invalid configuration.
- Single-feature behavior.
# Transformer Model

## Overview

The Transformer Model combines the previously implemented neural network components into a complete Transformer backbone.

It converts input token IDs into contextualized vector representations by passing them through an embedding layer, a stack of Transformer blocks, and a final Layer Normalization layer.

The current architecture follows a decoder-only design and uses causal self-attention to preserve autoregressive behavior.

The Transformer backbone does not directly produce token probabilities. A separate Language Model Head will be responsible for converting its output representations into vocabulary logits.

## Configuration

| Parameter | Value |
|---|---:|
| Vocabulary size | 8,000 |
| Embedding dimension | 256 |
| Number of Transformer blocks | 6 |
| Number of attention heads | 4 |
| Attention head dimension | 64 |
| Feed-Forward hidden dimension | 1,024 |
| Maximum sequence length | 512 |

## Architecture

The model consists of three main components:

1. **Input Embedding** — converts token IDs into token and positional representations.
2. **Transformer Blocks** — processes representations through multiple layers of causal self-attention and feed-forward transformations.
3. **Final Layer Normalization** — normalizes the output representations after the last Transformer block.

The forward pass follows this structure:

`X = InputEmbedding(token_ids)`

`X = TransformerBlock_1(X)`

`X = TransformerBlock_2(X)`

`...`

`X = TransformerBlock_6(X)`

`Output = FinalLayerNorm(X)`

Each Transformer block has independent trainable parameters.

## Tensor Dimensions

For the initial training configuration:

| Component | Tensor shape |
|---|---|
| Input token IDs | `[16, 512]` |
| Input Embedding output | `[16, 512, 256]` |
| Each Transformer Block output | `[16, 512, 256]` |
| Final LayerNorm output | `[16, 512, 256]` |

The sequence length and embedding dimension remain unchanged throughout the Transformer backbone.

## Parameter Count

The initial architecture contains the following trainable parameters:

| Component | Parameters |
|---|---:|
| Input Embedding | 2,179,072 |
| Six Transformer Blocks | 4,738,560 |
| Final LayerNorm | 512 |
| **Total** | **6,918,144** |

The parameter count includes all embedding matrices, attention projections, feed-forward layers, and normalization parameters.

It does not include the Language Model Head, which will be implemented separately.

## Implementation

Implemented in `model/transformer_model.py` using:

- `InputEmbedding`
- `nn.ModuleList`
- `TransformerBlock`
- `LayerNorm`

The Transformer blocks are stored in `nn.ModuleList` to ensure that PyTorch registers all their parameters and includes them in device transfers, optimization, and state dictionary operations.

The forward method validates input dimensions, batch size, sequence length, and the configured maximum sequence length.

## Design Decisions

- The architecture uses six independently parameterized Transformer blocks.
- Pre-LayerNorm is used inside each Transformer block.
- Causal masking prevents tokens from accessing future positions.
- All Transformer blocks preserve the embedding dimension.
- A final Layer Normalization layer is applied after the last block.
- The maximum sequence length is initially limited to 512 tokens.
- The Transformer backbone and Language Model Head are implemented as separate components.

## Tests

The test suite verifies:

- Output tensor shape.
- Output data type.
- Finite output values.
- Number of Transformer blocks.
- Total parameter count.
- Gradient propagation through embeddings, Transformer blocks, and final normalization.
- Rejection of invalid input dimensions.
- Rejection of empty batches and sequences.
- Rejection of sequences exceeding the maximum length.
- Invalid configuration handling.

Test file: `tests/test_transformer_model.py`

## Architectural Role

The Transformer Model is the main feature extraction component of the language model.

It transforms each input token into a contextualized representation that incorporates information from preceding tokens.

Its output will be passed to the Language Model Head, which will project each contextualized representation into a vector of vocabulary logits.

These logits will be used during training to calculate the next-token prediction loss and during inference to select the next token.
# Language Model Head

## Overview

The Language Model Head is responsible for converting contextualized token representations produced by the Transformer backbone into vocabulary logits.

Each sequence position receives a separate vector of scores corresponding to all tokens in the tokenizer vocabulary.

These scores are used to predict the next token during autoregressive language modeling.

## Mathematical Representation

The Language Model Head applies a linear transformation:

`Logits = XW^T + b`

Where:

- **X** — contextualized Transformer representations.
- **W** — trainable projection weight matrix.
- **b** — trainable bias vector.
- **Logits** — unnormalized scores for vocabulary tokens.

For the current configuration:

- Input dimension: 256
- Vocabulary size: 8,000
- Weight matrix: `[8000, 256]`
- Bias vector: `[8000]`

## Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 256 |
| Vocabulary size | 8,000 |
| Projection type | Linear |
| Trainable parameters | 2,056,000 |
| Activation function | None |
| Softmax | Not applied |

## Tensor Dimensions

For batch size 16 and sequence length 512:

| Component | Tensor shape |
|---|---|
| Transformer output | `[16, 512, 256]` |
| Projection weight | `[8000, 256]` |
| Projection bias | `[8000]` |
| Language Model Head output | `[16, 512, 8000]` |

The linear projection operates on the last dimension of the input tensor while preserving batch size and sequence length.

## Implementation

Implemented in `model/language_model_head.py` using:

- `torch.nn.Linear`

The implementation validates input dimensions, embedding dimension, and configuration parameters.

## Design Decisions

- A separate linear projection is used to map Transformer representations to vocabulary logits.
- The projection has independent trainable weights and bias.
- Softmax is not applied inside the module.
- Raw logits are returned for compatibility with `torch.nn.CrossEntropyLoss`.
- Weight tying between token embeddings and the output projection is not implemented in the initial version.

## Tests

The test suite verifies:

- Output tensor shape.
- Output data type.
- Finite output values.
- Correct parameter count.
- Gradient propagation to input representations and projection parameters.
- Rejection of invalid input dimensions.
- Rejection of incorrect embedding dimensions.
- Rejection of empty sequences.
- Invalid configuration handling.

Test file: `tests/test_language_model_head.py`

## Architectural Role

The Language Model Head is the final projection component of the language model.

It converts contextualized representations into vocabulary logits, allowing the model to estimate the next-token distribution.

During training, these logits will be compared against the expected next-token IDs using a cross-entropy loss
# Language Model

## Overview

The Language Model is the complete decoder-only causal language model. It combines the Transformer Model backbone with a Language Model Head to predict the next token in a sequence.

It receives token IDs and produces vocabulary logits for every position in the input sequence.

## Architecture

The model consists of two main components:

1. **Transformer Model** — converts input token IDs into contextualized hidden representations.
2. **Language Model Head** — projects hidden representations into vocabulary-sized logits.

The data flow is:

```text
Input Token IDs
      |
      v
Transformer Model
      |
      v
Contextualized Representations
      |
      v
Language Model Head
      |
      v
Vocabulary Logits
```

## Tensor Dimensions

Let:

- B — batch size
- T — sequence length
- E — embedding dimension
- V — vocabulary size

Input:

`[B, T]`

Transformer output:

`[B, T, E]`

Language Model output:

`[B, T, V]`

Each position produces a vector of V logits representing the model's predictions for the next token.

## Configuration

The model uses the following parameters:

- `vocab_size` — tokenizer vocabulary size
- `embedding_dim` — hidden representation dimension
- `num_layers` — number of Transformer blocks
- `num_heads` — number of attention heads
- `hidden_dim` — Feed-Forward Network hidden dimension
- `max_seq_len` — maximum supported sequence length

## Forward Pass

1. Input token IDs are passed to the Transformer Model.
2. Token and positional embeddings are combined.
3. The sequence passes through all Transformer blocks.
4. Final Layer Normalization is applied.
5. The Language Model Head projects each hidden representation into vocabulary logits.
6. The resulting logits are returned without applying Softmax.

Softmax is not applied inside the model because training uses `CrossEntropyLoss`, which performs the required normalization internally.

## Parameter Count

For the configuration:

- Vocabulary size: 8,000
- Embedding dimension: 256
- Transformer layers: 6
- Attention heads: 4
- Feed-Forward hidden dimension: 1,024
- Maximum sequence length: 512

Parameter distribution:

| Component | Parameters |
|---|---:|
| Transformer Model | 6,918,144 |
| Language Model Head | 2,056,000 |
| **Total** | **8,974,144** |

The input token embeddings and output projection use separate parameters. Weight tying is not implemented.

## Design Decisions

- The model uses a decoder-only Transformer architecture.
- Causal self-attention prevents positions from accessing future tokens.
- Pre-Layer Normalization is used inside Transformer blocks.
- The output layer returns raw logits rather than probabilities.
- The architecture supports next-token prediction training.

## Architectural Role

The Language Model combines all previously implemented neural network components into a complete model capable of autoregressive language modeling.

It provides the model interface required by the training and inference pipelines.
## Language Model Loss

The `LanguageModelLoss` module calculates the training loss for next-token prediction.

### Purpose

During training, the Language Model predicts the next token for every position in an input sequence.

The loss function compares the predicted vocabulary logits with the corresponding target token IDs and measures the prediction error.

### Input and Output

The loss function accepts two tensors:

- `logits`: Model predictions with shape `[batch_size, seq_len, vocab_size]`.
- `targets`: Expected token IDs with shape `[batch_size, seq_len]`.

It returns a scalar loss value.

### Loss Calculation

The implementation uses PyTorch's `nn.CrossEntropyLoss`.

Cross-entropy measures how well the predicted probability distribution matches the correct target token.

`CrossEntropyLoss` internally applies the required log-softmax operation, so the model must provide raw logits without applying Softmax beforehand.

Before calculating the loss, the tensors are reshaped:

- `logits`: `[batch_size, seq_len, vocab_size]` -> `[batch_size * seq_len, vocab_size]`
- `targets`: `[batch_size, seq_len]` -> `[batch_size * seq_len]`

This allows the loss function to evaluate every token prediction across the entire batch.

By default, `nn.CrossEntropyLoss` uses mean reduction, returning the average loss across all token positions.

### Validation

The module validates:

- `logits` must have three dimensions.
- `targets` must have two dimensions.
- Batch size and sequence length must match between the two tensors.

Invalid input shapes raise a `ValueError`.

### Testing

Unit tests cover:

- Scalar output shape.
- Non-negative loss values.
- Gradient propagation through logits.
- Processing different batch sizes, sequence lengths, and vocabulary sizes.
- Invalid logits and targets dimensions.
- Mismatched batch and sequence dimensions.

### AdamW Optimizer

The `ModelOptimizer` class wraps PyTorch's built-in AdamW optimizer and manages the optimization of model parameters.

**Optimizer:** AdamW (`torch.optim.AdamW`)

**Default hyperparameters:**
- Learning rate: `3e-4`
- Betas: `(0.9, 0.999)`
- Epsilon: `1e-8`
- Weight decay: `0.01`

The optimizer receives all trainable parameters from the language model through `model.parameters()`.

AdamW maintains exponential moving averages of gradients and squared gradients to adapt parameter updates. It applies weight decay separately from the gradient-based update.

The class validates the model instance and optimizer hyperparameters before initialization.

During training:
1. `zero_grad()` clears gradients from the previous iteration.
2. `backward()` computes gradients using automatic differentiation.
3. `step()` updates model parameters.

The current implementation uses a single parameter group. Separate weight decay settings for biases and normalization parameters may be introduced later.
### Trainer

The `Trainer` class coordinates the model, loss function, optimizer, and data loaders during training and validation.

**Implemented methods:**

- `train_epoch()` — performs one complete training epoch, including forward pass, loss calculation, backpropagation, and optimizer updates.
- `validate()` — evaluates the model without calculating gradients or updating parameters.
- `train()` — runs multiple training epochs, performs validation after each epoch, and records loss history.

The Trainer supports CPU and CUDA devices. Model inputs and targets are moved to the configured device before computation.

Training and validation losses are stored separately in a history dictionary.

Validation uses `torch.no_grad()` to reduce memory consumption and avoid building computation graphs.