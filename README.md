# MyLLM

A small decoder-only causal Transformer language model built from scratch using PyTorch.

The project aims to develop a language model focused on understanding and generating programming code in multiple languages, including Python, C++, C#, and Java.

## Project Status

The project is currently under active development.

### Completed

**Environment and Data Preparation**
- [x] Python and PyTorch environment setup
- [x] Multi-language source dataset preparation
- [x] Train/validation dataset splitting
- [x] Byte-Level BPE tokenizer implementation
- [x] Binary tokenization using uint16
- [x] PyTorch Dataset and DataLoader

**Model Architecture**
- [x] Token Embedding
- [x] Positional Embedding
- [x] Input Embedding
- [x] Causal Multi-Head Self-Attention
- [x] Feed-Forward Network
- [x] Layer Normalization
- [x] Residual Connections
- [x] Transformer Block
- [x] Transformer Model
- [x] Language Model Head
- [x] Complete Language Model architecture

**Training Components**
- [x] Language Model Loss (Cross Entropy)

### In Progress

- [ ] Optimizer implementation
- [ ] Training and validation loop
- [ ] Model checkpointing
- [ ] Training metrics and evaluation

### Planned

- [ ] Pretraining on the prepared multi-language dataset
- [ ] Model inference and text generation
- [ ] Instruction fine-tuning
- [ ] Improved dataset processing and validation

## Technology Stack

- Python 3.11
- PyTorch
- NumPy
- Hugging Face Tokenizers
- pytest

## Model Architecture

The project uses a decoder-only causal Transformer architecture.

Core components:
- Byte-Level BPE tokenizer
- Learned token and positional embeddings
- Causal multi-head self-attention
- Position-wise feed-forward networks
- Pre-Layer Normalization
- Residual connections
- Vocabulary projection head

The model is designed for autoregressive next-token prediction.

## Supported Programming Languages

- Python
- C++
- C#
- Java

The dataset also includes documentation.

## Current Limitations

- The model architecture is implemented but has not yet been pretrained.
- Dataset processing and document-boundary handling may require further improvements.
