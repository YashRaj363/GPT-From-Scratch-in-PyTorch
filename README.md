# GPT From Scratch in PyTorch

A small **GPT-style language model built from scratch using PyTorch**, following Andrej Karpathy's "Let's build GPT from scratch" / Zero to Hero approach.

This project is built for understanding how a Transformer-based language model works internally — from tokenization and batching to self-attention, multi-head attention, Transformer blocks, and autoregressive text generation.

---

## Project Overview

The model is trained on the **Tiny Shakespeare** dataset and performs **character-level language modeling**.

The main task is:

> Given the previous characters, predict the next character.

For example:

```text
Input  : "To be or no"
Target : "t"
```

The model learns this task by repeatedly predicting the next character and updating its parameters using backpropagation.

---

## Architecture

```text
Input Text
    ↓
Character Encoding
    ↓
Token Embedding
    +
Positional Embedding
    ↓
Transformer Block
    ├── Layer Normalization
    ├── Multi-Head Self-Attention
    ├── Residual Connection
    ├── Layer Normalization
    ├── Feed Forward Network
    └── Residual Connection
    ↓
Repeat Transformer Block
    ↓
Final Layer Normalization
    ↓
Linear Layer
    ↓
Logits
    ↓
Softmax
    ↓
Next Character Prediction
```

The final model contains multiple Transformer blocks stacked together.

---

## Key Concepts Implemented

This project implements the main building blocks of a GPT-style Transformer:

- Character-level tokenization
- Character-to-integer encoding
- Training and validation split
- Mini-batch data generation
- Next-token prediction
- Bigram language model as a baseline
- Self-attention
- Causal masking
- Scaled dot-product attention
- Query, Key and Value projections
- Multi-head self-attention
- Feed-forward neural network
- Residual connections
- Layer Normalization
- Token embeddings
- Positional embeddings
- Autoregressive text generation

---

## Dataset

The model uses the **Tiny Shakespeare** dataset.

The dataset can be downloaded with:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

The text is then converted into integer IDs using a character-level vocabulary.

---

## Tokenization

This project uses **character-level tokenization** rather than word-level tokenization.

A vocabulary is created from all unique characters in the dataset.

Example:

```text
"hii there"
```

is converted into integer IDs.

The project creates two mappings:

```text
stoi → string to integer
itos → integer to string
```

These mappings are used for encoding input text and decoding generated output.

---

## Training Data

The dataset is divided into:

```text
90% → Training data
10% → Validation data
```

The model uses a fixed context length:

```text
block_size = 32
```

This means the model can use up to 32 previous characters as context when making predictions.

---

## Self-Attention

The project implements self-attention manually using:

```text
Query (Q)
Key   (K)
Value (V)
```

The attention score is calculated using:

```text
QKᵀ / √d
```

Then:

```text
Attention Weights
        ↓
   Weighted Values
        ↓
 Attention Output
```

A causal mask is applied so that a token cannot look at future tokens.

For example:

```text
Token 1 → can see Token 1
Token 2 → can see Token 1, Token 2
Token 3 → can see Token 1, Token 2, Token 3
...
```

This makes the model suitable for autoregressive next-token prediction.

---

## Multi-Head Attention

Instead of using only one attention head, the model uses multiple attention heads in parallel.

For this implementation:

```text
n_head = 4
```

Each head learns its own Query, Key and Value projections.

The outputs of the heads are concatenated and passed through a final linear projection.

---

## Transformer Blocks

The model uses:

```text
n_layer = 4
```

Transformer blocks.

Each block contains:

```text
LayerNorm
   ↓
Multi-Head Self-Attention
   ↓
Residual Connection
   ↓
LayerNorm
   ↓
Feed Forward Network
   ↓
Residual Connection
```

The model treats self-attention as the communication mechanism between tokens and the feed-forward network as the computation stage.

---

## Model Configuration

The current implementation uses:

```text
Batch Size         = 16
Context Length     = 32
Embedding Size     = 64
Attention Heads    = 4
Transformer Layers = 4
Learning Rate      = 0.001
Training Steps     = 5000
Dropout            = 0.0
```

The model automatically uses a CUDA GPU when available.

---

## Training

The model is trained using:

```text
Optimizer: AdamW
Loss: Cross Entropy
```

The training process is:

```text
Get Batch
   ↓
Forward Pass
   ↓
Calculate Loss
   ↓
Backpropagation
   ↓
AdamW Parameter Update
```

Training and validation loss are evaluated periodically during training.

---

## Text Generation

After training, the model can generate new Shakespeare-style text.

Generation works autoregressively:

```text
Starting context
      ↓
Predict next character
      ↓
Append predicted character
      ↓
Predict next character
      ↓
Append predicted character
      ↓
Repeat
```

The next character is sampled from the predicted probability distribution.

---

## Example

Given a starting context such as:

```text
""
```

the model generates text character by character.

The generated text is produced entirely by the trained model.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- CUDA (when available)
- Google Colab / Jupyter Notebook

---

## Project Structure

```text
GPT-from-Scratch/
│
├── gpt_dev.ipynb
├── input.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd GPT-from-Scratch
```

### 2. Install PyTorch

```bash
pip install torch
```

### 3. Run the notebook

Open:

```text
gpt_dev.ipynb
```

using:

- Google Colab
- Jupyter Notebook
- VS Code

### 4. Train the model

Run the notebook cells sequentially.

---

## Learning Goals

The main purpose of this project is to understand how a GPT-style Transformer works internally rather than treating it as a black box.

Through this project, I explored:

```text
Tokenization
    ↓
Embeddings
    ↓
Positional Embeddings
    ↓
Self-Attention
    ↓
Causal Masking
    ↓
Multi-Head Attention
    ↓
Feed Forward Network
    ↓
Layer Normalization
    ↓
Residual Connections
    ↓
Transformer Blocks
    ↓
Next-Token Prediction
    ↓
Text Generation
```

---

## Reference

This project follows the educational approach and notebook associated with Andrej Karpathy's **Zero to Hero** GPT material.

- Andrej Karpathy: https://karpathy.ai/
- GPT from Scratch video: https://www.youtube.com/watch?v=kCc8FmEb1nY

---

## Disclaimer

This is a **small educational GPT implementation**, not a production-scale language model.

The purpose of the project is to understand the internal components of Transformer-based language models by implementing them in PyTorch.
