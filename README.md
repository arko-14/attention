# Attention is all you Need

# [Read the full paper here]:-(https://arxiv.org/abs/1706.03762)

# Transformer Paper Implementation Notebook

This Colab notebook provides a paper‑from‑scratch implementation of the original Transformer architecture described in “Attention Is All You Need” (Vaswani et al., 2017) using TensorFlow 2.x and NumPy. It walks you through each component—from tokenization to training—within a single, self‑contained notebook.

---

## Table of Contents

1. [Overview]
2. [Notebook Sections]
3. [How to Use]
4. [Runtime Requirements]
5. [Hyperparameters]
6. [Troubleshooting & Tips]
7. [Acknowledgments]

---

## Overview

This notebook demonstrates a minimal Transformer setup on a small ~30 k‑word corpus, with reduced model size (2 layers, `d_model=64`, `num_heads=2`). it contains:

- Building a vocabulary and tokenizing sentences
- Implementing positional encodings
- Scaled dot‑product attention and multi‑head layers
- Encoder and decoder layers (with masked self‑attention and cross‑attention)
- A custom learning‑rate schedule matching the original paper
- Training loop with padding and look‑ahead masks
- New dataset derived from the 4.5M English–German WMT14 corpus

---

## Notebook Sections

1. **Imports & Tokenization**
    
    Load TensorFlow, NumPy, and prepare `tokenized_sentences`, `word_to_idx`, and `vocab_size`.
    
2. **CustomSchedule**
    
    Define the warm‑up learning rate scheduler.
    
3. **Attention & Layers**
    
    Implement `scaled_dot_product_attention`, `MultiHeadAttention`, feed‑forward networks, `EncoderLayer`, and `DecoderLayer` (both self‑ and cross‑attention).
    
4. **Model Definition**
    
    Build `Encoder`, `Decoder`, and `Transformer` classes along with positional encoding and mask utilities.
    
5. **Data Pipeline**
    
    Pad sequences, create input/target pairs, and assemble a `tf.data.Dataset`.
    
6. **Training**
    
    Set up loss, metrics, `@tf.function train_step`, and run the epoch loop with per‑batch logging.
    
7. **Results & Debugging**
    
    Inspect flat loss/accuracy, examine logits, and apply sanity checks.
    

---

## How to Use

1. **Open in Colab**: Click the badge below to launch the notebook.
    
    ![](https://colab.research.google.com/assets/colab-badge.svg)
    
2. **Select Runtime**: Choose **Runtime ▶️ Change runtime type** and enable **GPU**.
3. **Run All Cells**: Execute the notebook from top to bottom.
4. **Customize**: Modify corpus data, adjust `max_length`, or experiment with hyperparameters in the **Hyperparameters** cell.

---

## Runtime Requirements

- **Python 3.8+** (pre‑installed on Colab)
- **TensorFlow 2.x** and **NumPy** (already available in Colab)
- **GPU runtime** recommended for faster training (optional)

---

## Hyperparameters

| Parameter | Default | Description |
| --- | --- | --- |
| `num_layers` | 2 | Number of encoder/decoder layers |
| `d_model` | 64 | Embedding and hidden size |
| `num_heads` | 2 | Attention heads |
| `dff` | 256 | Feed‑forward hidden size |
| `batch_size` | 8 | Examples per batch |
| `EPOCHS` | 61 | Training epochs |

---

## Troubleshooting & Tips

- **Flat Loss/Accuracy**: Verify that your `DecoderLayer` has both masked self‑attention and cross‑attention with correct masks.
- **Mask Creation**: Use `create_masks(inp, tar)`—not `create_masks(inp, inp)`—so decoder sees the look‑ahead mask on targets.
- **Metrics Reset**: Call `reset_states()`, not `reset_state()`, on TensorFlow metrics.
- **Sanity Checks**: After `train_step`, print a few logits (e.g., `predictions[0, :5, :]`) to confirm they're not uniform.

---

## Acknowledgments

Based on “Attention Is All You Need” by Vaswani et al., 2017.

Inspired by community TensorFlow tutorials and guided debugging tips.

---

*Enjoy exploring Transformers!*

