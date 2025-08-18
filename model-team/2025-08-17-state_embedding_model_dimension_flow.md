# STATE Model: Data Dimension Flow for a Single Cell Embedding

This document summarizes the transformation of data dimensions when processing a single cell to generate a final embedding, based on the `SE600M` model configuration.

## 1. Raw Input

The process begins with two vectors for each cell, representing the top expressed genes (up to 2047) plus a special `[CLS]` token at the beginning, making the sequence length 2048.

-   **Gene Indices**: A vector of integers identifying each gene in the model's vocabulary.
-   **Expression Values**: A vector of floats representing the expression level for each corresponding gene.

For a batch of cells, the input consists of two tensors:

-   `gene_indices`: `[batch_size, 2048]`
-   `expression_values`: `[batch_size, 2048]`

## 2. Embedding Layer

The model uses the `gene_indices` to look up the corresponding gene embeddings from its vocabulary. Each gene has a pre-trained embedding of size 5120 (`tokenizer.token_dim`). These embeddings are then scaled by their respective `expression_values`.

This step results in a single, high-dimensional tensor:

-   **Shape**: `[batch_size, 2048, 5120]`

## 3. Linear Projection to Model Dimension

The `5120`-dimensional vectors are too large for the main Transformer model to process efficiently. A linear layer projects (reduces) the dimension from `5120` down to the model's internal working dimension, which is `2048` (`model.emsize`).

This is the tensor that enters the core Transformer layers:

-   **Shape**: `[batch_size, 2048, 2048]`

## 4. Transformer Encoder Output

The `[batch_size, 2048, 2048]` tensor is processed through the 16 layers of the Transformer. The self-attention mechanism allows the `[CLS]` token to aggregate information from all other gene tokens.

The output of the Transformer is a tensor of the same shape, but with contextualized information embedded in each token's vector.

-   **Shape**: `[batch_size, 2048, 2048]`

## 5. Final Embedding Extraction

To get the final representation for each cell, we discard the outputs for all the actual gene tokens and **only take the output vector corresponding to the `[CLS]` token** (which is the first token in the sequence).

This extraction process reduces the matrix to a single vector per cell. The dimension of this vector is defined by `model.output_dim`.

-   **Final Cell Embedding Shape**: `[batch_size, 2048]`
