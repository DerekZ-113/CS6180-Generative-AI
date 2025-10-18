# CS 6180 Lecture 10: Self-Attention and Transformer Decoder
**Date:** October 8, 2025  
**Topic:** From Cross-Attention to Self-Attention & Building the Transformer Decoder

**Administrative Note:** HW2 released!

---

## Overview

This lecture transitions from cross-attention (between encoder and decoder) to **self-attention** (within the same sequence), introducing the key components needed to build a complete Transformer decoder architecture. The lecture addresses critical challenges: position encoding, masking for causality, and the need for non-linearities.

---

## 1. Review: Cross-Attention in Translation

### Recap from Lecture 9

In neural machine translation, we use **cross-attention** between different models:

```
Source Sentence → [Encoder LSTM] → [Decoder LSTM] → Target Sentence
                        ↓                    ↑
                   Attention weights
```

**Key idea:** Put attention on the source sentence to generate each word in the target sentence.

### Query-Key-Value Framework (Review)

For each encoder hidden state $\vec{e}_i$:

$$\vec{K}_i = K\vec{e}_i \quad \text{(Key representation)}$$
$$\vec{V}_i = V\vec{e}_i \quad \text{(Value representation)}$$

For decoder state, we have the query:

$$\vec{q}_i = Q\vec{e}_i \quad \text{(Query from decoder)}$$

**Cross-attention computation:**

1. **Compute similarity scores:**
   $$s_{it} = \vec{K}_i^T \vec{q}_t$$

2. **Apply softmax to get attention weights:**
   $$\alpha_{it} = \frac{\exp(s_{it})}{\sum_{j=1}^{m} \exp(s_{jt})}$$

3. **Compute attention output:**
   $$\vec{a}_t = \sum_{i=1}^{m} \alpha_{it} \vec{V}_i$$

4. **Use in decoder:**
   $$\vec{h}_t^{\text{dec}}, C_t = \text{LSTM}(\vec{h}_{t-1}^{\text{dec}}, \vec{C}_{t-1}, \vec{e}_t)$$

The attention output $\vec{a}_t$ is concatenated with decoder hidden state: $[\vec{a}_t; \vec{h}_t^{\text{dec}}]$

---

## 2. Why Attention Helps: Interaction Distance

### The Problem: Information Propagation in Seq2Seq

**Without attention:**
- Information flows from source sentence to target through a **single fixed vector** (the final encoder hidden state)
- Distance between relevant words in source and target: **O(length of source sentence)**

**Example:**
```
French:  "Je m'appelle Nadim et j'aime le chocolat beaucoup"
English: "My name is Nadim and I love chocolate a lot"
         ↑                           ↑
    (position 2)              (position 7)
```

To translate "chocolate" correctly, information must propagate through the entire source sentence encoding.

### With Attention: Direct Access

**Interaction distance: O(1)**

```
Source:  Je  m'appelle  Nadim  et  j'aime  le  chocolat  beaucoup
         ↓      ↓        ↓     ↓     ↓     ↓      ↓        ↓
         ╰──────┴────────┴─────┴─────┴─────┴──────┴────────╯
                            ↓
                   (Attention weights)
                            ↓
Target:  My  name  is  Nadim  and  I  love  chocolate  a  lot
```

**Key insight:** The decoder can get **direct information from every source word** at each generation step. The distance between "chocolate" in source and target is now constant!

### Dual Role of Attention

Attention helps with:
1. **Cross-sentence information flow:** Accessing source words when generating target words
2. **Within-sentence information flow:** Propagating information from previous words (this motivates RNNs/LSTMs)

**Key question:** Why not give up on RNNs/LSTMs entirely and build models based purely on attention?

---

## 3. Self-Attention: The Core Innovation

### Motivation

Instead of attention **between** encoder and decoder, apply attention **within** the same sentence.

### Self-Attention Framework

Given a sentence with words $\vec{x}_1, \vec{x}_2, \vec{x}_3, \ldots, \vec{x}_m$:

```
Input:    x₁      x₂      x₃  ...  xₘ
          ↓       ↓       ↓        ↓
Embed:   e₁      e₂      e₃  ...  eₘ
```

For each word embedding $\vec{e}_i$, create three representations:

$$\vec{q}_i = Q\vec{e}_i \quad \text{(Query)}$$
$$\vec{K}_i = K\vec{e}_i \quad \text{(Key)}$$
$$\vec{V}_i = V\vec{e}_i \quad \text{(Value)}$$

**Self-attention computation:**

1. **Compute attention scores:**
   $$s_{ij} = \vec{q}_i^T \vec{K}_j$$

2. **Normalize with softmax:**
   $$\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{k=1}^{m} \exp(s_{ik})}$$

3. **Compute output:**
   $$\vec{a}_i = \sum_{j=1}^{m} \alpha_{ij} \vec{V}_j$$

**Key difference from cross-attention:** All queries, keys, and values come from the **same sequence**.

---

## 4. Two Critical Issues with Self-Attention

### Issue 1: Position Information Lost

**Problem:** The order (position) of words is **not taken into account** in self-attention.

**Example demonstrating the problem:**

```
Sentence 1: "Zuko made his uncle some tea"
Sentence 2: "His uncle made Zuko some tea"
```

- **Same words** in both sentences
- **Different meanings** (who made the tea for whom?)
- With only self-attention: Output would be identical for "Zuko's" in both sentences

**Why this matters:** 
If we're only using self-attention, the output at position corresponding to "Zuko's" should be the same in both sentences. We don't want the model to learn this! The model needs to distinguish between these sentences.

### Issue 2: Cannot Peek at Future Words

**Problem:** For language modeling, we must ensure **causality** - the model cannot use information from future words when predicting the current word.

**Example:**
```
"Zuko made his uncle tea at Zuko's place"
      ↑                         ↑
  (position 2)            (position 7)
```

When generating word at position 2, we cannot see positions 3-7.

---

## 5. Solution to Issue 1: Positional Encoding

### Adding Position Information

**Solution:** Add position vectors to embeddings.

For input word $\vec{x}_i$ at position $i$:

$$\vec{x}_i \rightarrow \vec{e}_i + \vec{p}_i$$

Where:
- $\vec{e}_i$ = word embedding
- $\vec{p}_i$ = position vector for position $i$

**Final input representation:**

$$\vec{x}_1 \rightarrow \vec{e}_1 + \vec{p}_1$$
$$\vec{x}_2 \rightarrow \vec{e}_2 + \vec{p}_2$$
$$\vdots$$
$$\vec{x}_m \rightarrow \vec{e}_m + \vec{p}_m$$

### Why This Works: Distinguishing Similar Sentences

**Example revisited:**

```
Sentence 1: Zuko made his uncle tea at Zuko's place
Position:     1    2    3    4     5  6    7     8

Sentence 2: His uncle made Zuko tea at Iroh's place
Position:    1    2     3    4    5  6    7      8
```

**Without position encoding:**
- Both sentences have same words
- Self-attention outputs would be identical at "Zuko's" and "Iroh's"

**With position encoding:**
- Position information is embedded in the representation
- "Zuko" at position 1 has different representation than "Zuko" at position 4
- Model has **flexibility to learn the difference** between the sentences
- Adding position parameters gives the model capacity to distinguish word order

---

## 6. Solution to Issue 2: Masking

### The Masking Mechanism

**Goal:** Prevent the model from attending to future words during training.

### How Masking Works

When computing attention for position $t$, set future attention scores to $-\infty$:

$$s_{it} = \begin{cases} 
\vec{q}_t^T \vec{K}_i & \text{if } i \leq t \\
-\infty & \text{if } i > t
\end{cases}$$

**Example at position $t=4$:**

```
Sentence: Zuko made his uncle tea at Zuko's place
Position:   1    2    3    4    5  6    7     8
            ↓    ↓    ↓    ↓
Scores:   s₁₄  s₂₄  s₃₄  s₄₄  -∞  -∞   -∞    -∞
```

### Mathematical Implementation

**Attention scores before masking:**
$$s_{1t}, s_{2t}, s_{3t}, s_{4t}, s_{5t}, s_{6t}, s_{7t}$$

**After masking (at position $t=4$):**
$$s_{1t}, s_{2t}, s_{3t}, s_{4t}, -\infty, -\infty, -\infty$$

**After softmax:**

$$\alpha_{it} = \frac{\exp(s_{it})}{\sum_{j=1}^{t} \exp(s_{jt})}$$

Since $\exp(-\infty) = 0$, future positions contribute nothing:

$$\alpha_{it} = \begin{cases}
\frac{\exp(s_{it})}{\sum_{j=1}^{t} \exp(s_{jt})} & \text{if } i \leq t \\
0 & \text{if } i > t
\end{cases}$$

**Attention output (no future information):**

$$\vec{a}_t = \sum_{i=1}^{t} \alpha_{it} \vec{V}_i$$

This ensures we are **not peeking** at future words!

---

## 7. Building the Transformer Decoder

### Architecture Overview

```
Input: x
   ↓
[Embedding E]
   ↓
e + p (add position)
   ↓
┌─────────────────────┐
│  Masked Self-       │
│  Attention          │
└─────────────────────┘
   ↓
[Linear Projection]
   ↓
┌─────────────────────┐
│  Multi-Layer        │
│  Perceptron (MLP)   │
└─────────────────────┘
   ↓
[Repeat unit several times]
   ↓
[Linear Projection]
   ↓
[Softmax]
   ↓
Output
```

### Component Details

#### 1. Masked Self-Attention
- Applies self-attention with masking
- Ensures causal structure (no future information)

#### 2. Multi-Layer Perceptron (Feed-Forward Network)

After self-attention, apply a two-layer MLP to each position:

$$\vec{h}^{(1)} = \sigma(W^{(1)} \vec{x} + \vec{b}^{(1)})$$

$$\vec{h}^{(2)} = \sigma(W^{(2)} \vec{h}^{(1)} + \vec{b}^{(2)})$$

Where $\sigma$ is a non-linear activation (typically ReLU or GeLU).

**Why MLPs?** Need non-linearities to handle complex data and learn intricate patterns.

#### 3. Linear Projection + Softmax

Final output layer:
- Projects hidden state to vocabulary size
- Applies softmax to get probability distribution over next word

### Stacking Decoder Blocks

The unit (Masked Self-Attention + MLP) is **repeated several times** (typically 6-12+ layers in modern Transformers).

**This is (almost) the Transformer Decoder architecture!**

---

## 8. Three Main Components for Full Transformer

To get the complete Transformer architecture, we need three additional components:

### Component 1: Multi-Head Attention

**Motivation:** Different attention heads can capture different linguistic phenomena.

Instead of one set of $(Q, V, K)$ matrices, use **multiple attention heads**:
- Head 1: Focus on **meaning/semantics**
- Head 2: Focus on **grammar**
- Head 3: Focus on **syntax**
- And so on...

**Each head learns different attention patterns**, providing richer representations.

### Component 2: Residual Connections (Skip Connections)

Add the input of each layer to its output:

$$\text{output} = \text{LayerFunction}(\text{input}) + \text{input}$$

**Benefits:**
- Helps with gradient flow during training
- Allows the model to learn identity mappings if needed
- Improves training stability

### Component 3: Layer Normalization

Normalize activations across features at each layer:

$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta$$

**Benefits:**
- Stabilizes training
- Allows for higher learning rates
- Reduces internal covariate shift

---

## Key Insights

### 1. Self-Attention as Generalization
**Cross-attention** (Lecture 9) attends between encoder and decoder.
**Self-attention** (this lecture) attends within the same sequence.
Both use the same Query-Key-Value framework!

### 2. Position Encoding Necessity
Without position information, self-attention is **permutation invariant** - it treats the input as a set rather than a sequence. Position encoding restores sequence ordering.

### 3. Masking for Causality
Masking is essential for autoregressive generation - ensuring each position can only attend to previous positions, maintaining the causal structure needed for language modeling.

### 4. Why Give Up RNNs?
**Advantages of pure attention:**
- **Parallelization:** All positions can be computed simultaneously (unlike sequential RNNs)
- **Direct connections:** O(1) path between any two positions (vs. O(n) in RNNs)
- **Better gradient flow:** No vanishing gradients from long sequential chains

### 5. MLPs Add Expressiveness
Self-attention is linear in the value vectors. MLPs add the **non-linear transformations** needed to learn complex functions and representations.

---

## From Seq2Seq to Transformer Decoder

### Evolution of Architecture

**Seq2Seq with Attention:**
```
Encoder RNN → Context → Decoder RNN
                ↓           ↑
            Attention
```

**Transformer Decoder (Decoder-only):**
```
Input → Position Encoding → [Masked Self-Attention + MLP]ₓₙ → Output
```

**Key differences:**
- No RNNs - pure attention
- Self-attention replaces both encoder and decoder RNNs
- Massively parallelizable
- Scales much better to longer sequences

---

## Mathematical Notation Summary

### Self-Attention Components
- $\vec{e}_i$ = word embedding at position $i$
- $\vec{p}_i$ = position encoding at position $i$
- $\vec{q}_i$ = query vector for position $i$
- $\vec{K}_i$ = key vector for position $i$
- $\vec{V}_i$ = value vector for position $i$

### Attention Computation
- $s_{ij}$ = attention score between positions $i$ and $j$
- $\alpha_{ij}$ = attention weight (after softmax)
- $\vec{a}_i$ = attention output at position $i$

### Masking
- $s_{it} = -\infty$ for $i > t$ (mask future positions)
- $\alpha_{it} = 0$ for $i > t$ (after softmax)

### Dimensions
- $d$ = embedding dimension
- $m$ = sequence length
- $Q, K, V \in \mathbb{R}^{d \times d}$ = transformation matrices

### MLP Parameters
- $W^{(1)}, W^{(2)}$ = weight matrices for feed-forward layers
- $\vec{b}^{(1)}, \vec{b}^{(2)}$ = bias vectors
- $\sigma$ = activation function (ReLU or GeLU)

---

## Critical Takeaways

1. **Self-attention** allows each word to attend to all other words in the same sequence
2. **Position encoding** is essential because self-attention has no inherent notion of word order
3. **Masking** ensures causality in autoregressive generation
4. **MLPs** provide the non-linearity needed for complex function learning
5. **Pure attention architectures** eliminate RNNs, enabling parallelization and better scaling
6. The Transformer decoder is built by stacking [Masked Self-Attention + MLP] blocks
7. We're three components away from the full Transformer: multi-head attention, residual connections, and layer normalization