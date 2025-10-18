# CS 6180 Lecture 9: Attention Mechanisms
**Date:** October 1, 2025  
**Topic:** Introduction to Attention for Neural Machine Translation

---

## Overview

This lecture introduces the **attention mechanism**, a breakthrough that addressed critical limitations in sequence-to-sequence models for tasks like machine translation. Attention allows models to focus on relevant parts of the input when generating each output word, rather than compressing all information into a single fixed vector.

---

## 1. Neural Machine Translation (NMT)

### Task Definition
**Goal:** Generate a sentence in one language from a sentence in another language.

**Example:**
- **French:** "Il faut profiter du moment présent"
- **English:** "You have to enjoy the present moment"
- **Latin:** "Carpe Diem"

### Mathematical Objective
Learn a model that computes:

$$P(Y | X)$$

Where:
- $X$ = source sentence (e.g., French)
- $Y$ = target sentence (e.g., English)

**Training objective:** Maximize this conditional probability across all sentence pairs in the training data.

---

## 2. Encoder-Decoder Architecture

### Basic Structure

Neural Machine Translation uses a **sequence-to-sequence (seq2seq)** architecture with two main components:

```
Source Sentence → [ENCODER] → Context Vector → [DECODER] → Target Sentence
```

### Encoder (Encoder RNN)

**Purpose:** Process the source sentence and compress it into a fixed-size representation.

**Example:** "Je m'appelle Nadim" (French)

```
Input:    Je    m'appelle    Nadim
          ↓         ↓          ↓
         e^(1)    e^(2)      e^(3)
          ↓         ↓          ↓
        h^(0) → h^(1) → h^(2) → h^(3)
                                  ↓
                          [Context Vector]
```

**Mathematical formulation:**

$$\vec{h}^{(t)} = \text{RNN}_{\text{enc}}(\vec{h}^{(t-1)}, \vec{e}^{(t)})$$

The final hidden state $\vec{h}^{(n)}$ (shown in pink box above) serves as the **context vector** that summarizes the entire source sentence.

### Decoder (Decoder RNN)

**Purpose:** Generate the target sentence word-by-word, conditioned on the context vector.

**Example:** Generating "My name is Nadim"

```
[Context] → [Decoder] → [Decoder] → [Decoder] → [Decoder]
   ↓           ↓           ↓           ↓           ↓
 <START>      My         name         is        Nadim
              ↓           ↓           ↓           ↓
             My         name         is        Nadim       <END>
```

**Mathematical formulation:**

$$\vec{s}^{(t)} = \text{RNN}_{\text{dec}}(\vec{s}^{(t-1)}, \vec{y}^{(t-1)}, \vec{c})$$

$$P(y^{(t)} | y^{(1)}, \ldots, y^{(t-1)}, X) = \text{softmax}(W_s \vec{s}^{(t)} + \vec{b})$$

Where:
- $\vec{s}^{(t)}$ = decoder hidden state at time $t$
- $\vec{y}^{(t-1)}$ = previously generated word
- $\vec{c}$ = context vector from encoder
- $W_s, \vec{b}$ = output projection parameters

---

## 3. The Bottleneck Problem

### Key Limitation of Basic Seq2Seq

**Problem:** The entire source sentence's information is compressed into a **single fixed-size context vector** (the final encoder hidden state).

**Consequences:**
1. **Information loss:** Long sequences lose critical details
2. **No selective focus:** Decoder cannot emphasize relevant source words for each target word
3. **Poor performance on long sentences:** Performance degrades significantly as sentence length increases

**Historical context:** These were state-of-the-art for translation around 2017, but had clear limitations.

### Additional Limitations
- **Not parallelizable:** Sequential processing makes computation expensive
- **Vanishing gradients:** Despite LSTMs, very long sequences still problematic

**Key insight:** We need a mechanism that allows the decoder to "look back" at the source sentence and focus on relevant parts dynamically.

---

## 4. Attention Mechanism: The Solution

### Core Idea

Instead of using only the final encoder hidden state, allow the decoder to create a **weighted combination** of **all** encoder hidden states at each decoding step.

**Intuition:** When translating word $i$ in the target, the model should "attend to" relevant words in the source.

### High-Level Process

For each decoder time step:
1. **Compare** the current decoder state with all encoder hidden states
2. **Compute attention scores** indicating relevance
3. **Create a weighted combination** of encoder states (the "context vector")
4. **Use this context** to generate the next word

---

## 5. Attention Mechanism: Mathematical Details

### Setup

Consider translating "Wo jiao Nadim" (Chinese: "My name is Nadim") to English.

**Encoder hidden states:**
- $\vec{e}_1$ for "Wo" (我)
- $\vec{e}_2$ for "jiao" (叫)  
- $\vec{e}_3$ for "Nadim"

**Decoder state:** $\vec{q}_i$ when generating word $i$

### Step 1: Compute Attention Scores

For each encoder hidden state $\vec{e}_j$, compute a score indicating how relevant it is to the current decoder state $\vec{q}_i$:

$$\text{score}(\vec{q}_i, \vec{e}_j) = \vec{q}_i^T \vec{e}_j$$

This is a **dot product** measuring similarity between the query (decoder state) and each encoder state.

**Example scores:**
- $\vec{q}_i^T \vec{e}_1$ = relevance of "Wo" to current target word
- $\vec{q}_i^T \vec{e}_2$ = relevance of "jiao" to current target word
- $\vec{q}_i^T \vec{e}_3$ = relevance of "Nadim" to current target word

### Step 2: Normalize with Softmax (Attention Weights)

Convert scores to a probability distribution:

$$\alpha_j = \frac{\exp(\vec{q}_i^T \vec{e}_j)}{\sum_{k=1}^{n} \exp(\vec{q}_i^T \vec{e}_k)}$$

Where:
- $\alpha_j$ = attention weight for encoder position $j$
- $n$ = source sentence length
- $\sum_{j=1}^{n} \alpha_j = 1$ (weights sum to 1)

**Interpretation:** $\alpha_j$ represents "how much attention" to pay to source word $j$ when generating the current target word.

### Step 3: Compute Context Vector

Create a weighted sum of encoder hidden states:

$$\vec{c}_i = \sum_{j=1}^{n} \alpha_j \vec{e}_j = \alpha_1 \vec{e}_1 + \alpha_2 \vec{e}_2 + \alpha_3 \vec{e}_3$$

This context vector $\vec{c}_i$ is **specific to decoding step $i$** and focuses on relevant source words.

### Step 4: Generate Output

Use the context vector in decoding:

$$\vec{s}^{(i)} = \text{RNN}_{\text{dec}}(\vec{s}^{(i-1)}, \vec{y}^{(i-1)}, \vec{c}_i)$$

$$P(y^{(i)} | y^{(1)}, \ldots, y^{(i-1)}, X) = \text{softmax}(W[\vec{s}^{(i)}; \vec{c}_i] + \vec{b})$$

Note: The context vector $\vec{c}_i$ changes at each decoding step!

---

## 6. Query-Key-Value Framework

### Motivation: Separating Responsibilities

**Problem with using raw embeddings $\vec{e}_i$ directly:**

Word embeddings are being asked to serve multiple purposes:
1. **Semantic meaning** of the word
2. **Similarity to other words** (in same language)
3. **Cross-lingual similarity** (for translation)
4. **Attention computation** (relevance scoring)

This is too much responsibility for a single representation!

### Solution: Learn Specialized Transformations

Transform each encoder hidden state $\vec{e}_i$ into three specialized representations:

$$\vec{K}_i = K \vec{e}_i \quad \text{(Key)}$$
$$\vec{Q}_i = Q \vec{e}_i \quad \text{(Query)}$$
$$\vec{V}_i = V \vec{e}_i \quad \text{(Value)}$$

Where:
- $K \in \mathbb{R}^{d \times d}$ = Key transformation matrix (learnable parameter)
- $Q \in \mathbb{R}^{d \times d}$ = Query transformation matrix (learnable parameter)
- $V \in \mathbb{R}^{d \times d}$ = Value transformation matrix (learnable parameter)

### Roles of Each Component

| Component | Role | Usage |
|-----------|------|-------|
| **Query** ($\vec{Q}_i$) | "What am I looking for?" | Represents the decoder's current focus |
| **Key** ($\vec{K}_i$) | "What do I contain?" | Represents what each encoder state offers |
| **Value** ($\vec{V}_i$) | "What information do I provide?" | The actual content to be retrieved |

### Updated Attention Computation

**Step 1: Compute scores using Keys and Queries**

$$\text{score}(\vec{Q}_i, \vec{K}_j) = \vec{Q}_i^T \vec{K}_j = (\mathbf{Q} \vec{q}_i)^T (\mathbf{K} \vec{e}_j) = \vec{q}_i^T \mathbf{Q}^T \mathbf{K} \vec{e}_j$$

**Step 2: Normalize**

$$\alpha_j = \frac{\exp(\vec{Q}_i^T \vec{K}_j)}{\sum_{k=1}^{n} \exp(\vec{Q}_i^T \vec{K}_k)}$$

**Step 3: Weighted sum using Values**

$$\vec{c}_i = \sum_{j=1}^{n} \alpha_j \vec{V}_j$$

### Why This Separation Helps

**Computational efficiency insight:**

The dot product can be rewritten as:

$$\vec{Q}_i^T \vec{K}_j = \vec{q}_i^T \underbrace{\mathbf{Q}^T \mathbf{K}}_{\mathbf{M}} \vec{e}_j$$

We can **pre-compute** the matrix $\mathbf{M} = \mathbf{Q}^T \mathbf{K}$ once:
- $\mathbf{M} \in \mathbb{R}^{d \times d}$
- This has $d^2$ parameters

**Parameter comparison:**

| Approach | Number of Parameters |
|----------|---------------------|
| Separate $Q$ and $K$ matrices | $2d \cdot d_Q$ |
| Pre-computed $M = Q^T K$ | $d^2$ |
| **Advantage** | When $d_Q \ll d$: $2dd_Q < d^2$ |

**Example:** If $d = 512$ and $d_Q = 64$:
- Separate: $2 \times 512 \times 64 = 65,536$ parameters
- Combined: $512^2 = 262,144$ parameters
- **Savings:** ~4x fewer parameters!

This separation leads to:
- **Fewer parameters** to learn
- **More efficient computation**
- **Specialized representations** for different purposes

---

## 7. General Applicability of Attention

### Beyond Translation

**Important note:** Attention is not limited to machine translation!

Attention can be used for:
- **Language modeling:** Attending to previous words when predicting the next
- **Question answering:** Attending to relevant passage segments
- **Summarization:** Focusing on key sentences
- **Any sequence-to-sequence task**

The key requirement is having a **query** and a set of **key-value pairs** to attend over.

---

## 8. Bidirectional RNNs/LSTMs

### Motivation

**Problem with unidirectional RNNs:** When processing "The movie is fantastically horrible," the hidden state at "fantastically" doesn't know about "horrible" yet.

**Solution:** Process the sequence in **both directions**.

### Architecture

For each word at position $t$, combine information from both directions:

**Forward LSTM:** $\vec{h}^{(t)}_{\rightarrow}$
- Processes: word 1 → word 2 → ... → word $t$
- Captures left context

**Backward LSTM:** $\vec{h}^{(t)}_{\leftarrow}$
- Processes: word $n$ → ... → word $t+1$ → word $t$
- Captures right context

**Combined representation:**

$$\vec{h}^{(t)} = \begin{bmatrix} \vec{h}^{(t)}_{\rightarrow} \\ \vec{h}^{(t)}_{\leftarrow} \end{bmatrix}$$

This concatenation provides full context (both past and future) for each word.

### Mathematical Formulation

**Forward pass:**
$$\vec{h}^{(t)}_{\rightarrow} = \text{LSTM}_{\rightarrow}(\vec{h}^{(t-1)}_{\rightarrow}, \vec{e}^{(t)})$$

**Backward pass:**
$$\vec{h}^{(t)}_{\leftarrow} = \text{LSTM}_{\leftarrow}(\vec{h}^{(t+1)}_{\leftarrow}, \vec{e}^{(t)})$$

**Final representation:**
$$\vec{h}^{(t)} = [\vec{h}^{(t)}_{\rightarrow}; \vec{h}^{(t)}_{\leftarrow}]$$

### When to Use Bidirectional LSTMs

| Use Case | Can Use Bi-LSTM? | Reason |
|----------|------------------|--------|
| **Encoder** | ✅ Yes | Full source sentence available |
| **Decoder** | ❌ No | Cannot see future target words |
| **Language Model** | ❌ No | Must predict sequentially |
| **Sentence Classification** | ✅ Yes | Full sentence available |

**Key principle:** Bidirectional processing requires access to the entire sequence, which is fine for encoding but violates causality for generation.

---

## Key Takeaways

### 1. The Bottleneck Problem
- Basic seq2seq compresses entire source into one vector
- Performance degrades on long sentences
- Critical information can be lost

### 2. Attention as Solution
- Allows dynamic focus on relevant source positions
- Different context vector for each decoding step
- Dramatically improves long-sentence translation

### 3. Attention Computation
- **Scores:** Measure relevance (dot product)
- **Weights:** Normalize with softmax
- **Context:** Weighted combination of encoder states

### 4. Query-Key-Value Framework
- Separates responsibilities of embeddings
- More efficient computation
- Specialized transformations for matching and retrieval

### 5. Computational Efficiency
- Pre-computing $M = Q^T K$ saves computation
- Fewer parameters when $d_Q < d$
- Critical for scaling to large models

### 6. Bidirectional Processing
- Provides full context for each word
- Essential for encoders
- Cannot be used for decoders or language models

---

## Mathematical Notation Legend

### Variables
- $X$ = source sentence
- $Y$ = target sentence  
- $\vec{e}_i$ = encoder hidden state at position $i$
- $\vec{q}_i$ = decoder query state at position $i$
- $\vec{s}^{(t)}$ = decoder hidden state at time $t$
- $\vec{c}_i$ = context vector at decoding step $i$

### Attention Components
- $\vec{K}_i$ = key vector for position $i$
- $\vec{Q}_i$ = query vector for position $i$
- $\vec{V}_i$ = value vector for position $i$
- $\alpha_j$ = attention weight for position $j$
- $\text{score}(\cdot, \cdot)$ = attention scoring function

### Transformations
- $K, Q, V$ = learnable transformation matrices
- $\mathbf{M} = Q^T K$ = pre-computed attention matrix
- $W_s$ = output projection matrix

### Dimensions
- $d$ = embedding dimension
- $d_Q$ = query/key dimension (often smaller than $d$)
- $n$ = source sequence length
- $m$ = target sequence length

### Operators
- $\cdot^T$ = transpose
- $[\cdot; \cdot]$ = concatenation
- $\exp(\cdot)$ = exponential function
- $\sum$ = summation