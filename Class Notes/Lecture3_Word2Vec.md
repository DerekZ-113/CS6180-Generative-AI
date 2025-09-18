# CS 6180 - Lecture 3: Word2Vec

## Overview
Word2Vec is a technique for learning word embeddings from large corpora of text using neural networks.

## Core Ideas & Insights

### 1. The Fundamental Principle
**"You shall know a word by the company it keeps"** - Words with similar meanings appear in similar contexts. Word2Vec operationalizes this linguistic insight into a learning algorithm.

### 2. The Two-Vector Architecture
Every word maintains **two distinct representations**:
- **Center vector** ($\vec{v}$): Used when the word is the focus word
- **Context vector** ($\vec{u}$): Used when the word appears in another word's context

This design enables asymmetric relationships (e.g., "drink water" vs. "water drink" can have different probabilities).

### 3. The Learning Objective
Train by predicting context words from center words using softmax probability:
$P(O = o | C = c) = \frac{\exp(\vec{u}_o^T \vec{v}_c)}{\sum_{w \in V} \exp(\vec{u}_w^T \vec{v}_c)}$

The dot product measures how well words "fit" together in context.

### 4. The "Observed minus Expected" Principle  
The gradient elegantly reduces to:
- **Pull** vectors toward what was actually observed in the data
- **Push** away from what the model currently expects on average

This principle appears throughout machine learning and makes training intuitive.

### 5. The Computational Challenge
Normalizing requires summing over the entire vocabulary for every update - this motivates approximations like negative sampling in practice.

### 6. Why This Matters for Gen AI
Word2Vec demonstrated how to:
- Convert discrete symbols into continuous, learnable vectors
- Learn rich representations from unlabeled text
- Capture semantic relationships mathematically

These embeddings became the foundation for RNNs, Transformers, and all modern language models.

## Core Concepts

### Model Setup
- **Input**: Large corpus of text
- **Fixed window approach**: 
  - Center word (C)
  - Context words (O) outside

### Probability Model
The model computes: $P(O = o | C = c)$

### Design Choices for Word Vectors

**Two vector approach**:
- For every word $w$, we maintain:
  - $\vec{u}_w$: when $w$ is a center word
  - $\vec{v}_w$: when $w$ is a context (outside) word

**Why two vectors?**
- If we use one vector for both roles:
  - $P(O = \text{"ice"} | C = \text{"cold"})$ and $P(O = \text{"cold"} | C = \text{"ice"})$ would be forced to be the same
  - This constraint is undesirable as these probabilities might naturally differ

### Mathematical Formulation

The probability of observing context word $o$ given center word $c$:

$P(O = o | C = c) = \frac{\exp(\vec{u}_o^T \vec{v}_c)}{\sum_{w \in V} \exp(\vec{u}_w^T \vec{v}_c)}$

Where:
- $V$ = vocabulary
- $\vec{u}_o$ = context vector for word $o$
- $\vec{v}_c$ = center vector for word $c$
- $|V| = K$ (size of vocabulary)

### Vector Representations

We need to learn:
- Center vectors: $\vec{v}_1, \vec{v}_2, \ldots, \vec{v}_K$ (each of length $d$)
- Context vectors: $\vec{u}_1, \vec{u}_2, \ldots, \vec{u}_K$ (each of length $d$)

Example vocabulary mapping:
- "aardvark" $\rightarrow 1$
- "zebra" $\rightarrow K$

## Likelihood Function

For a corpus with $T$ words:

$L = \prod_{t=1}^{T} \prod_{\substack{-4 \leq j \leq 4 \\ j \neq 0}} P(O = w_{t+j} | C = w_t)$

We want to maximize this likelihood function.

### Log-Likelihood

Applying log (which is monotonically increasing and doesn't change the optimum):

$\log L = \sum_{t=1}^{T} \sum_{\substack{-4 \leq j \leq 4 \\ j \neq 0}} \log P(O = w_{t+j} | C = w_t)$

## Gradient Computation

### Derivative with respect to $\vec{v}_c$

Starting with:
$\frac{\partial}{\partial \vec{v}_c} \log P(O = o | C = c)$

Through derivation:

$= \frac{\partial}{\partial \vec{v}_c} \left[ \vec{u}_o^T \vec{v}_c - \log\left(\sum_{w \in V} \exp(\vec{u}_w^T \vec{v}_c)\right) \right]$

$= \vec{u}_o - \sum_{w \in V} \vec{u}_w \cdot \frac{\exp(\vec{u}_w^T \vec{v}_c)}{\sum_{x \in V} \exp(\vec{u}_x^T \vec{v}_c)}$

$= \vec{u}_o - \sum_{w \in V} \vec{u}_w \cdot P(O = w | C = c)$

This gives us the elegant gradient form:

$\boxed{\frac{\partial}{\partial \vec{v}_c} \log P(O = o | C = c) = \vec{u}_o - \sum_{w \in V} \vec{u}_w \cdot P(O = w | C = c)}$

$\text{[observed] - [expected]}$

### Interpretation

The gradient has the form: **observed - expected**

When gradient = 0:
$\vec{u}_o = \sum_{w \in V} \vec{u}_w \cdot P(O = w | C = c)$

This means at the optimum, the observed context vector equals the expected context vector under the model's probability distribution.

## Key Insights

1. **Two-vector design**: Separating center and context vectors provides more modeling flexibility
2. **Softmax normalization**: The exponential and normalization create a valid probability distribution
3. **Gradient structure**: The "observed minus expected" form is common in maximum likelihood estimation
4. **Computational challenge**: The normalization term requires summing over the entire vocabulary (K terms), making it computationally expensive for large vocabularies

## Notes
- The scalar term in the derivatives (e.g., $\frac{\partial}{\partial \vec{v}_c}$ of scalar expressions) follows standard calculus rules
- The computation already accounts for the specific structure where we differentiate with respect to $\vec{v}_c$