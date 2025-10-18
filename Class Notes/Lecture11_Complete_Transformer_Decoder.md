# CS 6180 Lecture 11: Complete Transformer Decoder Architecture
**Date:** October 15, 2025  
**Topic:** Layer Normalization, Residual Connections, Multi-Head Attention, and Autoregressive Models

**Administrative Note:** HW2 released - Yaaay!

---

## Overview

This lecture completes the Transformer decoder architecture by introducing three critical components: **layer normalization**, **residual connections**, and **multi-head attention**. We also formally define **autoregressive models** and show how Transformers, LSTMs, and RNNs all satisfy the autoregressive property.

---

## 1. Review: Building Blocks So Far

From previous lectures, we have:

```
Input x → Embedding E → e + p (position encoding)
                              ↓
                    [Masked Self-Attention]
                              ↓
                    [Feed-Forward Network]
                              ↓
                    [Linear Projection + Softmax]
```

**What's missing?** Three components needed for the full Transformer:
1. Layer Normalization (Add & Norm)
2. Residual Connections
3. Multi-Head Attention

---

## 2. Component 1: Layer Normalization

### Motivation: The Gradient Problem

Consider a simple neural network layer:
$$z = Wx + b$$

**Example 1:** $W = 1, b = 0, x \sim \mathcal{N}(0, 1)$
- Result: $z \sim \mathcal{N}(0, 1)$
- Gradients flow nicely ✓

**Example 2:** $W = 3, b = 2, x \sim \mathcal{N}(0, 1)$
- Result: $z \sim \mathcal{N}(2, 9)$
- After activation $h = \sigma(z)$:
  - Many values in saturation regions of sigmoid
  - Gradients close to zero
  - Learning will be slow ✗

**Visualization:**

```
     σ(z)
  1  ┌─────────
     │    ┌────── ← z ~ N(2,9): mostly in flat region
     │   ╱       gradients ≈ 0 → slow learning
  0.5├──╱
     │ ╱
     │╱ ← z ~ N(0,1): in steep region
  0  └───────── gradients close to 0 → good learning!
    -4  0  4  z
```

### The Solution: Layer Normalization

**Goal:** Standardize features to have mean 0 and variance 1.

**Formula:**
$$\hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

Where:
- $\mu = \mathbb{E}[z] = \frac{1}{d}\sum_{i=1}^{d} z_i$ (average over the features)
- $\sigma^2 = \text{Var}(z) = \frac{1}{d}\sum_{i=1}^{d} (z_i - \mu)^2$ (variance over the features)
- $\epsilon$ = small constant for numerical stability (e.g., $\epsilon \sim 10^{-5}$ to $10^{-6}$)

**Result:** $\mathbb{E}[\hat{z}] = 0$, $\text{Var}(\hat{z}) = 1$

### Why Add ε?

**Limitation of basic normalization:** If $\sigma^2$ is very small, dividing by it causes **exploding values**.

**Fix:** Add small constant $\epsilon$ to the denominator:
$$\hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

This ensures stability when variance is near zero.

### Adding Learnable Parameters

**Problem:** We might need to end up at regimes where gradients are close to 0, especially when we have almost converged.

**Solution:** Add learnable scale and shift parameters:

$$y = \gamma \hat{z} + \beta$$

Where:
- $\gamma$ (gamma) = learnable scale parameter
- $\beta$ (beta) = learnable shift parameter

**Complete Layer Normalization:**

$$y = \gamma \cdot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

### Training Dynamics with γ and β

**At the beginning of training:**
- Initialize: $\gamma = 1, \beta = 0$
- Model tries to keep features standardized: $\mathbb{E}[y] = 0, \text{Var}(y) = 1$

**As model learns and approaches convergence:**
- Model has **flexibility through $\gamma$ and $\beta$** to go back to the unstandardized features if needed
- Can escape standardization when beneficial for the task

**Key insight:** Layer normalization mainly helps with training by ensuring flowing gradients, but model can also converge to regimes with gradients near zero when appropriate.

---

## 3. Component 2: Residual Connections (Add & Norm)

### The Residual Addition Pattern

Instead of just applying transformations, **add the input back** to the output:

$$\text{output} = \text{Function}(\text{input}) + \text{input}$$

### Two Architectural Approaches

#### Approach 1: Post-Layer Normalization
**Used in original "Attention is All You Need" Transformer paper**

$$y_1 = \text{LN}(x + \text{AH}(x))$$
$$y_2 = \text{LN}(y_1 + \text{FFN}(y_1))$$

Where:
- AH = Attention Head
- FFN = Feed-Forward Network
- LN = Layer Normalization

**Pattern:**
```
x → [Attention] → (+) → [LayerNorm] → y₁
    ↑_______________|

y₁ → [FFN] → (+) → [LayerNorm] → y₂
     ↑__________|
```

#### Approach 2: Pre-Layer Normalization
**More commonly used in modern implementations (e.g., GPT models)**

$$y_1 = x + \text{AH}(\text{LN}(x))$$
$$y_2 = y_1 + \text{FFN}(\text{LN}(y_1))$$

**Pattern:**
```
x → [LayerNorm] → [Attention] → (+) → y₁
    ↑___________________________|

y₁ → [LayerNorm] → [FFN] → (+) → y₂
     ↑____________________|
```

### Why Approach 2 is Better

**Gradient flow advantage:**
- Gradients flow **more directly** through residual connections
- More stable for deep networks
- Currently the preferred approach in practice

### Purpose of Residual Connections

Residual connections ensure:
- We have **not lost any previous information**
- We **keep adding new information** at each layer
- Gradients can flow directly back through the network
- Enables training of very deep networks (100+ layers)

---

## 4. Component 3: Multi-Head Attention

### Motivation

**Question:** Instead of only one attention head, why not use many?

**Answer:** Each attention head can attend to **different features** of the sentence:
- **Head 1:** Meaning/semantics
- **Head 2:** Grammar
- **Head 3:** Syntax
- **Head 4:** Pronoun relationships
- And so on...

*Note:* This is the **intuition/hope**, but what attention heads actually learn is an **open research question** in **mechanistic interpretability**.

### Mathematical Formulation

For each attention head $h \in \{1, 2, \ldots, H\}$, we have separate transformation matrices:

$$\vec{K}_i^{(h)} = K^{(h)} \vec{e}_i$$
$$\vec{q}_i^{(h)} = Q^{(h)} \vec{e}_i$$
$$\vec{V}_i^{(h)} = V^{(h)} \vec{e}_i$$

**Attention computation for head $h$:**

1. **Compute scores:**
   $$s_{ij}^{(h)} = \vec{q}_i^{(h)^T} \vec{K}_j^{(h)} \quad \text{for } j = 1, 2, \ldots, m$$

2. **Apply softmax:**
   $$\alpha_{ij}^{(h)} = \frac{\exp(s_{ij}^{(h)})}{\sum_{k=1}^{m} \exp(s_{ik}^{(h)})}$$

3. **Compute output:**
   $$\vec{o}_i^{(h)} = \sum_{j=1}^{m} \alpha_{ij}^{(h)} \vec{V}_j^{(h)}$$

### Combining Multiple Heads

**Concatenate outputs from all heads:**

$$\vec{o}_i = [\vec{o}_i^{(1)}, \vec{o}_i^{(2)}, \vec{o}_i^{(3)}] \in \mathbb{R}^{d \times 3}$$

For $H$ heads with dimension $d$ each:
$$\vec{o}_i = [\vec{o}_i^{(1)}, \vec{o}_i^{(2)}, \ldots, \vec{o}_i^{(H)}] \in \mathbb{R}^{d \times H}$$

### Parameter Count

Each attention head has its own parameters:
- $Q^{(h)}, K^{(h)}, V^{(h)}$ for head $h$

**Total parameters for multi-head attention:**
- $H$ heads × 3 matrices × $d \times d$ parameters
- Total: $3Hd^2$ parameters

---

## 5. Scaled Dot-Product Attention

### The Scaling Issue

**Problem:** If dot products $\vec{q}_i^T \vec{K}_j$ become very large, softmax saturates.

**Example:**
```
Scores:  [10, 2, 3, 1]
Softmax: [0.9997, 0.0001, 0.0001, 0.0000]
```

Almost all attention goes to one position - not enough distribution across relevant words!

### Solution: Scale by Dimension

**Scaled attention formula:**

$$\vec{o}^{(h)} = \text{softmax}\left(\frac{(XQ^{(h)})(XK^{(h)})^T}{\sqrt{d}}\right) XV^{(h)}$$

**In matrix form for all positions at once:**

$$O^{(h)} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

Where:
- $Q = XQ^{(h)} \in \mathbb{R}^{m \times d}$ (queries for all positions)
- $K = XK^{(h)} \in \mathbb{R}^{m \times d}$ (keys for all positions)
- $V = XV^{(h)} \in \mathbb{R}^{m \times d}$ (values for all positions)
- $m$ = sequence length
- $d$ = dimension

**Why divide by $\sqrt{d}$?**

If dot product components are independent with mean 0 and variance 1:
- Dot product $\vec{q}^T\vec{k}$ has variance $d$
- Dividing by $\sqrt{d}$ gives variance 1
- Prevents extremely large values going into softmax
- Maintains stable attention distributions

---

## 6. Complete Transformer Decoder Architecture

### Full Architecture Diagram

```
Input: x
   ↓
[Embedding E] → e
   ↓
[Add Position] → e + p
   ↓
┌────────────────────────────┐
│  Transformer Unit          │
│  ┌──────────────────────┐  │
│  │ Masked Self-Attention│  │  ← Multi-head
│  └──────────────────────┘  │
│           ↓                │
│      [Add & Norm]          │
│           ↓                │
│  ┌──────────────────────┐  │
│  │  Feed-Forward (FFN)  │  │
│  └──────────────────────┘  │
│           ↓                │
│      [Add & Norm]          │
└────────────────────────────┘
   ↓
[Repeat unit several times]
   ↓
[Linear Projection]
   ↓
[Softmax]
   ↓
Output
```

### Mathematical Flow Through One Transformer Unit

**Using Approach 2 (Pre-LayerNorm, modern style):**

$$y_1 = x + \text{MultiHeadAttention}(\text{LN}(x))$$
$$y_2 = y_1 + \text{FFN}(\text{LN}(y_1))$$

Where:
- **MultiHeadAttention** includes masking and scaling
- **FFN** is a two-layer MLP with non-linear activation

### Complete Forward Pass

**Step 1: Input Processing**
$$x_i \rightarrow e_i + p_i \quad \text{for each position } i$$

**Step 2: Stack of Transformer Units** (typically 6-12 layers)

For layer $\ell$:
$$y_1^{(\ell)} = x^{(\ell)} + \text{MultiHeadAttention}(\text{LN}(x^{(\ell)}))$$
$$y_2^{(\ell)} = y_1^{(\ell)} + \text{FFN}(\text{LN}(y_1^{(\ell)}))$$

Set $x^{(\ell+1)} = y_2^{(\ell)}$ for next layer

**Step 3: Output Projection**
$$\text{logits} = W_{\text{out}} \cdot y_2^{(L)} + b_{\text{out}}$$
$$P(\text{next word}) = \text{softmax}(\text{logits})$$

---

## 7. Autoregressive Models

### Definition

**Autoregressive property:** A model where each output depends **only on the past** (previously observed values).

### Mathematical Formulation

The joint probability of a sequence can be factored using the chain rule:

$$P(x_1, x_2, x_3, \ldots, x_m) = P(x_1) \cdot P(x_2 | x_1) \cdot P(x_3 | x_1, x_2) \cdots P(x_m | x_1, x_2, \ldots, x_{m-1})$$

**Autoregressive property:**
$$x_m \text{ depends on the past words that occurred}$$

### Origins: Time Series

The term comes from **time series literature** - using observations from previous times to predict output at the next time step.

### Neural Networks as Autoregressive Models

We can use a **neural network to model** the conditional probability function:

$$P(x_i | x_1, \ldots, x_{i-1}) = f_\theta(x_1, \ldots, x_{i-1})$$

Where $\theta$ represents the neural network parameters.

---

## 8. Examples: Which Models Are Autoregressive?

### Example 1: Standard RNN

```
Input:    x₁    x₂    x₃    x₄
          ↓     ↓     ↓     ↓
Hidden:  h₁ → h₂ → h₃ → h₄
          ↓     ↓     ↓     ↓
Output:  y₁    y₂    y₃    y₄
```

**Analysis:**
- $y_1 = f(h_1) = f(x_1)$ ✓
- $y_2 = f(h_2) = f(h_1, x_2) = f(x_1, x_2)$ ✓
- $y_3 = f(h_3) = f(h_2, x_3) = f(x_1, x_2, x_3)$ ✓
- $y_4 = f(h_4) = f(h_3, x_4) = f(x_1, x_2, x_3, x_4)$ ✓

**Autoregressive property is satisfied** ✓

### Example 2: Convolutional Neural Network (CNN)

```
Input:    x₁    x₂    x₃    x₄
          ↓ ╲   ↓ ╲   ↓ ╲   ↓
Hidden:  h₁ → h₂ → h₃ → h₄
          ↓     ↓     ↓     ↓
Output:  y₁    y₂    y₃    y₄
```

**Analysis:**
- $h_4$ depends on $x_2, x_3, x_4$ (due to cross-connections shown in yellow)
- $y_4$ depends on future input $x_4$

**Not autoregressive** ✗

**Removing specific edges makes it autoregressive:**

Removing the three forward edges (yellow crosses in notes):
- Edge from $x_2$ to $h_4$
- Edge from $x_3$ to $h_4$  
- Edge from $x_4$ to $h_4$

**Priority order for valid edges:** Each hidden state can only depend on current and past inputs.

Valid dependencies:
- $1 \rightarrow 1$ ✓
- $1 \rightarrow 2$ ✓
- $2 \rightarrow 2$ ✓
- $2 \rightarrow 3$ ✓
- $3 \rightarrow 3$ ✓
- $3 \rightarrow 4$ ✓

This creates **6 total valid edges** with proper autoregressive structure (priority: $2^1 = 2$ edges for 1, $2^2 = 4$ more for 2-3, etc.)

### Example 3: Transformer with Masking

```
x₁, x₂, x₃, x₄
↓   ↓   ↓   ↓
[Self-Attention with Masking]
```

**Analysis:**
- Position 1: Can only attend to $x_1$
- Position 2: Can only attend to $x_1, x_2$
- Position 3: Can only attend to $x_1, x_2, x_3$
- Position 4: Can only attend to $x_1, x_2, x_3, x_4$

**Autoregressive property is satisfied** ✓ (due to masking)

---

## Key Takeaways

### 1. Layer Normalization
- Standardizes features to prevent gradient saturation
- Includes $\epsilon$ for numerical stability
- Learnable $\gamma$ and $\beta$ provide flexibility
- Mainly helps during training but can adapt at convergence

### 2. Residual Connections
- Add input to output: $y = F(x) + x$
- Two approaches: post-norm (original) and pre-norm (modern)
- Pre-norm has better gradient flow and stability
- Essential for training deep networks

### 3. Multi-Head Attention
- Multiple parallel attention mechanisms
- Each head learns different patterns (hopefully)
- Outputs are concatenated
- What heads actually learn is an open research question

### 4. Scaled Dot-Product
- Divide attention scores by $\sqrt{d}$
- Prevents softmax saturation
- Maintains stable attention distributions

### 5. Autoregressive Models
- Each position depends only on past positions
- RNNs, LSTMs, and masked Transformers are all autoregressive
- Essential property for language modeling
- Comes from time series literature

### 6. Complete Decoder Architecture
We now have all components of the Transformer decoder:
- Position encoding
- Masked self-attention  
- Multi-head attention
- Layer normalization
- Residual connections
- Feed-forward networks

**This is the architecture used in GPT and other decoder-only language models!**

---

## Mathematical Notation Legend

### Layer Normalization
- $z$ = pre-normalization values
- $\hat{z}$ = normalized values (mean 0, var 1)
- $\mu$ = mean across features
- $\sigma^2$ = variance across features
- $\epsilon$ = stability constant ($10^{-5}$ to $10^{-6}$)
- $\gamma$ = learnable scale parameter
- $\beta$ = learnable shift parameter
- $y$ = final layer norm output

### Multi-Head Attention
- $H$ = number of attention heads
- $h \in \{1, 2, \ldots, H\}$ = head index
- $Q^{(h)}, K^{(h)}, V^{(h)}$ = transformation matrices for head $h$
- $\vec{o}_i^{(h)}$ = output from head $h$ at position $i$
- $[\cdot, \cdot, \ldots]$ = concatenation operator

### Scaled Attention
- $\sqrt{d}$ = scaling factor (square root of dimension)
- $QK^T$ = attention score matrix
- $\frac{QK^T}{\sqrt{d}}$ = scaled attention scores

### Architecture Components
- LN = Layer Normalization
- AH = Attention Head / Multi-Head Attention
- FFN = Feed-Forward Network (MLP)
- $y_1, y_2$ = intermediate outputs after each sub-layer

### Autoregressive Notation
- $x_1, x_2, \ldots, x_m$ = sequence of inputs/words
- $P(x_i | x_1, \ldots, x_{i-1})$ = conditional probability
- $\theta$ = model parameters