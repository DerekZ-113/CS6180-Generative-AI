# Long Short-Term Memory (LSTM) Networks: Comprehensive Guide
**CS 6180 - Generative AI**  
**Compiled from Lectures 5-8**

---

## Table of Contents
1. [The Problem: RNN Limitations](#1-the-problem-rnn-limitations)
2. [LSTM Architecture](#2-lstm-architecture)
3. [Why LSTMs Work: Mathematical Perspective](#3-why-lstms-work-mathematical-perspective)
4. [Backpropagation in LSTMs](#4-backpropagation-in-lstms)
5. [Numerical Evidence](#5-numerical-evidence)
6. [Partial Solutions vs Complete Solution](#6-partial-solutions-vs-complete-solution)
7. [Practical Considerations](#7-practical-considerations)
8. [Model Evaluation](#8-model-evaluation)

---

## 1. The Problem: RNN Limitations

### 1.1 RNN Architecture Recap

Standard RNN forward pass:
$$h^{(t)} = \sigma(W_h h^{(t-1)} + W_e \vec{e}^{(t)} + b)$$
$$y^{(t)} = \text{softmax}(\vec{P} \cdot \vec{h}^{(t)} + \vec{b}_2)$$

### 1.2 Backpropagation Through Time (BPTT)

The gradient with respect to hidden weights involves:

$$\frac{\partial L}{\partial W_h} = \sum_{t} \frac{\partial L^{(t)}}{\partial W_h}$$

When expanded recursively, this contains products of the form:

$$\frac{\partial h^{(t)}}{\partial h^{(1)}} \propto \prod_{i=2}^{t} \text{diag}(\sigma'(\tilde{a}^{(i)})) \cdot W_h$$

### 1.3 The Vanishing/Exploding Gradient Problem

**Two sources of instability:**

#### Source 1: Sigmoid Derivative Saturation

$$\sigma'(z) = \sigma(z) \cdot [1 - \sigma(z)]$$

Since $0 < \sigma(z) < 1$, we have $\sigma'(z) \leq 0.25$ for all $z$.

**Consequence:** Multiplying many sigmoid derivatives:
$$(0.25)^8 \approx 1.5 \times 10^{-5}$$

Gradients vanish exponentially with sequence length!

#### Source 2: Weight Matrix Eigenvalues

The repeated multiplication by $W_h$ means $(W_h)^{t-1}$ appears in gradients.

**Eigenvalue perspective:**

For matrix $A$ with eigenvalue $\lambda$ and eigenvector $\vec{x}$:
$$A\vec{x} = \lambda \vec{x}$$

| Eigenvalue Magnitude | Behavior | Effect on Gradients |
|---------------------|----------|---------------------|
| $\|\lambda\| > 1$ | Exponential growth | **Exploding gradients** |
| $\|\lambda\| = 1$ | Stable | **Stable gradients** ✓ |
| $\|\lambda\| < 1$ | Exponential decay | **Vanishing gradients** |

**The fundamental problem:** Random initialization typically produces eigenvalues far from 1, causing gradient instability over long sequences.

**Result:** RNNs struggle to learn dependencies beyond 5-10 time steps.

---

## 2. LSTM Architecture

### 2.1 Core Innovation: Dual State System

LSTMs maintain two states:
- **Cell state ($\vec{c}^{(t)}$)**: Long-term memory with protected gradient flow
- **Hidden state ($\vec{h}^{(t)}$)**: Short-term memory for output

### 2.2 The Three Gates

All gates use sigmoid activation, producing values in $(0,1)$:

#### Forget Gate: $\vec{f}^{(t)}$
**Purpose:** Controls what information to discard from cell state

$$\vec{f}^{(t)} = \sigma(W_f \vec{h}^{(t-1)} + U_f \vec{e}^{(t)} + \vec{b}_f)$$

- $\vec{f}^{(t)} \approx 0$: Forget this dimension
- $\vec{f}^{(t)} \approx 1$: Retain this dimension

#### Input Gate: $\vec{i}^{(t)}$
**Purpose:** Controls what new information to store

$$\vec{i}^{(t)} = \sigma(W_i \vec{h}^{(t-1)} + U_i \vec{e}^{(t)} + \vec{b}_i)$$

- $\vec{i}^{(t)} \approx 0$: Ignore new information
- $\vec{i}^{(t)} \approx 1$: Accept new information

#### Output Gate: $\vec{o}^{(t)}$
**Purpose:** Controls what information to expose to hidden state

$$\vec{o}^{(t)} = \sigma(W_o \vec{h}^{(t-1)} + U_o \vec{e}^{(t)} + \vec{b}_o)$$

- $\vec{o}^{(t)} \approx 0$: Hide cell content
- $\vec{o}^{(t)} \approx 1$: Expose cell content

### 2.3 Cell State Updates

#### Step 1: Compute Candidate Cell State

$$\tilde{c}^{(t)} = \tanh(W_c \vec{h}^{(t-1)} + U_c \vec{e}^{(t)} + \vec{b}_c)$$

**Why tanh and not sigmoid?**

| Function | Range | Purpose |
|----------|-------|---------|
| Sigmoid | $(0, 1)$ | Gates (control signals) |
| Tanh | $(-1, 1)$ | Content (allows negation) |

**Critical insight:** Negative values allow reversing/negating stored information.

**Example - Sentiment Analysis:**

Tracking sentiment through: "The movie is fantastically bad"
- After "fantastically": $c^{(t-1)} = 0.8$ (positive)
- Need to reverse when seeing "bad"
- Tanh allows: $\tilde{c}^{(t)} = -0.6$ (negation)
- Result: $c^{(t)} = 0.2$ (weakly positive)

#### Step 2: Update Cell State (THE KEY EQUATION)

$$\vec{c}^{(t)} = \vec{f}^{(t)} \odot \vec{c}^{(t-1)} + \vec{i}^{(t)} \odot \tilde{c}^{(t)}$$

where $\odot$ denotes element-wise multiplication (Hadamard product)

**Interpretation:**
- First term: Selectively **forget** old information
- Second term: Selectively **add** new information
- **Additive structure** is crucial for gradient flow!

### 2.4 Hidden State Update

$$\vec{h}^{(t)} = \vec{o}^{(t)} \odot \tanh(\vec{c}^{(t)})$$

The tanh normalizes cell state values to $(-1, 1)$ before the output gate selects what to expose.

### 2.5 Complete LSTM Equations Summary

**Forward pass computations:**

```
1. Forget gate:    f^(t) = σ(W_f h^(t-1) + U_f e^(t) + b_f)
2. Input gate:     i^(t) = σ(W_i h^(t-1) + U_i e^(t) + b_i)
3. Output gate:    o^(t) = σ(W_o h^(t-1) + U_o e^(t) + b_o)
4. Candidate:      c̃^(t) = tanh(W_c h^(t-1) + U_c e^(t) + b_c)
5. Cell update:    c^(t) = f^(t) ⊙ c^(t-1) + i^(t) ⊙ c̃^(t)
6. Hidden update:  h^(t) = o^(t) ⊙ tanh(c^(t))
```

### 2.6 Computational Graph

```
   h^(t-1) ────────┬──────────┬──────────┬──────────┐
                   │          │          │          │
   e^(t) ──────────┼──────────┼──────────┼──────────┤
                   ↓          ↓          ↓          ↓
                 f^(t)      i^(t)      o^(t)     c̃^(t)
                   │          │          │          │
   c^(t-1) ────→ [×]         │          │          │
                   │          ↓          │          │
                   └────→ [+ ←──────×]──┘          │
                          │                         │
                        c^(t) ────→ [tanh] ─────→ [×] ──→ h^(t)
                                                    ↑
                                                  o^(t)
```

**Key observation:** Cell state has a direct path through only element-wise operations.

---

## 3. Why LSTMs Work: Mathematical Perspective

### 3.1 Gradient Flow Through Cell State

#### The Critical Derivative

From the cell state update equation:
$$\vec{c}^{(t)} = \vec{f}^{(t)} \odot \vec{c}^{(t-1)} + \vec{i}^{(t)} \odot \tilde{c}^{(t)}$$

Taking the derivative:
$$\frac{\partial \vec{c}^{(t)}}{\partial \vec{c}^{(t-1)}} = \vec{f}^{(t)}$$

**This is element-wise multiplication, NOT matrix multiplication!**

#### Recursive Gradient Flow

For $K$ timesteps backward:

$$\frac{\partial \mathcal{L}}{\partial \vec{c}^{(t-K)}} = \frac{\partial \mathcal{L}}{\partial \vec{c}^{(t)}} \cdot \prod_{j=0}^{K-1} \vec{f}^{(t-j)}$$

**Key differences from RNN:**

| RNN | LSTM |
|-----|------|
| Matrix multiplication $(W_h)^K$ | Element-wise multiplication $\prod f^{(j)}$ |
| Eigenvalue problems | No eigenvalue issues |
| All dimensions coupled | Each dimension independent |
| Fixed transformation | Learnable gates |

### 3.2 Comparison: RNN vs LSTM Gradient Flow

**Standard RNN:**
$$\frac{\partial h^{(t)}}{\partial h^{(t-1)}} = \text{diag}(\sigma'(\tilde{a}^{(t)})) \cdot W_h$$

Problems:
- Sigmoid derivatives $\leq 0.25$
- Matrix multiplication with potentially problematic eigenvalues
- Product decays/explodes exponentially

**LSTM:**
$$\frac{\partial c^{(t)}}{\partial c^{(t-1)}} = \text{diag}(\vec{f}^{(t)})$$

Advantages:
- Forget gates can be $\approx 1$
- Element-wise operations only
- Each dimension has independent gradient control
- Gates are learned, not fixed

### 3.3 The Additive vs Multiplicative Path

**Why the additive structure matters:**

RNN: $h^{(t)} = \sigma(W_h h^{(t-1)} + ...)$
- Gradient must flow through $W_h$ at each step
- Repeated matrix multiplication causes eigenvalue compounding

LSTM: $c^{(t)} = f^{(t)} \odot c^{(t-1)} + i^{(t)} \odot \tilde{c}^{(t)}$
- Gradient has direct additive path through $f^{(t)}$
- Can flow largely unchanged when $f^{(t)} \approx 1$
- This is the "gradient highway" or "gradient superhighway"

---

## 4. Backpropagation in LSTMs

### 4.1 Computing Gradients: General Approach

To compute $\frac{\partial L}{\partial W_f}$ (or any LSTM parameter):

**Step 1: Apply chain rule across timesteps**

$$\frac{\partial L}{\partial W_f} = \sum_{t} \frac{\partial L}{\partial \vec{f}^{(t)}} \cdot \frac{\partial \vec{f}^{(t)}}{\partial W_f}$$

**Step 2: Compute gradients through cell states**

$$\frac{\partial L}{\partial \vec{f}^{(t)}} = \frac{\partial L}{\partial \vec{c}^{(t)}} \cdot \frac{\partial \vec{c}^{(t)}}{\partial \vec{f}^{(t)}}$$

From $\vec{c}^{(t)} = \vec{f}^{(t)} \odot \vec{c}^{(t-1)} + \vec{i}^{(t)} \odot \tilde{c}^{(t)}$:

$$\frac{\partial \vec{c}^{(t)}}{\partial \vec{f}^{(t)}} = \vec{c}^{(t-1)}$$

**Step 3: Recursively compute cell state gradients**

$$\frac{\partial L}{\partial \vec{c}^{(t)}} = \frac{\partial L}{\partial \vec{h}^{(t)}} \cdot \frac{\partial \vec{h}^{(t)}}{\partial \vec{c}^{(t)}} + \frac{\partial L}{\partial \vec{c}^{(t+1)}} \cdot \frac{\partial \vec{c}^{(t+1)}}{\partial \vec{c}^{(t)}}$$

Where:
$$\frac{\partial \vec{h}^{(t)}}{\partial \vec{c}^{(t)}} = \vec{o}^{(t)} \odot (1 - \tanh^2(\vec{c}^{(t)}))$$

$$\frac{\partial \vec{c}^{(t+1)}}{\partial \vec{c}^{(t)}} = \vec{f}^{(t+1)}$$

### 4.2 Working from Boundary Conditions

**Computational order (backward pass):**

1. Start at final timestep: Compute $\frac{\partial L}{\partial \vec{h}^{(T)}}$ from loss
2. Propagate to cell states: $\frac{\partial L}{\partial \vec{c}^{(T)}} \leftarrow \frac{\partial L}{\partial \vec{c}^{(T-1)}} \leftarrow ...$
3. Compute gate gradients: $\frac{\partial L}{\partial \vec{f}^{(t)}}, \frac{\partial L}{\partial \vec{i}^{(t)}}, \frac{\partial L}{\partial \vec{o}^{(t)}}$
4. Finally compute parameter gradients: $\frac{\partial L}{\partial W_f}, \frac{\partial L}{\partial W_i}, ...$

**Note:** PyTorch handles all of this automatically through autograd!

### 4.3 Key Gradient Path

**For long-range dependencies:**

$$\frac{\partial L^{(t)}}{\partial \vec{c}^{(t-k)}} = \frac{\partial L^{(t)}}{\partial \vec{h}^{(t)}} \cdot \frac{\partial \vec{h}^{(t)}}{\partial \vec{c}^{(t)}} \cdot \prod_{i=t-k+1}^{t} \vec{f}^{(i)}$$

This product of forget gates (not matrices!) allows gradients to flow effectively.

---

## 5. Numerical Evidence

### 5.1 Concrete 10-Timestep Example

**Scenario:** Word at position 2 is relevant for prediction at position 10.

**LSTM behavior with strategic gate values:**

**Timestep 1:** Initialize
- $c^{(0)} = 0$, $f^{(1)} = 1$, $i^{(1)} = 0$
- Result: $c^{(1)} = 0$

**Timestep 2:** Store important information
- $f^{(2)} = 1$, $i^{(2)} = 1$
- Result: $c^{(2)} = \tilde{c}^{(2)}$ (stores word 2 info)

**Timesteps 3-10:** Preserve information
- $f^{(t)} = 1$, $i^{(t)} = 0$ for all $t \in [3,10]$
- Result: $c^{(3)} = c^{(4)} = ... = c^{(10)} = \tilde{c}^{(2)}$

**The cell state perfectly preserves information from timestep 2 through timestep 10!**

### 5.2 Gradient Magnitude Comparison

**Computing gradient at timestep 2:**

$$\frac{\partial L^{(10)}}{\partial c^{(2)}} = \frac{\partial L^{(10)}}{\partial h^{(10)}} \cdot \frac{\partial h^{(10)}}{\partial c^{(10)}} \cdot \prod_{i=3}^{10} f^{(i)}$$

**Scenario: Forget gates near 1**

With $f^{(i)} \approx 0.95$ for $i = 3, ..., 10$ (8 timesteps):

$$\prod_{i=3}^{10} f^{(i)} \approx (0.95)^8 \approx 0.663$$

**The gradient is still ~66% of its original magnitude!**

**Compare with vanilla RNN:**

With sigmoid derivatives $\sigma'(z) \leq 0.25$:

$$(0.25)^8 \approx 1.5 \times 10^{-5} \approx 0.000015$$

**The gradient has vanished to nearly zero!**

### 5.3 Summary Table

| Architecture | Gradient Flow Factor | Gradient after 8 steps | Can Learn Long-Range? |
|--------------|---------------------|------------------------|----------------------|
| Vanilla RNN | $(0.25)^8$ | $\approx 0.00002$ | ❌ No |
| LSTM with $f \approx 0.95$ | $(0.95)^8$ | $\approx 0.663$ | ✅ Yes |
| LSTM with $f \approx 1.0$ | $(1.0)^8$ | $= 1.0$ | ✅ Yes (perfect) |

**Key insight:** LSTMs can learn to set forget gates near 1 for important information, maintaining gradient flow over long sequences.

---

## 6. Partial Solutions vs Complete Solution

### 6.1 Partial Solutions (For RNNs)

#### Solution 1: Gradient Clipping

**Implementation:**
```python
threshold = 10
if ||grad|| > threshold:
    grad = (grad / ||grad||) * threshold
```

**Effectiveness:**
- ✅ Prevents exploding gradients
- ❌ Doesn't address vanishing gradients
- ❌ Loses some learning information

#### Solution 2: Orthogonal Initialization

**Goal:** Initialize $W_h$ as orthogonal matrix with eigenvalues $|\lambda| = 1$

**Implementation:**
```python
# Use QR decomposition
A = np.random.randn(hidden_dim, hidden_dim)
Wh_orth, _ = np.linalg.qr(A)
```

**Effectiveness:**
- ✅ Helps with exploding gradients
- ⚠️ Partially helps with vanishing gradients
- ❌ Sigmoid derivatives still cause decay

#### Solution 3: Truncated BPTT

**Idea:** Only backpropagate through last $k$ timesteps

**Effectiveness:**
- ✅ Reduces computation
- ✅ Somewhat mitigates gradient issues
- ❌ Cannot learn dependencies beyond window
- ❌ Still has vanishing within window

### 6.2 Complete Solution: LSTM Architecture

**Why LSTMs are fundamentally different:**

| Aspect | RNN + Tricks | LSTM |
|--------|--------------|------|
| Approach | Training modifications | Architectural solution |
| Gradient path | Multiplicative through $W_h$ | Additive through cell state |
| Long-term memory | Decays over time | Protected by gates |
| Dependency range | ~20-50 steps | 100+ steps |
| Eigenvalue issues | Still present | Eliminated |

**The fundamental difference:** LSTMs solve the problem through architecture (additive path + learnable gates) rather than training tricks.

---

## 7. Practical Considerations

### 7.1 Parameter Count

For hidden size $h$ and input size $d$:

**RNN parameters:**
- $W_h$: $h \times h$
- $W_e$: $h \times d$
- $b$: $h$
- Total: $\mathcal{O}(h^2 + hd)$

**LSTM parameters:**
- 4 gates × [$W$: $h \times h$ + $U$: $h \times d$ + $\vec{b}$: $h$]
- Total: $4(h^2 + hd + h) = \mathcal{O}(4(h^2 + hd))$

**LSTM has ~4× more parameters than RNN**

### 7.2 Computational Cost

**Per timestep:**
- RNN: 1 matrix multiplication + 1 activation
- LSTM: 4 matrix multiplications + 6 element-wise operations + 4 activations

**Training time:**
- Slower per step than RNN
- But often needs fewer epochs due to better gradient flow

### 7.3 Initialization Best Practices

**Forget gate bias:** Often initialized to 1
- Encourages model to remember by default
- Prevents early training issues

**Other biases:** Typically initialized to 0

**Weight matrices:** Xavier/Glorot initialization or orthogonal initialization

### 7.4 Common Variants

**Peephole Connections:**
- Gates can observe cell state directly
- $f^{(t)} = \sigma(W_f h^{(t-1)} + U_f e^{(t)} + V_f c^{(t-1)} + b_f)$

**GRU (Gated Recurrent Unit):**
- Simplified LSTM with 2 gates instead of 3
- Combines cell and hidden state
- Fewer parameters, similar performance

**Coupled Gates:**
- Force $i^{(t)} = 1 - f^{(t)}$
- Reduces parameters, adds inductive bias

---

## 8. Model Evaluation

### 8.1 Perplexity

**Definition:** Standard metric for language models

$$\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(x^{(t+1)} | x^{(1)}, \ldots, x^{(t)})\right)$$

Equivalently:
$$\text{Perplexity} = \sqrt[T]{\prod_{t=1}^{T} \frac{1}{P(x^{(t+1)} | x^{(1)}, \ldots, x^{(t)})}}$$

**Interpretation:**
- Lower is better
- Perplexity of 100 means model is as confused as choosing uniformly from 100 options
- Directly related to cross-entropy: Perplexity = $e^{\text{cross-entropy}}$

**Typical values:**
- Good language models: 20-100 on standard benchmarks
- State-of-the-art models: <50

### 8.2 Why Perplexity?

1. **Interpretable:** Represents effective vocabulary size
2. **Comparable:** Standard across models and datasets
3. **Probabilistic:** Measures model confidence

---

## Key Takeaways: The Complete Picture

### The Problem
1. RNNs suffer from vanishing/exploding gradients due to:
   - Sigmoid derivative saturation ($\leq 0.25$)
   - Eigenvalue compounding in $(W_h)^t$

2. Partial solutions (clipping, orthogonal init, truncated BPTT) are insufficient

### The Solution
1. LSTMs introduce **cell state** with **additive gradient path**
2. Three **learnable gates** control information flow
3. Gradient flows through **element-wise operations**, not matrix multiplications

### Why It Works
1. $\frac{\partial c^{(t)}}{\partial c^{(t-1)}} = f^{(t)}$ (simple, no eigenvalues)
2. Gates can be $\approx 1$ to preserve gradients: $(0.95)^8 \approx 0.663$
3. Each dimension has independent gradient control

### The Trade-off
- 4× more parameters
- Slower computation per step
- But can learn 100+ step dependencies vs 5-10 for RNN

### Historical Context
- LSTMs (1997) remained state-of-the-art for sequence modeling until Transformers (2017)
- Still widely used for time series, speech, and applications requiring sequential processing
- Foundation for understanding modern architectures (GRUs, Transformers, attention)

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\vec{h}^{(t)}$ | Hidden state at time $t$ |
| $\vec{c}^{(t)}$ | Cell state at time $t$ |
| $\vec{e}^{(t)}$ | Input embedding at time $t$ |
| $\vec{f}^{(t)}$ | Forget gate at time $t$ |
| $\vec{i}^{(t)}$ | Input gate at time $t$ |
| $\vec{o}^{(t)}$ | Output gate at time $t$ |
| $\tilde{c}^{(t)}$ | Candidate cell state at time $t$ |
| $W_*$ | Weight matrix for hidden state |
| $U_*$ | Weight matrix for input embedding |
| $\vec{b}_*$ | Bias vector |
| $\odot$ | Element-wise (Hadamard) product |
| $\sigma$ | Sigmoid function |
| $\tanh$ | Hyperbolic tangent function |
| $\lambda$ | Eigenvalue |
| $\mathcal{L}$ | Loss function |