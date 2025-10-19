# CS 6180 Lecture 6: RNN Gradient Problems - Advanced Perspectives
**Date:** September 24, 2025  
**Topic:** Eigenvalue analysis and partial solutions to gradient instability

**Prerequisites:** Lecture 5 (gradient problems, LSTM architecture)

---

## 1. The Eigenvalue Perspective on Gradient Flow

Building on Lecture 5's discussion of gradient problems in RNNs, we now examine WHY the weight matrix $W_h$ causes instability through the lens of eigenvalues.

### What Are Eigenvalues?

For a square matrix $A$ and vector $\vec{x}$, if:

$$A\vec{x} = \lambda \vec{x} \quad \text{where } \vec{x} \neq \vec{0}$$

Then:
- $\lambda$ is an **eigenvalue** of $A$
- $\vec{x}$ is an **eigenvector** of $A$

**Geometric Interpretation:**
- $\lambda$ controls **magnitude** (scaling factor)
- $\vec{x}$ controls **direction**
- The matrix $A$ stretches/shrinks the eigenvector by factor $\lambda$ without changing its direction

### Finding Eigenvalues

To find eigenvalues, solve the characteristic equation:

$$\det(A - \lambda I) = 0$$

**Example:**

$$A = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

$$\det\begin{pmatrix} 1-\lambda & 1 \\ 1 & 1-\lambda \end{pmatrix} = (1-\lambda)^2 - 1 = 0$$

$$\lambda^2 - 2\lambda = 0 \Rightarrow \lambda = 0 \text{ or } \lambda = 2$$

### Eigenvalue Magnitude and Gradient Behavior

| Eigenvalue Magnitude | Behavior Over Time | Effect on Gradients |
|---------------------|-------------------|---------------------|
| $\|\lambda\| > 1$ | Exponential growth | **Exploding gradients** |
| $\|\lambda\| = 1$ | Stable (no growth/decay) | **Stable gradients** ✓ |
| $\|\lambda\| < 1$ | Exponential decay | **Vanishing gradients** |
| $\lambda < -1$ | Oscillating + growing | **Exploding with oscillation** |

### Connection to RNN Gradient Flow

Recall from Lecture 5 that gradient flow involves:

$$\frac{\partial h^{(t)}}{\partial h^{(1)}} \propto \prod_{i=2}^{t} \text{diag}(\sigma'(\tilde{a}^{(i)})) \cdot W_h$$

The repeated multiplication by $W_h$ means:
- Powers of $W_h$ appear: $(W_h)^{t-1}$
- If eigenvalues $\|\lambda\| > 1$: $(W_h)^{t-1}$ explodes
- If eigenvalues $\|\lambda\| < 1$: $(W_h)^{t-1}$ vanishes

**The Root Cause:** Random initialization of $W_h \sim \mathcal{N}(0, \sigma^2)$ can produce eigenvalues far from 1, causing gradient instability.

The derivative also relies on $W_h$, which can cause changes in learning through eigenvalue effects.

---

## 2. Orthogonal Matrices and Initialization

### Goal: Eigenvalues Near 1

**Ideal:** Initialize $W_h$ such that all eigenvalues have magnitude $\approx 1$, preventing both exploding and vanishing gradients.

**Solution:** Use an **orthogonal matrix**.

### Properties of Orthogonal Matrices

A matrix $Q$ is orthogonal if its columns are **orthonormal**:
- Orthogonal to each other (dot product = 0)
- Unit length (norm = 1)

**Key Properties:**
1. $Q^T Q = I$ (transpose equals inverse)
2. $\det(Q) = \pm 1$ (always invertible)
3. **All eigenvalues have magnitude exactly 1**: $|\lambda| = 1$

### Example: 2D Rotation Matrix

Rotation matrices are orthogonal:

$$R = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}$$

For $Q = \begin{pmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{pmatrix}$:

Finding eigenvalues:

$$Q - \lambda I = \begin{pmatrix} \frac{1}{\sqrt{2}} - \lambda & -\frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} - \lambda \end{pmatrix}$$

$$\det(Q - \lambda I) = \left(\frac{1}{\sqrt{2}} - \lambda\right)\left(-\frac{1}{\sqrt{2}} - \lambda\right) - \frac{1}{2} = 0$$

$$-\frac{1}{2} - \frac{\lambda}{\sqrt{2}} + \frac{\lambda}{\sqrt{2}} + \lambda^2 - \frac{1}{2} = 0$$

$$\lambda^2 = 1 \Rightarrow \lambda = \pm 1$$

Both eigenvalues have magnitude 1! ✓

### Why This Helps

For orthogonal $Q$:

$$(Q\vec{x}) = \lambda\vec{x}$$
$$(Q - \lambda I)\vec{x} = \vec{0}$$

For non-trivial solutions: $\det(Q - \lambda I) = 0$

This constraint ensures $|\lambda| = 1$, providing stable gradient flow.

### Orthogonal Initialization in Practice

**Implementation using QR Decomposition (Gram-Schmidt Process):**

```python
import numpy as np

# Standard random initialization (BAD - random eigenvalues)
# Wh = np.random.randn(hidden_dim, hidden_dim)

# Orthogonal initialization (GOOD - eigenvalues near 1)
A = np.random.randn(hidden_dim, hidden_dim)
Wh_orth, _ = np.linalg.qr(A)  # Q from QR decomposition
```

The QR decomposition (Gram-Schmidt process) converts any matrix into an orthogonal matrix.

**Result:** Helps prevent both exploding and vanishing gradients, though doesn't fully eliminate vanishing gradients when combined with sigmoid derivatives.

---

## 3. Gradient Clipping

### The Exploding Gradient Problem

Even with careful initialization, gradients can still explode during training due to:
- Unfortunate data sequences
- Learning dynamics pushing $W_h$ eigenvalues $> 1$

### Solution: Gradient Clipping

**Algorithm:**

```python
threshold = 10  # typical value

if ||grad|| > threshold:
    grad = (grad / ||grad||) * threshold
```

**How it works:**
1. Compute gradient norm: $\|\nabla_W \mathcal{L}\| = \sqrt{\sum_i \sum_j (\frac{\partial \mathcal{L}}{\partial W_{ij}})^2}$
2. If norm exceeds threshold, rescale gradient to have norm = threshold
3. Direction is preserved, magnitude is capped

**Example:**

$$\text{grad} = \begin{pmatrix} 8 \\ 6 \end{pmatrix}, \quad \|\text{grad}\| = 10$$

If threshold = 10, gradient is unchanged.

$$\text{grad} = \begin{pmatrix} 80 \\ 60 \end{pmatrix}, \quad \|\text{grad}\| = 100$$

If threshold = 10:

$$\text{grad}_{\text{clipped}} = \frac{\begin{pmatrix} 80 \\ 60 \end{pmatrix}}{100} \times 10 = \begin{pmatrix} 8 \\ 6 \end{pmatrix}$$

### Pros and Cons

**✓ Pros:**
- Prevents training instability from exploding gradients
- Simple to implement
- Computationally cheap

**✗ Cons:**
- Model loses some learning information (gradients artificially capped)
- Doesn't address vanishing gradients
- Requires manual threshold tuning

---

## 4. Truncated Backpropagation Through Time (TBPTT)

### The Long-Range Dependency Problem

Consider:

> "The boy who is in France, a country ... [many words] ... croissant ... [many words] ... is a student."

**Problem:** Later words in a sentence are more affected by vanishing gradients because we include more derivatives of sigmoid for previous words.

With a small window of previous words, the model may not properly learn the relationship between "student" and "boy."

### Truncated BPTT Solution

**Idea:** Limit the number of time steps used for backpropagation.

Instead of backpropagating through all $t$ time steps:

$$\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{j=1}^{t} \frac{\partial \mathcal{L}^{(t)}}{\partial h^{(j)}} \cdot \frac{\partial h^{(j)}}{\partial W_h}$$

Only backpropagate through last $k$ steps:

$$\frac{\partial \mathcal{L}}{\partial W_h} \approx \sum_{j=\max(1, t-k)}^{t} \frac{\partial \mathcal{L}^{(t)}}{\partial h^{(j)}} \cdot \frac{\partial h^{(j)}}{\partial W_h}$$

### Typical Window Sizes

| Window Size | Assessment | Trade-offs |
|-------------|-----------|------------|
| 20-50 words | Good ✓ | Manageable computation, decent context |
| 100 words | Meh | More computation, vanishing gradient risk returns |
| Full sequence | Bad | Computationally expensive, severe vanishing gradients |

### Pros and Cons

**✓ Pros:**
- Reduces computational cost significantly
- Makes training feasible for very long sequences
- Somewhat mitigates gradient issues

**✗ Cons:**
- Still susceptible to vanishing gradients within the window
- Cannot learn dependencies beyond the truncation window
- Model may miss important long-range relationships
- Limits the theoretical capacity to model long sequences

---

## 5. Why These Are Only Partial Solutions

### Summary of Limitations

| Solution | Helps With | Limitations |
|----------|-----------|-------------|
| **Gradient Clipping** | Exploding gradients | Doesn't address vanishing; loses learning information |
| **Orthogonal Init** | Both | Sigmoid derivatives still cause vanishing over time |
| **Truncated BPTT** | Computation, some gradient issues | Cannot learn beyond window; still has vanishing within window |

### The Fundamental Problem Remains

Even with all three techniques:
- Vanishing gradients persist for long sequences
- The additive path for gradient flow doesn't exist
- Information still "leaks out" through repeated sigmoid applications

**The complete solution:** LSTMs (covered in Lecture 5) fundamentally solve this through:
- Cell state providing an **additive path** for gradients
- Gates controlling information flow
- $\frac{\partial C_t}{\partial C_{t-1}} = f_t$ (simple, not chained multiplication)

---

## Key Takeaways

1. **Eigenvalue perspective explains WHY** gradient problems occur:
   - $|\lambda| > 1$ → exploding
   - $|\lambda| < 1$ → vanishing
   - $|\lambda| = 1$ → stable ✓

2. **Three partial solutions exist:**
   - Gradient clipping (exploding only)
   - Orthogonal initialization (helps both)
   - Truncated BPTT (computational + partial gradient help)

3. **Orthogonal matrices** have eigenvalues with magnitude 1, providing stable initialization

4. **These solutions are insufficient** for truly long-range dependencies

5. **LSTMs remain the fundamental solution** through architectural changes, not just training tricks

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\lambda$ | Eigenvalue (scalar) |
| $\vec{x}$ | Eigenvector |
| $Q$ | Orthogonal matrix |
| $\det(A)$ | Determinant of matrix $A$ |
| $\|\cdot\|$ | Magnitude/norm |
| $I$ | Identity matrix |