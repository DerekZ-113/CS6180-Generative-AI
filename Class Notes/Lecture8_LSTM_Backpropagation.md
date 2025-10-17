# CS 6180 Lecture 8: LSTM Backpropagation Through Time
**Date:** September 29, 2025  
**Topic:** Computing gradients in LSTMs - detailed mechanics and numerical example

**Administrative Notes:**
- HW2 promised for tomorrow (will be longer, 2 weeks)
- Google Cloud credits instructions shared on how to retrieve them

---

## 1. Computing Gradients in LSTMs

### Goal: Compute $\frac{\partial L}{\partial W_f}$

We'll work through a **2-timestep LSTM example** to understand the mechanics.

**LSTM parameters:**
- $W_f, U_f, \vec{b}_f$ (forget gate)
- $W_i, U_i, \vec{b}_i$ (input gate)
- $W_o, U_o, \vec{b}_o$ (output gate)
- $W_c, U_c, \vec{b}_c$ (candidate cell)

### Step 1: Apply Chain Rule

$$\frac{\partial L}{\partial W_f} = \frac{\partial L}{\partial \vec{f}^{(1)}} \cdot \frac{\partial \vec{f}^{(1)}}{\partial W_f} + \frac{\partial L}{\partial \vec{f}^{(2)}} \cdot \frac{\partial \vec{f}^{(2)}}{\partial W_f}$$

**Key observation:** We can compute $\frac{\partial \vec{f}^{(1)}}{\partial W_f}$ directly (similar to standard feedforward computation).

The challenge is computing $\frac{\partial L}{\partial \vec{f}^{(1)}}$ and $\frac{\partial L}{\partial \vec{f}^{(2)}}$.

### Step 2: Recursive Relationship Through Cell State

For $\frac{\partial L}{\partial \vec{f}^{(1)}}$:

$$\frac{\partial L}{\partial \vec{f}^{(1)}} = \frac{\partial L}{\partial \vec{c}^{(1)}} \cdot \frac{\partial \vec{c}^{(1)}}{\partial \vec{f}^{(1)}}$$

Expanding further:

$$\frac{\partial L}{\partial W_f} = \frac{\partial L}{\partial \vec{c}^{(1)}} \cdot \frac{\partial \vec{c}^{(1)}}{\partial \vec{f}^{(1)}} \cdot \frac{\partial \vec{f}^{(1)}}{\partial W_f} + \frac{\partial L}{\partial \vec{c}^{(2)}} \cdot \frac{\partial \vec{c}^{(2)}}{\partial \vec{f}^{(2)}} \cdot \frac{\partial \vec{f}^{(2)}}{\partial W_f}$$

**We're getting a recurrence relationship** linking the gradients through the cell states.

### Step 3: Compute Through Hidden States

For $\frac{\partial L}{\partial \vec{c}^{(1)}}$:

$$\frac{\partial L}{\partial \vec{c}^{(1)}} = \frac{\partial L}{\partial \vec{h}^{(1)}} \cdot \frac{\partial \vec{h}^{(1)}}{\partial \vec{c}^{(1)}} + \frac{\partial L}{\partial \vec{c}^{(2)}} \cdot \frac{\partial \vec{c}^{(2)}}{\partial \vec{c}^{(1)}}$$

Where:
$$\frac{\partial \vec{h}^{(1)}}{\partial \vec{c}^{(1)}} = \vec{o}^{(1)} \odot \left(1 - \tanh^2(\vec{c}^{(1)})\right)$$

### Step 4: Work from Boundary Conditions

**Start with boundary conditions:**

First compute $\frac{\partial L}{\partial \vec{h}^{(2)}}$ (boundary):

$$\rightarrow \frac{\partial L}{\partial \vec{h}^{(1)}}$$

Then use these to compute:

$$\rightarrow \frac{\partial L}{\partial \vec{c}^{(2)}}$$

$$\rightarrow \frac{\partial L}{\partial \vec{c}^{(1)}}$$

Finally compute $\frac{\partial L}{\partial W_f}$

**Bottom line:** We can finally compute $\frac{\partial L}{\partial W_f}$!

**Note:** PyTorch does all of this for you. We love PyTorch! 💙

---

## 2. Concrete Numerical Example: Long-Range Dependencies

### Setup: 10-Timestep Sequence

**Scenario:** Word 2 is relevant for predicting word 10.

With vanilla RNNs, we would have gotten vanishing gradients over 8 timesteps.

### LSTM with Specific Gate Values

**Initial condition:**
$$\vec{c}^{(0)} = \vec{0}$$

**Timestep 1:** Store information
$$\vec{c}^{(1)} = \vec{f}^{(1)} \odot \vec{c}^{(0)} + \vec{i}^{(1)} \odot \tilde{c}^{(1)}$$

With $\vec{i}^{(1)} = 0$, $\vec{f}^{(1)} = 1$:
$$\vec{c}^{(1)} = 1 \times 0 + 0 \times \tilde{c}^{(1)} = \vec{0}$$

**Timestep 2:** Store important information
$$\vec{c}^{(2)} = \vec{f}^{(2)} \odot \vec{c}^{(1)} + \vec{i}^{(2)} \odot \tilde{c}^{(2)}$$

With $\vec{i}^{(2)} = 1$, $\vec{f}^{(2)} = 1$:
$$\vec{c}^{(2)} = 1 \times 0 + 1 \times \tilde{c}^{(2)} = \tilde{c}^{(2)}$$

**Timesteps 3-10:** Preserve information

For $t = 3, 4, ..., 10$ with $\vec{i}^{(t)} = 0$, $\vec{f}^{(t)} = 1$:

$$\vec{c}^{(3)} = \vec{f}^{(3)} \odot \vec{c}^{(2)} + \vec{i}^{(3)} \odot \tilde{c}^{(3)} = \tilde{c}^{(2)}$$

The cell state **preserves** $\tilde{c}^{(2)}$ all the way to timestep 10!

### Gradient Flow Path

**Path:** $c^{(2)} \rightarrow c^{(1)} \rightarrow c^{(4)} \rightarrow ... \rightarrow c^{(10)} \rightarrow h^{(10)} \rightarrow L^{(10)}$

**Computing gradient at timestep 2:**

$$\frac{\partial L^{(10)}}{\partial \vec{c}^{(2)}} = \frac{\partial L^{(10)}}{\partial \vec{h}^{(10)}} \cdot \frac{\partial \vec{h}^{(10)}}{\partial \vec{c}^{(10)}} \cdot \frac{\partial \vec{c}^{(10)}}{\partial \vec{c}^{(9)}} \cdot ... \cdot \frac{\partial \vec{c}^{(3)}}{\partial \vec{c}^{(2)}}$$

Since $\frac{\partial \vec{c}^{(t)}}{\partial \vec{c}^{(t-1)}} = \vec{f}^{(t)}$:

$$\frac{\partial L^{(10)}}{\partial \vec{c}^{(2)}} = \frac{\partial L^{(10)}}{\partial \vec{h}^{(10)}} \cdot \vec{o}^{(10)} \odot \left(1-\tanh^2(\vec{c}^{(10)})\right) \cdot \prod_{i=3}^{10} \vec{f}^{(i)}$$

---

## 3. Numerical Result: Why LSTMs Work

### Forget Gates Near 1

If all forget gates $\vec{f}^{(i)} \approx 0.95$ for $i = 3, ..., 10$ (8 timesteps):

$$\prod_{i=3}^{10} \vec{f}^{(i)} \approx (0.95)^8 \approx 0.663$$

**Key insight:** The gradient remains **substantial** even after 8 timesteps!

### Comparison with Vanilla RNN

**Vanilla RNN gradient decay:**

With sigmoid derivatives $\sigma'(z) \leq 0.25$:

$$(0.25)^8 \approx 1.5 \times 10^{-5}$$

**The gradient has essentially vanished!**

### The Difference

| Architecture | Gradient after 8 steps | Can learn? |
|--------------|------------------------|------------|
| Vanilla RNN | $\approx 0.00002$ | ❌ No |
| LSTM (f≈0.95) | $\approx 0.663$ | ✅ Yes |

**LSTM advantage:** Forget gates can be close to 1, allowing gradients to flow largely unchanged through many timesteps.

---

## Key Takeaways

1. **Gradient computation in LSTMs:**
   - Use chain rule recursively through cell states
   - Work from boundary conditions backward
   - PyTorch handles this automatically

2. **Cell state gradient path:**
   - Direct path through element-wise multiplications
   - Each $\frac{\partial \vec{c}^{(t)}}{\partial \vec{c}^{(t-1)}} = \vec{f}^{(t)}$
   - No matrix multiplications to cause eigenvalue problems

3. **Numerical evidence:**
   - Forget gates ≈ 1 preserve gradients: $(0.95)^8 \approx 0.663$
   - Sigmoid derivatives cause vanishing: $(0.25)^8 \approx 0.00002$
   - This is why LSTMs can learn long-range dependencies

4. **Practical implication:**
   - Information stored at timestep 2 can influence predictions at timestep 10
   - Gradients flow back effectively for training
   - Vanilla RNNs would fail at this task

---

## Preview: Next Topic

**Transformers → Attention**

The famous paper: **"Attention is All You Need"** (by Google folks)

Transformers will address LSTM limitations and introduce parallel processing through attention mechanisms.