# CS 6180 Lecture 7: LSTM Architecture and Mechanics
**Date:** September 24, 2025  
**Topic:** Detailed LSTM mechanics, cell state updates, and gradient flow

**Note:** HW2 will be released tomorrow and will be longer (2 weeks instead of 1)

---

## 1. The Three LSTM Gates

### Gate 1: Forget Gate $\vec{f}^{(t)}$

**Purpose:** Controls what information in the cell state will be forgotten

$$\vec{f}^{(t)} = \sigma\left(W_f \vec{h}^{(t-1)} + U_f \vec{e}^{(t)} + \vec{b}_f\right)$$

**Output range:** $(0, 1)$ element-wise

### Gate 2: Input Gate $\vec{i}^{(t)}$

**Purpose:** Controls what new information will be stored in the cell

$$\vec{i}^{(t)} = \sigma\left(W_i \vec{h}^{(t-1)} + U_i \vec{e}^{(t)} + \vec{b}_i\right)$$

**Output range:** $(0, 1)$ element-wise

### Gate 3: Output Gate $\vec{o}^{(t)}$

**Purpose:** Controls what information will be exposed in the hidden state

$$\vec{o}^{(t)} = \sigma\left(W_o \vec{h}^{(t-1)} + U_o \vec{e}^{(t)} + \vec{b}_o\right)$$

**Output range:** $(0, 1)$ element-wise

---

## 2. Cell State Update Mechanism

### Step 1: Compute Candidate Cell State

$$\tilde{c}^{(t)} = \tanh\left(W_c \vec{h}^{(t-1)} + U_c \vec{e}^{(t)} + \vec{b}_c\right)$$

**Output range:** $(-1, 1)$

### Why tanh Instead of Sigmoid?

**Critical insight:** We need negative values to represent negation or reversal of information.

**Example:** Sentiment tracking through "The movie is fantastically bad"
- "fantastically" might produce positive sentiment
- "bad" needs to reverse/negate this
- Tanh's negative range allows this reversal

| Function | Range | Purpose |
|----------|-------|---------|
| Sigmoid | $(0, 1)$ | Gates (control signals) |
| Tanh | $(-1, 1)$ | Content (allows negation) |

### Step 2: Update Cell State

$$\vec{c}^{(t)} = \vec{f}^{(t)} \odot \vec{c}^{(t-1)} + \vec{i}^{(t)} \odot \tilde{c}^{(t)}$$

where $\odot$ is element-wise multiplication

**Interpretation:**
- Left term: Selectively forget old information
- Right term: Selectively add new information

### Concrete Numerical Example

**Given:**
- $c^{(t-1)} = 0.8$ (previous positive sentiment)
- $\tilde{c}^{(t)} = -0.6$ (new negative information)
- $f^{(t)} = 1$ (keep everything)
- $i^{(t)} = 1$ (accept all new info)

**Calculation:**

$$c^{(t)} = f^{(t)} \times c^{(t-1)} + i^{(t)} \times \tilde{c}^{(t)}$$

$$c^{(t)} = 1 \times 0.8 + 1 \times (-0.6) = 0.2$$

**Result:** Sentiment reduced from 0.8 to 0.2, showing the negation effect.

---

## 3. Hidden State Update

$$\vec{h}^{(t)} = \vec{o}^{(t)} \odot \tanh(\vec{c}^{(t)})$$

**Why tanh again?** Normalizes cell state values to $(-1, 1)$ before the output gate selects what to expose.

---

## 4. Complete LSTM Equations

**All gates:**
$$\vec{f}^{(t)} = \sigma(W_f \vec{h}^{(t-1)} + U_f \vec{e}^{(t)} + \vec{b}_f)$$
$$\vec{i}^{(t)} = \sigma(W_i \vec{h}^{(t-1)} + U_i \vec{e}^{(t)} + \vec{b}_i)$$
$$\vec{o}^{(t)} = \sigma(W_o \vec{h}^{(t-1)} + U_o \vec{e}^{(t)} + \vec{b}_o)$$

**Cell update:**
$$\tilde{c}^{(t)} = \tanh(W_c \vec{h}^{(t-1)} + U_c \vec{e}^{(t)} + \vec{b}_c)$$
$$\vec{c}^{(t)} = \vec{f}^{(t)} \odot \vec{c}^{(t-1)} + \vec{i}^{(t)} \odot \tilde{c}^{(t)}$$

**Hidden state:**
$$\vec{h}^{(t)} = \vec{o}^{(t)} \odot \tanh(\vec{c}^{(t)})$$

### Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\vec{f}^{(t)}$ | Forget gate |
| $\vec{i}^{(t)}$ | Input gate |
| $\vec{o}^{(t)}$ | Output gate |
| $\tilde{c}^{(t)}$ | Candidate cell state (new info) |
| $\vec{c}^{(t)}$ | Cell state |
| $\vec{h}^{(t)}$ | Hidden state |
| $\vec{e}^{(t)}$ | Input embedding |
| $\odot$ | Element-wise multiplication |

---

## 5. LSTM Computational Graph

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

**Key observation:** The cell state has a direct path from $c^{(t-1)}$ to $c^{(t)}$ through only element-wise operations.

---

## 6. Gradient Flow Through Cell State

### Exercise: Gradient with Respect to Previous Cell State

**Show:**

$$\frac{\partial \mathcal{L}}{\partial \vec{c}^{(t-1)}} = \frac{\partial \mathcal{L}}{\partial \vec{c}^{(t)}} \cdot \vec{f}^{(t)}$$

**Key insight:** This is element-wise multiplication (via the $\odot$ operation in the forward pass).

### Recursive Gradient Flow

**For $K$ steps backward:**

$$\frac{\partial \mathcal{L}}{\partial \vec{c}^{(t-K)}} = \frac{\partial \mathcal{L}}{\partial \vec{c}^{(t)}} \cdot \prod_{j=0}^{K-1} \vec{f}^{(t-j)}$$

**Critical observation:** This is a product of element-wise multiplications, NOT matrix multiplications.

**Why this matters:**
- Standard RNN: Matrix multiplication $W_h$ causes eigenvalue problems
- LSTM: Element-wise multiplication allows selective gradient preservation
- Each dimension of the cell state has independent gradient flow
- If $f^{(t-j)} \approx 1$, gradients flow unchanged through that timestep

---

## Key Takeaways

1. **Three gates serve different purposes:**
   - Forget: What to remove from memory
   - Input: What to add to memory
   - Output: What to expose from memory

2. **Tanh enables negation:**
   - Negative values allow reversing/negating stored information
   - Essential for tasks like sentiment analysis

3. **Cell state update is additive:**
   - Combines old information (gated by $\vec{f}$) and new information (gated by $\vec{i}$)
   - Provides gradient highway through element-wise operations

4. **Gradient flow is element-wise:**
   - Products of forget gates, not matrix multiplications
   - Avoids eigenvalue-related vanishing/exploding problems
   - Each cell dimension has independent gradient control

---

## Connection to Next Lecture

**Lecture 8:** Introduction to Attention mechanisms - building on the selective information flow concepts from LSTMs