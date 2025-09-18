# CS 6180 Lecture 5: Long Short-Term Memory (LSTMs)
**Date:** September 17, 2025  
**Topic:** Understanding LSTMs and how they solve RNN limitations

## 1. Recap: The Gradient Problem in RNNs

### Backpropagation Through Time (BPTT) Derivation

From previous lecture, we derived the gradient of loss with respect to hidden weights:

$\frac{\partial L}{\partial W_h} = \frac{\partial L^{(t)}}{\partial W_h} + \frac{\partial L^{(t)}}{\partial h^{(t-1)}} \cdot \frac{\partial h^{(t-1)}}{\partial W_h}$

Expanding this recursively:

$\frac{\partial L}{\partial W_h} = \frac{\partial L^{(t)}}{\partial W_h}\bigg|_t + \frac{\partial L^{(t)}}{\partial h^{(t-1)}} \cdot \left[\frac{\partial h^{(t-1)}}{\partial W_h}\bigg|_{t-1} \cdot (W_h \cdot h^{(t-2)}(W_h))\right]$

This can be rewritten as:

$\frac{\partial L^{(t)}}{\partial W_h} = \frac{\partial L^{(t)}}{\partial W_h}\bigg|_t + \frac{\partial L^{(t)}}{\partial h^{(t-1)}} \cdot \left[\frac{\partial h^{(t-1)}}{\partial W_h}\bigg|_{t-1} + \frac{\partial h^{(t-1)}}{\partial h^{(t-2)}} \cdot \frac{\partial h^{(t-2)}}{\partial W_h} \cdot (\ldots)\right]$

### The Core Problem

The gradient contains products of the form: $(W_h)^n$ as we backpropagate through $n$ time steps.

**Consequences:**
- If $||W_h|| < 1$: $(W_h)^n \rightarrow 0$ (vanishing gradients)
- If $||W_h|| > 1$: $(W_h)^n \rightarrow \infty$ (exploding gradients)
- Result: RNNs struggle to learn long-range dependencies (typically > 5-10 steps)

## 2. LSTM: The Solution

LSTMs (Hochreiter & Schmidhuber, 1997) introduce a **cell state** that allows information to flow unchanged through time, with **gates** controlling what information to keep, forget, or output.

### How LSTMs Fix the Gradient Problem
Instead of having gradients flow through repeated matrix multiplications $(W_h)^T$, LSTMs create an **additive path** through the cell state where gradients can flow with minimal transformation. The key is replacing multiplication with addition!

### Key Innovation: Two State Vectors
- **Hidden state ($h_t$)**: Short-term memory, used for output
- **Cell state ($C_t$)**: Long-term memory, protected from gradient issues

## 3. LSTM Architecture

### The Three Gates

#### 1. **Forget Gate** ($f_t$)
Decides what information to discard from previous cell state.

$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

- Output: $[0,1]$ for each dimension
- $0$ = completely forget, $1$ = completely remember

#### 2. **Input Gate** ($i_t$) and Candidate Values ($\tilde{C}_t$)
Decides what new information to store.

$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$

- $i_t$: what to update
- $\tilde{C}_t$: candidate values

#### 3. **Output Gate** ($o_t$)
Controls what parts of cell state to output.

$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

### Cell State Update Equation

$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$

This is the **key equation** - note the additive path! (where $\odot$ denotes element-wise multiplication)

### Hidden State Update

$h_t = o_t \odot \tanh(C_t)$

## 4. Why LSTMs Solve the Gradient Problem

### Gradient Flow Through Cell State
When we compute $\frac{\partial C_t}{\partial C_{t-1}}$:

$\frac{\partial C_t}{\partial C_{t-1}} = f_t$

Instead of repeated multiplication by $W_h$, we have:
- Multiplication by different forget gates at each step
- **Additive connections** that preserve gradients
- Gates can learn to be $\approx 1$ when needed to preserve gradients

### Comparison with RNN
| Aspect | RNN | LSTM |
|--------|-----|------|
| Gradient path | Multiplicative $(W_h)^t$ | Additive with gates |
| Long-term memory | Decays exponentially | Protected by cell state |
| Information flow | Fixed transformation | Learnable gates |
| Typical range | 5-10 steps | 100+ steps |

## 5. LSTM in Practice

### Implementation Tips
1. **Initialization**: 
   - Forget gate bias often initialized to 1 (remember by default)
   - Other biases typically initialized to 0

2. **Variants**:
   - **Peephole connections**: Gates can look at cell state
   - **GRU** (Gated Recurrent Unit): Simplified with 2 gates
   - **Coupled gates**: $i_t = 1 - f_t$

### Common Applications
- Language modeling
- Machine translation
- Speech recognition
- Time series prediction
- Music generation

## 6. Computational Considerations

### Parameter Count
For hidden size $h$ and input size $d$:
- RNN: $\mathcal{O}(h^2 + hd)$
- LSTM: $\mathcal{O}(4(h^2 + hd))$ - roughly 4x more parameters

### Training Time
- Slower than RNN per step due to more computations
- But often needs fewer epochs due to better gradient flow

## Key Takeaways
1. LSTMs solve vanishing/exploding gradient through **additive cell state path**
2. **Three gates** (forget, input, output) control information flow
3. Can maintain information over **100+ time steps**
4. Trade-off: More parameters and computation for better long-term memory

## Questions for Understanding
- Why is the additive path in $C_t$ crucial for gradient flow?
- How do the gates learn what to remember vs forget?
- When might you choose GRU over LSTM?

## 7. Model Evaluation: Perplexity

Now that we can train our LSTM model, let's evaluate it through testing.

### Perplexity
**Perplexity** is a standard evaluation metric for language models that measures how well a model predicts a sample.

$\text{Perplexity} = \sqrt[T]{\prod_{t=1}^{T} \frac{1}{P(\vec{x}^{(t+1)} | x^{(1)}, \ldots, x^{(t)})}}$

Or equivalently in log form:

$\text{Perplexity} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(x^{(t+1)} | x^{(1)}, \ldots, x^{(t)})\right)$

### What Perplexity Means
- **Lower is better** - indicates model is less "perplexed" by the test data
- Perplexity of 100 means the model is as confused as if it had to choose uniformly from 100 options at each step
- Directly related to cross-entropy loss: Perplexity = $e^{\text{cross-entropy}}$
- Good language models typically achieve perplexity scores between 20-100 on standard benchmarks

### Why Use Perplexity?
1. **Interpretable**: Represents the effective vocabulary size the model is choosing from
2. **Comparable**: Standard metric across different models and datasets
3. **Probabilistic**: Directly measures the model's confidence in its predictions