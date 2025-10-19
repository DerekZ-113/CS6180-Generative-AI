# CS 6180 - Lecture 2: Representation of Words
## Date: 09/08

## Neural Network Review

### Loss Function
For binary classification where $y \in \{0, 1\}$:

$$\ell(y, \hat{y}) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$$

**Case Analysis:**
- If $y = 0$: 
  - $\text{loss} = -(1-0)\log(1-\hat{y}) = -\log(1-\hat{y})$
  - When $\hat{y} \approx 1$ → error is huge
  - When $\hat{y} \approx 0$ → error $\approx 0$

### Neural Network Architecture

```
Input Layer    Hidden Layer    Output Layer
    x₁ ━━━━━┓
    x₂ ━━━━━╋━━━ [neurons] ━━━━ σ(·) → ŷ
    ...     ┃
    x₁₀ ━━━━┛
```

**Forward Pass:**

$$\vec{z} = W^{(1)}\vec{x} + \vec{b}^{(1)}$$
- Dimensions: $[10 \times 1] = [10 \times d] \cdot [d \times 1] + [10 \times 1]$

$$\vec{a} = g(\vec{z})$$

$$\hat{y} = \sigma(W^{(2)}\vec{a} + b^{(2)})$$
- Dimensions: $[1 \times 1] = [1 \times 10] \cdot [10 \times 1] + [1 \times 1]$

**Parameters of the model:**
- $W^{(1)} \in \mathbb{R}^{10 \times d}$ - first layer weights
- $\vec{b}^{(1)} \in \mathbb{R}^{10 \times 1}$ - first layer bias
- $W^{(2)} \in \mathbb{R}^{1 \times 10}$ - second layer weights
- $b^{(2)} \in \mathbb{R}^{1 \times 1}$ - second layer bias

**Objective:** Find the best values of these parameters that minimize the loss function.

## Gradient Descent

**Update Rule:**
$$X_{k+1} = X_k - \alpha \nabla f(X_k)$$

Where $\alpha$ is the learning rate and we need to compute derivatives of the loss function with respect to the parameters.

### Partial Derivatives

**Chain Rule for $W^{(2)}$:**
$$\frac{\partial \ell}{\partial W^{(2)}} = \frac{\partial \ell}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W^{(2)}}$$

**Key Derivatives:**

1. Loss derivative:
   $\frac{\partial \ell}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$

2. Sigmoid function and its derivative:
   $\sigma(z) = \frac{1}{1 + e^{-z}}$
   $\sigma'(z) = \sigma(z)[1 - \sigma(z)]$

3. Weight gradient:
   $\frac{\partial \hat{y}}{\partial W^{(2)}} = \sigma(z)[1-\sigma(z)] \cdot \vec{a}^T$
   
   where $z = W^{(2)}\vec{a} + b^{(2)}$

**For bias term:**
$$\frac{\partial \hat{y}}{\partial b^{(2)}} = \sigma(z) \cdot [1-\sigma(z)]$$

*(Problem in HW to review these computations further)*

## Representing Words with Vectors

### One-hot Encoding

**Simple idea:** Represent each word as a vector with all zeros except for a single 1.

**Example:**
```
aardvark = [1, 0, 0, ..., 0]ᵀ
coffee   = [0, 1, 0, ..., 0]ᵀ
zebra    = [0, 0, 0, ..., 1]ᵀ
```

**Problem:** 
- Can be very large (size of vocabulary)
- No notion of similarity between different words

### Word2Vec Approach

**Key Insight:** Words in the context of a word are important.

> "A word's meaning is given by the words that frequently appear close by"
> 
> "You shall know a word by the company it keeps" - J.R. Firth, 1957

**Example word vectors (dense representations):**
```
are = [0.286, 0.792, -0.177, ..., 0.276]ᵀ
is  = [0.279, 0.790, -0.169, ..., 0.272]ᵀ
```

### Word2Vec Algorithm

**Setup:**
1. Large corpus of text (lots of words)
2. Fixed window size (e.g., $m = 2$)
3. Goal: Figure out probabilities $P(o|c)$ where:
   - $c$ = center word
   - $o$ = outside (context) word

**Example with window size 2:**
```
"Quick brown fox jumps over the rabbit"
  ↑     ↑     ↑    ↑     ↑
 W_{t-2} W_{t-1} W_t W_{t+1} W_{t+2}
```

When "fox" is the center word ($W_t$), we compute:
- $P(W_{t-2}|W_t)$ = P("Quick"|"fox")
- $P(W_{t-1}|W_t)$ = P("brown"|"fox")
- $P(W_{t+1}|W_t)$ = P("jumps"|"fox")
- $P(W_{t+2}|W_t)$ = P("over"|"fox")

**Likelihood Function:**
$$L = \prod_{t=1}^{T} \prod_{\substack{-m \leq j \leq m \\ j \neq 0}} P(W_{t+j}|W_t)$$

### Probability Representation

For every word $w$ in vocabulary $V$:
- $\vec{v}_w$ = vector when $w$ is a **center** word
- $\vec{u}_w$ = vector when $w$ is a **context** word

**Softmax Function:**
$$P(o|c) = \frac{\exp(\vec{u}_o^T \vec{v}_c)}{\sum_{w \in V} \exp(\vec{u}_w^T \vec{v}_c)}$$

**Properties:**
- Sum of all probabilities = 1
- All values are positive and $\leq 1$

**Intuition:**
- If word 1 is similar to $c$: $\exp(\vec{u}_1^T \vec{v}_c)$ is large
- If words 2 and 3 never appear in context of $c$: 
  - $\exp(\vec{u}_2^T \vec{v}_c) \approx 0$
  - $\exp(\vec{u}_3^T \vec{v}_c) \approx 0$

This creates word embeddings where similar words have similar vectors in the embedding space.

---

## Symbol Legend

**Mathematical Notation:**
- $\vec{v}$ or $\vec{u}$ = vector (indicated by arrow)
- $v^T$ = transpose of vector $v$
- $\in$ = "is an element of" or "belongs to"
- $\mathbb{R}^{m \times n}$ = set of real-valued matrices with $m$ rows and $n$ columns
- $\prod$ = product operator (multiplication across all terms)
- $\sum$ = summation operator (addition across all terms)
- $\nabla$ = gradient operator (vector of partial derivatives)
- $\partial$ = partial derivative symbol
- $\sigma$ = sigmoid activation function
- $\exp(x) = e^x$ = exponential function
- $\log$ = natural logarithm (base $e$)
- $\approx$ = approximately equal to
- $\alpha$ = alpha (learning rate)
- $\ell$ = loss function

**Model-Specific Notation:**
- $\hat{y}$ = predicted output (y-hat)
- $W^{(i)}$ = weight matrix for layer $i$
- $\vec{b}^{(i)}$ = bias vector for layer $i$
- $c$ = center word (in Word2Vec)
- $o$ = outside/context word (in Word2Vec)
- $V$ = vocabulary (set of all words)
- $T$ = total number of words in corpus
- $m$ = window size (context window radius)
- $d$ = dimension of input/embedding