# CS 6180 - Lecture 4: From Word2Vec to GloVe & Introduction to RNNs
## Date: 09/15/2025

## Part 1: Addressing Word2Vec's Computational Limitations

### Review: Word2Vec Model
- **Goal**: Learn vector representations for words
- **Key formula**: 
$P(O = o | C = c) = \frac{\exp(\vec{u}_o^T \vec{v}_c)}{\sum_{w \in V} \exp(\vec{u}_w^T \vec{v}_c)}$

- **Gradient derived**: 
$\frac{\partial L}{\partial \vec{v}_c} = \vec{u}_o - \sum_{w \in V} \vec{u}_w \cdot P(O = w | C = c)$

### The Computational Problem
**Key limitation**: The denominator requires summing over the entire vocabulary $V$
- This is computationally expensive for large vocabularies (e.g., 100,000+ words)
- Need to compute this for every training example
- **Solution needed**: Find another approach to avoid computing this expensive normalization

## Part 2: GloVe - Global Vectors Approach

### Co-occurrence Matrix
Instead of predicting context words, GloVe works directly with word co-occurrence statistics:

**Define co-occurrence matrix $X$:**
- $X_{ij}$ = number of times word $j$ appears in the context of word $i$
- $X_i = \sum_k X_{ik}$ = total number of times word $i$ appears as a center word

**Probability formulation:**
- $P_{ij} = \frac{X_{ij}}{X_i}$ = empirical probability of seeing word $j$ in context of word $i$
- $Q_{ij} = P(O = j | C = i)$ = model's predicted probability

### Objective Function Transformation

Starting from the negative log-likelihood over the corpus:

$$-\sum_{i \in \text{corpus}} \sum_{j \in \text{context}(i)} \log Q_{ij}$$

Since the same word pairs $(i,j)$ appear multiple times in the corpus, we can rewrite this as:

$$L = -\sum_{i=1}^{V} \sum_{j=1}^{V} X_{ij} \log Q_{ij}$$

Where:
- $V$ = vocabulary size
- $X_{ij}$ acts as a weight for how often this pair occurs
- This formulation works with pre-computed statistics rather than online computation

### GloVe Objective Function Development

**Initial idea**: We want the model's prediction to match the co-occurrence counts:
$\exp(\vec{u}_i^T \vec{v}_j) \approx X_{ij}$

**First attempt** (problematic):
$\sum_i \sum_j (\exp(\vec{u}_i^T \vec{v}_j) - X_{ij})^2$

**Problem**: The exponential of dot products can be very large → overflow errors!

**Solution**: Take logarithms before squaring

$\sum_i \sum_j f(X_{ij}) \left[\log(\exp(\vec{u}_i^T \vec{v}_j)) - \log(X_{ij})\right]^2$

Simplifying:
$\sum_i \sum_j f(X_{ij}) \left[\vec{u}_i^T \vec{v}_j - \log(X_{ij})\right]^2$

Where:
- $f(X_{ij})$ is a weighting function (to be defined)
- The log-exp cancels out to just the dot product
- This formulation avoids numerical overflow issues

### Key Insights
1. **Numerical stability**: Working in log-space prevents overflow
2. **Weighted loss**: Not all co-occurrences are equally important
3. **Symmetric formulation**: Can treat center and context vectors more symmetrically

### Advantages of GloVe
1. **Efficiency**: Works with pre-computed co-occurrence matrix
2. **Global statistics**: Captures corpus-wide patterns
3. **No expensive normalization**: Avoids the softmax denominator computation at each step
4. **Numerical stability**: Log-space computation prevents overflow

---

## Part 3: Language Modeling

### The Task
**Language Modeling**: Predict what word comes next in a sequence

**Example**: 
"The students opened their ____"
- High probability words: books, laptops, bottles
- Low probability words: sadness, purple, jumped

### Mathematical Formulation

Given a sequence of $t$ words:
$\vec{x}^{(1)}, \vec{x}^{(2)}, \ldots, \vec{x}^{(t)}$

**Goal**: Compute the probability of the next word:
$P(x^{(t+1)} | \vec{x}^{(1)}, \vec{x}^{(2)}, \ldots, \vec{x}^{(t)})$

### Probability of a Complete Sentence

A language model assigns a probability to an entire sentence using the **chain rule of probability**:

$P(\vec{x}^{(1)}, \vec{x}^{(2)}, \ldots, \vec{x}^{(T)}) = P(\vec{x}^{(1)}) \cdot P(\vec{x}^{(2)} | \vec{x}^{(1)}) \cdot P(\vec{x}^{(3)} | \vec{x}^{(1)}, \vec{x}^{(2)}) \cdots P(\vec{x}^{(T)} | \vec{x}^{(1)}, \ldots, \vec{x}^{(T-1)})$

Or more compactly:
$P(\vec{x}^{(1:T)}) = \prod_{t=1}^{T} P(\vec{x}^{(t)} | \vec{x}^{(1:t-1)})$

### Applications
- **Word suggestions**: Autocomplete in search bars, text editors
- **Large Language Models**: Foundation of GPT, BERT, and other modern AI systems
- **Machine translation**: Generating fluent translations
- **Speech recognition**: Choosing between similar-sounding phrases

### Learning Approach
**To learn these conditional probabilities, we will use neural networks**
- Challenge: How to handle variable-length history?
- Solution: Recurrent Neural Networks (RNNs) - process sequences step by step
- Key idea: Maintain a "memory" of previous context

---

## Part 4: Recurrent Neural Networks (RNNs)

### Core Concept
**RNNs use their outputs as inputs for the next generation** - creating a "memory" mechanism

### RNN Architecture

```
Initial state: h^(0) → h^(1) → h^(2) → h^(3) → h^(4) → ...
                         ↑       ↑       ↑       ↑
                        W_e     W_e     W_e     W_e
                         ↑       ↑       ↑       ↑
                        e_1     e_2     e_3     e_4
                         ↑       ↑       ↑       ↑
                        "The" "students" "opened" "their"
                        x^(1)   x^(2)    x^(3)    x^(4)
```

### Key Components

1. **Input sequence**: $x^{(1)}, x^{(2)}, x^{(3)}, x^{(4)}, \ldots$
   - Example: "The", "students", "opened", "their"

2. **Word Embeddings**: 
   - Convert words to vectors using an **Embedding Matrix** $E$
   - $\vec{e}_t = E \cdot \vec{x}^{(t)}$ (where $\vec{x}^{(t)}$ is one-hot encoded)
   - This matrix $E$ combines the $\vec{u}$ and $\vec{v}$ vectors from Word2Vec/GloVe

3. **Hidden States**: $\vec{h}^{(0)}, \vec{h}^{(1)}, \vec{h}^{(2)}, \ldots$
   - $\vec{h}^{(t)}$ captures information from all previous words
   - Acts as the "memory" of the network

4. **Recurrence Relation**:
   $\vec{h}^{(t)} = f(\vec{h}^{(t-1)}, \vec{e}_t)$
   
   More specifically:
   $\vec{h}^{(t)} = \tanh(W_h \vec{h}^{(t-1)} + W_e \vec{e}_t + \vec{b})$

### Weight Sharing
**Critical insight**: The same weights ($W_h$, $W_e$) are used at every time step
- This allows the network to process sequences of any length
- Enables the network to learn patterns that work regardless of position

### Mathematical Formulation

#### Step 1: Word to Embedding
- $\vec{x}^{(t)}$ = one-hot vector representation of word at time $t$
- $\vec{e}^{(t)} = E \cdot \vec{x}^{(t)}$ where $E \in \mathbb{R}^{d \times |V|}$ is the embedding matrix

#### Step 2: Hidden State Update
$\vec{h}^{(t)} = \sigma(W_h \vec{h}^{(t-1)} + W_e \vec{e}^{(t)} + \vec{b}_1)$

Where:
- $\sigma$ = nonlinear activation function (typically tanh or ReLU)
- $W_h \in \mathbb{R}^{d_h \times d_h}$ = hidden-to-hidden weight matrix
- $W_e \in \mathbb{R}^{d_h \times d}$ = embedding-to-hidden weight matrix
- $\vec{b}_1 \in \mathbb{R}^{d_h}$ = bias vector
- $d_h$ = hidden state dimension

#### Step 3: Computing Hidden States Sequentially
- $\vec{h}^{(0)}$ = initial hidden state (often initialized to zeros)
- $\vec{h}^{(1)} = \sigma(W_h \vec{h}^{(0)} + W_e \vec{e}^{(1)} + \vec{b}_1)$
- $\vec{h}^{(2)} = \sigma(W_h \vec{h}^{(1)} + W_e \vec{e}^{(2)} + \vec{b}_1)$
- ...
- $\vec{h}^{(k)} = \sigma(W_h \vec{h}^{(k-1)} + W_e \vec{e}^{(k)} + \vec{b}_1)$

#### Step 4: Output Generation
Project the hidden state back to vocabulary space:

$\vec{y}^{(t)} = \text{softmax}(U \vec{h}^{(t)} + \vec{b}_2)$

Where:
- $U \in \mathbb{R}^{|V| \times d_h}$ = output projection matrix
- $\vec{b}_2 \in \mathbb{R}^{|V|}$ = output bias
- $\vec{y}^{(t)} \in \mathbb{R}^{|V|}$ = probability distribution over vocabulary

### Complete RNN Forward Pass

For input sequence "The students opened their":

1. **Convert to embeddings**: 
   - "The" → $\vec{x}^{(1)}$ → $\vec{e}^{(1)}$
   - "students" → $\vec{x}^{(2)}$ → $\vec{e}^{(2)}$
   - etc.

2. **Process sequentially**:
   - $\vec{h}^{(1)}$ captures "The"
   - $\vec{h}^{(2)}$ captures "The students"
   - $\vec{h}^{(3)}$ captures "The students opened"
   - $\vec{h}^{(4)}$ captures "The students opened their"

3. **Predict next word**:
   - $\vec{y}^{(4)}$ = probability distribution over all possible next words
   - High probability for: "books", "laptops", "bags"
   - Low probability for: "jumped", "purple", "sadly"

### Information Flow Summary
- Each hidden state $\vec{h}^{(t)}$ contains:
  - Direct information from current word embedding $\vec{e}^{(t)}$
  - Compressed information from all previous words via $\vec{h}^{(t-1)}$
- The network learns how much information to use from previous context vs. current input
