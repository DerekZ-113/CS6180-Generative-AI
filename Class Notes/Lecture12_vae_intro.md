
# CS 6180 Lecture 12: Introduction to Variational Autoencoders (VAEs)
**Date:** October 20, 2025  
**Topic:** Generative Models for Images - VAE Fundamentals

**Administrative Notes:**
- HW3 released tonight (1.5 weeks, continuation of HW2 - part on transformers)
- Email on Canvas with Gradescope details
- Login with Northeastern account at www.gradescope.com

---

## Overview

This lecture introduces **Variational Autoencoders (VAEs)**, a generative model architecture for creating new images. VAEs are particularly effective at generating realistic images of faces, including faces of people that don't exist. The lecture covers the encoder-decoder framework, latent variables, and the mathematical foundations needed to understand VAE training.

---

## 1. Generative Models for Images

### The Goal

**Task:** Generate new, realistic images that don't exist in the training data.

**Example application:** Generating faces of people that don't exist
- VAEs are particularly good at this task
- Can generate diverse, realistic facial images

### Encoder-Decoder Framework for Images

**Architecture:**
```
Image of faces → [Encoder] → Important features → [Decoder] → Generate new images
```

**Key idea:** Extract **important features** from images, then use those features to generate new images of faces.

---

## 2. Modeling Images with Probability Distributions

### Decomposing Image Generation

**Approach 1: Direct modeling (complex)**

Model the full image probability directly:
$$P(X)$$

Where $X$ represents all pixels in the image.

**Relevant features in faces:**
- Eye color
- Eye shape  
- Nose shape
- Shape of the face
- Eyebrows (color/shape)
- Hair
- Many more complex features...

### Approach 2: Conditional modeling (simpler)

**Key insight:** Break down the problem using conditional probabilities.

Instead of modeling $P(X)$ directly, model:

$$P(X | z) \cdot P(z_1) \cdot P(z_2) \cdots P(z_k)$$

Where:
- $z = (z_1, z_2, \ldots, z_k)$ = **latent variables** (hidden features)
- $z_1$ might represent eye color
- $z_2$ might represent hair color
- $z_k$ might represent eyebrows

**Example breakdown:**
$$P(X) = P(X | z_1, z_2, \ldots, z_k) \cdot P(z_1) \cdot P(z_2) \cdots P(z_k)$$

**Why this is simpler:**
- Each $P(z_i)$ is easier to learn than the full $P(X)$
- Conditional dependencies are more tractable
- Can model each feature distribution independently

---

## 3. Latent Variables

### Definition

**Latent variables:** Variables that are **not observed** but influence the observed data.

```
Latent (not observed):   Hair color  Eye shape  Eyebrows  Face shape
                             ○          ○          ○          ○
                              ╲          │         ╱           ╱
                               ╲         │        ╱           ╱
                                ╲        │       ╱           ╱
                                 ╲       │      ╱           ╱
                                  ↘      ↓     ↙           ↙
Observed:                              Image X
                                      (pixels)
```

**How many features are relevant?** This is a modeling choice - we decide how many latent dimensions $k$ to use.

### Example: Simple 2×2 Image

Consider a tiny 2×2 pixel image:

```
┌──────┬──────┐
│ 0.01 │ 0.27 │
├──────┼──────┤
│ 0.29 │ 0.01 │
└──────┴──────┘
```

**Observation:** Values along both diagonals are similar!
- Main diagonal: ~0.01, ~0.01
- Anti-diagonal: ~0.27, ~0.29

**Latent space representation:**
Instead of learning 4 independent values, we can learn:
- 1 value specifying the main diagonal
- 1 value specifying the anti-diagonal

This is the **space of possible images** - a compressed latent representation.

---

## 4. Background: Normal Distributions and Central Limit Theorem

### Why Normal Distributions?

**Central Limit Theorem:** When you sum up many random variables, the result approaches a normal distribution.

**Example:** Sum of Bernoulli random variables (coin flips)

Let $X_1, X_2, X_3 \sim \text{Bernoulli}(1/2)$ where:
$$P(X_i) = \begin{cases} 1/2 & \text{if } X_i = 0 \text{ (tails)} \\ 1/2 & \text{if } X_i = 1 \text{ (heads)} \end{cases}$$

Define $Y = X_1 + X_2 + X_3$:

| Y | P(Y) | Outcomes |
|---|------|----------|
| 3 | 1/8 | HHH |
| 2 | 3/8 | HHT, HTH, THH |
| 1 | 3/8 | HTT, THT, TTH |
| 0 | 1/8 | TTT |

**Outcomes for $Y = 2$ (two heads, one tail):**
- HHT
- HTH
- THH

As $n \rightarrow \infty$, the distribution of $Y = X_1 + X_2 + \cdots + X_n$ approaches a normal distribution!

### Normal Distribution Properties

A normal distribution is characterized by:
$$\mathcal{N}(\mu, \sigma^2)$$

Where:
- $\mu$ = mean (center of distribution)
- $\sigma^2$ = variance (spread of distribution)

**Standard normal:** $\mathcal{N}(0, 1)$ with mean 0 and variance 1.

**Key property:** The normal distribution has the **largest entropy** among all distributions with the same mean and variance.

---

## 5. Back to VAEs: Learning the Latent Distribution

### The Challenge

To properly train the model and generate images from the region where most data points come from, we need to figure out:
- **Means:** $\mu_1, \mu_2, \ldots, \mu_k$
- **Variances:** $\sigma_1^2, \sigma_2^2, \ldots, \sigma_k^2$

for each latent dimension.

### VAE Architecture Components

**Encoder:** Maps images to latent space
- Input: Image $X$ (e.g., 4 pixels: 0.8, 0.1, 0.1, 0.8)
- Output: Latent representation $z$

**Decoder:** Maps latent space back to images
- Input: Latent code $z$
- Output: Reconstructed/generated image

### Linear vs Non-Linear Encoders

**Linear encoder (limited):**
```
0.8 ○──╲
        ╲  ½     ○ 0.8
0.1 ○────╋──½──→ ○ 0.1
        ╱  ½     ○ 0.1
0.1 ○──╱  ½
        
0.8 ○──────────→ ○ 0.8
```

**Problem:** Linear transformations can only represent simple patterns.

**Solution:** Add non-linearities (activation functions) to represent more complex images.
- Use sigmoid, tanh, or ReLU activations
- Enables learning of complex patterns and features

---

## 6. Comparing Distributions: KL Divergence

### The Need for a Metric

**Problem:** We need some metric that allows us to compare two probability distributions $p$ and $q$.

### KL Divergence Definition

**Kullback-Leibler (KL) Divergence:**

$$\text{KL}(p \| q) = \sum_i p_i \log\left(\frac{p_i}{q_i}\right)$$

Or in expectation form:
$$\text{KL}(p \| q) = \mathbb{E}_{x \sim p}\left[\log\frac{p(x)}{q(x)}\right]$$

**Example calculation:**

Distribution $p$: $[0.1, 0.2, 0.4, 0.2, 0.1]$
Distribution $q$: $[0.15, 0.15, 0.3, 0.2, 0.1]$

$$\text{KL}(p \| q) = 0.1\log\left(\frac{0.1}{0.15}\right) + 0.2\log\left(\frac{0.2}{0.15}\right) + \cdots + 0.1\log\left(\frac{0.1}{0.1}\right)$$

---

## 7. Background on Information Theory

### Entropy

**Definition:** Entropy measures the **amount of chaos** or **surprise** in a distribution.

$$\text{Entropy}(p) = \mathbb{E}_{x \sim p}[-\log p(x)] = -\sum_x p(x)\log p(x)$$

**Intuition:**
- **Common events** ($p(x)$ high): Not much new information/content
- **Rare events** ($p(x)$ low): Lots of new information that we didn't know before

**Example calculations:**

**Case 1:** Deterministic distribution
$$p(x) = \begin{cases} 1 & \text{for one value} \\ 0 & \text{for all others} \end{cases}$$

$$\text{Entropy}(p) = -1 \cdot \log(1) - 0 \cdot \log(0) = 0$$

No surprise - we always know what will happen!

**Case 2:** Uniform distribution over 2 outcomes
$$p(x) = 1/2 \text{ for both values}$$

$$\text{Entropy}(p) = -\frac{1}{2}\log\left(\frac{1}{2}\right) - \frac{1}{2}\log\left(\frac{1}{2}\right) = -\log\left(\frac{1}{2}\right) = \log(2)$$

(Using logarithm base $e$)

**Key insight:** Normal distribution has a **huge entropy** - it's the distribution with the **largest entropy** among all distributions with the same mean and variance.

---

## 8. Cross-Entropy

### Definition

**Cross-entropy** between true distribution $p$ and model distribution $q$:

$$\text{CE}(p, q) = -\mathbb{E}_{x \sim p}[\log q(x)] = -\sum_x p(x)\log q(x)$$

**Interpretation:**
- $p(x)$ = true distribution of data (e.g., $p_{\text{data}}(x)$)
- $q(x)$ = model distribution we are currently using (e.g., $p_\theta(x)$)
- Cross-entropy measures the **average amount of surprise** we see when using $q(x)$ instead of the actual true distribution $p(x)$

**Intuition:** If our model $q$ matches the data distribution $p$ well, cross-entropy will be low (not much extra surprise).

---

## 9. Relationship: Entropy, Cross-Entropy, and KL Divergence

### The Connection

Starting from the KL divergence definition:

$$\text{KL}(p \| q) = \mathbb{E}_p\left[\log\frac{p(x)}{q(x)}\right] = \mathbb{E}_p[\log p(x)] - \mathbb{E}_p[\log q(x)]$$

$$= -\text{Entropy}(p) - \mathbb{E}_p[\log q(x)]$$

$$= -\text{Entropy}(p) + \text{CE}(p, q)$$

**Therefore:**
$$\boxed{\text{KL}(p \| q) = \text{CE}(p, q) - \text{Entropy}(p)}$$

**Key insight:** 
- Entropy of $p$ is fixed (property of the data)
- Minimizing cross-entropy is equivalent to minimizing KL divergence
- When $\text{KL}(p \| q) = 0$, we have $\text{CE}(p, q) = \text{Entropy}(p)$ (best possible)

---

## 10. Jensen's Inequality: Proving KL ≥ 0

### Statement of Jensen's Inequality

For a **convex function** $f$ (curves upward):

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

**Geometric interpretation:**
```
f(x)
  ↑
  │     ╱────
  │    ╱      
  │   ╱    ● f(b)
  │  ╱    ╱
  │ ╱  ● ╱ ← function value at E[X]
  │╱  ╱
f(a)● ╱────────────── ← line connecting f(a) and f(b)
  │ ╱
──┴────────────→
  a  (a+b)/2  b
     = E[X]
```

The function always lies **below** the line connecting any two points.

**For random variable $X$ taking values $a$ or $b$ with equal probability:**
- $\mathbb{E}[X] = (a+b)/2$
- $\mathbb{E}[f(X)] = \frac{1}{2}f(a) + \frac{1}{2}f(b)$ (on the red line)
- $f(\mathbb{E}[X]) = f((a+b)/2)$ (on the curve, below the line)

Therefore: $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$ ✓

### Applying Jensen's to KL Divergence

**Goal:** Show that $\text{KL}(p \| q) \geq 0$ for any two distributions $p$ and $q$.

**Proof:**

Starting with KL divergence:
$$\text{KL}(p \| q) = \mathbb{E}_p\left[\log\frac{p}{q}\right] = \mathbb{E}_p\left[-\log\frac{q}{p}\right]$$

Let $f(x) = -\log(x)$. 

**Is $f$ convex?** 
```
  ↑
  │╲
  │ ╲
  │  ╲___
  │      ╲____
  │           ╲_____
──┴──────────────────→
```

Yes! The function $f(x) = -\log(x)$ is convex (curves upward/downward depending on perspective).

**Apply Jensen's inequality:**

Since $f(x) = -\log(x)$ is convex:
$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

Therefore:
$$-\log(\mathbb{E}[X]) \leq \mathbb{E}[-\log(X)]$$

**Applying to KL divergence:**

$$\text{KL}(p \| q) = \mathbb{E}_p\left[-\log\frac{q}{p}\right]$$

By Jensen's inequality:
$$\geq -\log\left(\mathbb{E}_p\left[\frac{q}{p}\right]\right)$$

$$= -\log\left(\sum_i p_i \cdot \frac{q_i}{p_i}\right)$$

$$= -\log\left(\sum_i q_i\right)$$

$$= -\log(1) = 0$$

**Therefore:** $\boxed{\text{KL}(p \| q) \geq 0}$

**Equality condition:** $\text{KL}(p \| q) = 0 \Leftrightarrow p = q$ (distributions are identical)

---

## Key Takeaways

### 1. VAE Framework
- **Encoder:** Extracts important latent features from images
- **Decoder:** Generates new images from latent features
- More effective than direct image generation

### 2. Latent Variables
- Hidden variables not directly observed
- Represent meaningful features (eye color, hair, face shape)
- Dimensionality $k$ is a modeling choice
- Enable more tractable probability modeling

### 3. Conditional Decomposition
- $P(X) = P(X|z_1, z_2, \ldots, z_k) \cdot P(z_1) \cdot P(z_2) \cdots P(z_k)$
- Simpler to model than direct $P(X)$
- Each component distribution is easier to learn

### 4. Normal Distributions
- Central Limit Theorem: sums of random variables → normal
- Maximum entropy property
- Natural choice for latent variable distributions

### 5. Information Theory Foundations
- **Entropy:** Measures surprise/chaos in a distribution
- **Cross-Entropy:** Measures surprise using model $q$ instead of true $p$
- **KL Divergence:** Measures difference between distributions

### 6. KL Divergence Properties
- Always non-negative: $\text{KL}(p \| q) \geq 0$
- Zero iff distributions are identical: $\text{KL}(p \| q) = 0 \Leftrightarrow p = q$
- Relationship: $\text{KL}(p \| q) = \text{CE}(p, q) - \text{Entropy}(p)$
- Not symmetric: $\text{KL}(p \| q) \neq \text{KL}(q \| p)$

### 7. Training Objective Preview
- Need to determine $\mu_1, \mu_2, \ldots, \mu_k$ and $\sigma_1^2, \sigma_2^2, \ldots, \sigma_k^2$
- Generate images from regions where most data points occur
- Use KL divergence to compare learned distribution to target

---

## Mathematical Notation Legend

### Probability and Distributions
- $P(X)$ = probability distribution over images
- $p(x)$ = probability of specific value $x$
- $q(x)$ = model/approximate distribution
- $p_{\text{data}}(x)$ = true data distribution
- $p_\theta(x)$ = parameterized model distribution

### Latent Variables
- $z$ = latent variable vector
- $z_i$ = individual latent dimension
- $k$ = number of latent dimensions
- $P(X | z)$ = conditional probability of image given latent code

### Normal Distribution
- $\mathcal{N}(\mu, \sigma^2)$ = normal distribution with mean $\mu$, variance $\sigma^2$
- $\mu$ = mean
- $\sigma^2$ = variance
- $\mathbb{E}[X]$ = expected value

### Information Theory
- $\text{Entropy}(p)$ = entropy of distribution $p$
- $\text{CE}(p, q)$ = cross-entropy between $p$ and $q$
- $\text{KL}(p \| q)$ = KL divergence from $q$ to $p$

### Mathematical Operators
- $\mathbb{E}[\cdot]$ = expectation operator
- $\sum$ = summation
- $\log$ = natural logarithm
- $\sim$ = "distributed as"
- $\Leftrightarrow$ = "if and only if"

### Jensen's Inequality
- $f$ = convex function
- Convex: curves upward (function below connecting line)
- Concave: curves downward (function above connecting line)