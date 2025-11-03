# CS 6180 Lecture 13: VAE Training and Evidence Lower Bound
**Date:** October 22, 2025  
**Topic:** VAE Optimization, ELBO Derivation, and Training Objective

**Administrative Note:** HW3 released tomorrow morning

---

## Overview

This lecture develops the complete training framework for Variational Autoencoders (VAEs). We derive the Evidence Lower Bound (ELBO), decompose it into interpretable components, and specify the complete VAE model for generating MNIST digits. The lecture bridges theoretical foundations from Lecture 12 to practical implementation.

---

## 1. Review: VAE Framework

### Core Components

**Encoder:** Learns important features from images
- Input: Image $X$
- Output: Latent representation $z = (z_1, z_2, \ldots, z_k)$

**Decoder:** Generates new images from latent features
- Input: Latent code $z$
- Output: Generated/reconstructed image

**Latent variables:** $z_1, z_2, \ldots, z_k$ (not observed)
- Represent important features in the images
- For simplicity in examples, we often consider two features: $z_1, z_2$

### Latent Space Visualization

```
      z₂
       ↑
       │     ┌────────┐
       │    │  Image  │
       │    │    X    │
       │     └────────┘
       │          ↓
       │     (multiple similar 
       │      images in this
       │      region)
       │
       │
       └──────────────→ z₁
```

The latent space contains regions where similar images cluster together.

---

## 2. Review: KL Divergence Properties

### Definition
$$\text{KL}(p \| q) = \sum_x p(x) \log\left(\frac{p(x)}{q(x)}\right) = \mathbb{E}_{p(x)}\left[\log\frac{p(x)}{q(x)}\right]$$

### Key Properties

1. **Non-negativity:** $\text{KL}(p \| q) \geq 0$ (proven via Jensen's inequality)

2. **Zero iff identical:** $\text{KL}(p \| q) = 0 \Leftrightarrow p = q$

3. **Asymmetry:** $\text{KL}(p \| q) \neq \text{KL}(q \| p)$

### Asymmetry Example

If $\frac{q(x)}{p(x)} = c$ (constant), then:

$$q(x) = c \cdot p(x)$$

$$\Rightarrow \sum_x q(x) = \sum_x c \cdot p(x) = c \cdot \sum_x p(x) = c \cdot 1 = c$$

Therefore $c = 1$, which means $q(x) = p(x)$.

**Comparing the two directions:**
$$\text{KL}(p \| q) = \sum_x p(x)\log\left(\frac{p(x)}{q(x)}\right)$$
$$\text{KL}(q \| p) = \sum_x q(x)\log\left(\frac{q(x)}{p(x)}\right)$$

These are generally **not equal** (antisymmetric).

### Which KL Divergence to Use?

For VAEs, we care about:

$$\boxed{\text{KL}(p_{\text{data}} \| p_\theta)}$$

Where:
- $p_{\text{data}}$ is the correct/true distribution
- $p_\theta$ is our model distribution
- We are computing the error we get if we were to use $p_\theta$ instead of the true distribution

---

## 3. Jensen's Inequality: When Does Equality Hold?

### Equality Condition

Jensen's inequality states for convex $f$:
$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

**This becomes an equality if:**
$$X = \mathbb{E}[X]$$

i.e., $X$ is just a constant (no randomness).

**Proof of equality case:**

If $X = c$ with probability 1, then:
$$\mathbb{E}[X] = c$$

Left side:
$$f(\mathbb{E}[X]) = f(c)$$

Right side:
$$\mathbb{E}[f(X)] = \mathbb{E}[f(c)] = f(c)$$

Therefore: $f(\mathbb{E}[X]) = \mathbb{E}[f(X)]$ ✓

---

## 4. Optimization Challenge: The Likelihood Function

### Dataset and Goal

**Given:** $n$ images $\vec{X}^{(1)}, \vec{X}^{(2)}, \vec{X}^{(3)}, \ldots, \vec{X}^{(n)}$
- Each image is turned into a vector (flatten pixels)

**Goal:** Maximize the likelihood of the data occurring

### Likelihood Function

Assuming independent samples:
$$\mathcal{L} = p(X^{(1)}) \cdot p(X^{(2)}) \cdot p(X^{(3)}) \cdots p(X^{(n)})$$

$$= \prod_{i=1}^{n} p(X^{(i)})$$

**Challenge:** We want $p(X^{(1)})$ high, $p(X^{(2)})$ high, ..., $p(X^{(n)})$ high.

**Problem:** Derivatives of products are not fun (product rule becomes messy).

### Log-Likelihood Function

**Solution:** Take the logarithm (monotonic transformation preserves optimum):

$$\log \mathcal{L} = \log\left(\prod_{i=1}^{n} p(X^{(i)})\right) = \sum_{i=1}^{n} \log p(X^{(i)})$$

**Advantage:** Sum of logarithms is much easier to differentiate than product!

---

## 5. The Challenge: Computing log p(X)

### Focus on One Example

Let's focus on optimizing $\log p_\theta(X)$ for a single image $X$:

$$\log p_\theta(X) = \log\left(\sum_z p(X|z) \cdot p(z)\right)$$

**Problem:** This is still very hard to optimize!
- Sum over all possible $z$ values
- Inside the logarithm (not convex)
- Cannot apply standard convex optimization techniques

**Strategy needed:** Come up with an approach to help us optimize the log-likelihood function.

---

## 6. The ELBO Derivation

### Introducing an Auxiliary Distribution Q(z)

**Key idea:** Introduce an arbitrary probability distribution $Q(z)$ over the latent variables.

Starting with the log-likelihood for one example:

$$\log p_\theta(X) = \log\left(\sum_z p(X, z)\right)$$

$$= \log\left(\sum_z \frac{p(X, z) \cdot Q(z)}{Q(z)}\right)$$

$$= \log\left(\mathbb{E}_{Q(z)}\left[\frac{p(X, z)}{Q(z)}\right]\right)$$

**Apply Jensen's inequality** (since $\log$ is concave, inequality flips):

For concave $g$: $g(\mathbb{E}[X]) \geq \mathbb{E}[g(X)]$

$$\log p_\theta(X) = \log\left(\mathbb{E}_{Q(z)}\left[\frac{p(X, z)}{Q(z)}\right]\right)$$

$$\geq \mathbb{E}_{Q(z)}\left[\log\frac{p(X, z)}{Q(z)}\right]$$

**This is the Evidence Lower Bound (ELBO)!**

$$\boxed{\log p_\theta(X) \geq \mathbb{E}_{Q(z)}\left[\log\frac{p(X, z)}{Q(z)}\right] = \text{ELBO}}$$

---

## 7. ELBO Decomposition

### Step-by-Step Derivation

Starting from the ELBO:

$$\mathbb{E}_{p(z|X)}\left[\log\frac{p(X, z)}{p(z|X)}\right]$$

Expand using $p(X, z) = p(X|z) \cdot p(z)$:

$$= \mathbb{E}_{p(z|X)}\left[\log\left(\frac{p(X|z) \cdot p(z)}{p(z|X)}\right)\right]$$

$$= \mathbb{E}_{p(z|X)}\left[\log p(X|z) + \log\frac{p(z)}{p(z|X)}\right]$$

Split the expectation:

$$= \mathbb{E}_{p(z|X)}[\log p(X|z)] + \mathbb{E}_{p(z|X)}\left[\log\frac{p(z)}{p(z|X)}\right]$$

The second term is a KL divergence:

$$= \mathbb{E}_{p(z|X)}[\log p(X|z)] - \sum_z p(z|X) \log\frac{p(z|X)}{p(z)}$$

$$= \mathbb{E}_{p(z|X)}[\log p(X|z)] - \text{KL}(p(z|X) \| p(z))$$

**Final ELBO form:**

$$\boxed{\log p_\theta(X) \geq \underbrace{\mathbb{E}_{p(z|X)}[\log p(X|z)]}_{\text{Reconstruction loss}} - \underbrace{\text{KL}(p(z|X) \| p(z))}_{\text{KL divergence term}}}$$

### Interpretation of Components

**Component 1: Reconstruction Loss** (hard to know what $p(z|X)$ is)
$$\mathbb{E}_{p(z|X)}[\log p(X|z)]$$
- Measures how well the decoder reconstructs $X$ from latent code $z$
- Want this to be high (good reconstruction)

**Component 2: KL Divergence Term**
$$\text{KL}(p(z|X) \| p(z))$$
- Measures how different the posterior $p(z|X)$ is from the prior $p(z)$
- Want this to be low (posterior stays close to prior)

---

## 8. The VAE Model for MNIST

### Problem Setup

**MNIST dataset:** Black and white images of handwritten digits
- Goal: Generate an image of a digit (0, 1, 2, ..., 9)
- Images: $\vec{X}^{(1)}, \vec{X}^{(2)}, \ldots, \vec{X}^{(n)}$ (n black/white images)
- Pixels: $x_i \in \{0, 1\}$ (binary: 0 = white, 1 = black)

### Model Components

**1. Prior Distribution (choose this):**
$$p(z_i) = \mathcal{N}(0, I) \quad \text{for } i = 1, \ldots, K$$

Where:
- $K$ = number of latent variables (hyperparameter)
- $I$ = identity matrix (standard normal)
- Each latent dimension is independent

**2. Likelihood Distribution (learn $\theta$):**
$$p(x_i | z) = \text{Bernoulli}(f_\theta(z))$$

Where:
- $f_\theta(z)$ is a neural network (decoder) with parameters $\theta$
- Outputs probability for each pixel being black (1) or white (0)

**3. Approximate Posterior (learn $\phi$):**
$$p(z | X) \approx q_\phi(z | X) = \mathcal{N}(\mu_\phi, \Sigma_\phi)$$

Where:
- $\mu_\phi, \Sigma_\phi$ are learned functions of $X$ (encoder outputs)
- $\phi$ represents encoder parameters

### How the Model Works

**Generation process:**

1. **First, generate random $z$ values:**
   $$z \sim \mathcal{N}(0, I)$$

2. **Use the $z$ values to generate a pixel:**
   $$p(x_i | z) = \text{Bernoulli}(f_\theta(z))$$

**Role of $z$:**
- Helps the model figure out **what area to focus on** (which digit to draw)
- Example: $z = 0.2$ might focus on the region for digit "6"

**Role of Bernoulli:**
- Allows the model to learn **within the area of interest** which pixels to set to 0 (white) and which pixels to set to 1 (black)
- Two colors: black/white

### Example: Generating a "6"

```
Region for "6":
┌───────────────┐
│ ○ ○         ○ │  ← Probabilities for each pixel
│               │
│  ╭───┐    ○.1│  ← Background (white, low prob)
│  │ ● │    0.1│
│  │ ● │  0.8  │  ← Stroke of "6" (black, high prob)
│  ╰───╯  0.1  │
│               │
│ ○ ○         ○ │
└───────────────┘

z = 0.1 → model focuses on "6" region
```

**Decoder output (Bernoulli parameters):**
```
┌──────┬──────┬──────┬──────┬──────┐
│ 0.5  │ 0.5  │ 0.5  │ 0.5  │ 0.5  │
├──────┼──────┼──────┼──────┼──────┤
│ 0.5  │ 0.5  │ 0.01 │ 0.5  │ 0.5  │
├──────┼──────┼──────┼──────┼──────┤
│ 0.5  │ 0.01 │ 0.55 │ 0.5  │ 0.5  │  ← Note: central region has
├──────┼──────┼──────┼──────┼──────┤     different probabilities
│ 0.5  │ 0.5  │ 0.5  │ 0.5  │ 0.5  │
├──────┼──────┼──────┼──────┼──────┤
│ 0.5  │ 0.5  │ 0.5  │ 0.5  │ 0.5  │
└──────┴──────┴──────┴──────┴──────┘
```

---

## 9. VAE Training Objective

### The ELBO for VAE

Substituting our model choices into the ELBO:

$$\log p_\theta(X) \geq \mathbb{E}_{q_\phi(z|X)}[\log p_\theta(X|z)] - \text{KL}(q_\phi(z|X) \| p(z))$$

**Training goal:** Maximize the ELBO with respect to both $\theta$ and $\phi$.

### The Two Terms

**Term 1: Reconstruction Loss**
$$\mathbb{E}_{q_\phi(z|X)}[\log p_\theta(X|z)]$$

**Term 2: KL Regularization**
$$\text{KL}(q_\phi(z|X) \| p(z))$$

**Note:** This is **not equal** to $\log p(X)$ anymore - it's a **lower bound**.

---

## 10. Why the Posterior Approximation?

### The True Posterior Problem

**What we really want:** $p(z|X)$ (the true posterior)

**Problem:** $p(z|X)$ is intractable to compute!

Using Bayes' rule:
$$p(z|X) = \frac{p(X|z) \cdot p(z)}{p(X)} = \frac{p(X|z) \cdot p(z)}{\sum_{z'} p(X|z') \cdot p(z')}$$

The denominator requires summing over all possible $z$ values - computationally infeasible for high-dimensional $z$.

### The Approximation Solution

**Solution:** Approximate $p(z|X)$ with a simpler distribution $q_\phi(z|X)$.

**Choice:** Use a Gaussian with learned parameters:
$$q_\phi(z|X) \approx \mathcal{N}(\mu_\phi(X), \Sigma_\phi(X))$$

Where:
- $\mu_\phi(X)$ = mean (function of $X$, output by encoder)
- $\Sigma_\phi(X)$ = covariance (function of $X$, output by encoder)
- $\phi$ = encoder parameters

**Key insight:** The encoder learns to map images to distributions over latent codes!

---

## 11. Complete VAE Architecture Summary

### Model Specification

**Prior (fixed):**
$$p(z) = \mathcal{N}(\vec{0}, I)$$

**Decoder (learn $\theta$):**
$$p_\theta(X|z) = \prod_{i} \text{Bernoulli}(x_i | f_\theta(z))$$

Where $f_\theta(z)$ is a neural network outputting Bernoulli parameters for each pixel.

**Encoder (learn $\phi$):**
$$q_\phi(z|X) = \mathcal{N}(\mu_\phi(X), \Sigma_\phi(X))$$

Where $\mu_\phi$ and $\Sigma_\phi$ are neural networks.

### Training Objective

**Maximize over all data points:**

$$\sum_{i=1}^{n} \text{ELBO}(X^{(i)})$$

$$= \sum_{i=1}^{n} \left[\mathbb{E}_{q_\phi(z|X^{(i)})}[\log p_\theta(X^{(i)}|z)] - \text{KL}(q_\phi(z|X^{(i)}) \| p(z))\right]$$

**Parameters to optimize:**
- $\theta$ (decoder parameters)
- $\phi$ (encoder parameters)

---

## 12. Hyperparameter: Number of Latent Variables

### The Choice of K

**Question:** How many latent variables should we use?

**Answer:** Currently it's fixed, but you will need to **play with different numbers to figure out the optimal one**.

**Trade-offs:**
- **Small $K$:** Simpler model, but may not capture all important features
- **Large $K$:** More expressive, but harder to train and may overfit

**In practice:** This is a hyperparameter you tune based on:
- Reconstruction quality
- Generation diversity
- Computational constraints

---

## Key Takeaways

### 1. Optimization Strategy
- Maximize log-likelihood instead of likelihood (easier derivatives)
- Cannot directly optimize $\log p(X)$ due to intractable sum
- Use ELBO as a tractable lower bound

### 2. ELBO Structure
$$\text{ELBO} = \text{Reconstruction} - \text{KL Regularization}$$
- Reconstruction encourages accurate image regeneration
- KL term keeps latent distribution close to prior

### 3. Jensen's Inequality Application
- Used to derive the ELBO
- Equality holds when random variable is constant
- Critical for proving ELBO is a valid lower bound

### 4. Posterior Approximation
- True posterior $p(z|X)$ is intractable
- Approximate with $q_\phi(z|X) = \mathcal{N}(\mu_\phi, \Sigma_\phi)$
- Encoder learns to produce distribution parameters

### 5. MNIST VAE Model
- Prior: $p(z) = \mathcal{N}(0, I)$
- Likelihood: $p(X|z) = \text{Bernoulli}(f_\theta(z))$
- Encoder: $q_\phi(z|X) = \mathcal{N}(\mu_\phi(X), \Sigma_\phi(X))$

### 6. Two-Stage Generation
- $z$ determines **what** to generate (which digit)
- Bernoulli determines **how** to generate (which pixels are black/white)

### 7. KL Divergence Direction
- Use $\text{KL}(p_{\text{data}} \| p_\theta)$ for training
- Asymmetry matters: forward vs reverse KL give different results

---

## Mathematical Notation Legend

### VAE Components
- $X$ = observed image (data)
- $z$ = latent variable vector
- $K$ = number of latent dimensions
- $\theta$ = decoder parameters
- $\phi$ = encoder parameters

### Probability Distributions
- $p(z)$ = prior distribution (standard normal)
- $p_\theta(X|z)$ = decoder/likelihood (Bernoulli)
- $p(z|X)$ = true posterior (intractable)
- $q_\phi(z|X)$ = approximate posterior (encoder output)
- $p_\theta(X)$ = marginal likelihood (evidence)

### Neural Network Functions
- $f_\theta(z)$ = decoder network
- $\mu_\phi(X)$ = encoder network for mean
- $\Sigma_\phi(X)$ = encoder network for covariance

### Information Theory Terms
- ELBO = Evidence Lower Bound
- $\mathcal{L}$ = likelihood function
- $\text{KL}(\cdot \| \cdot)$ = KL divergence

### Distributions
- $\mathcal{N}(\mu, \Sigma)$ = Normal/Gaussian distribution
- $\text{Bernoulli}(p)$ = Bernoulli distribution
- $I$ = identity matrix

### Operators
- $\mathbb{E}_{q}[\cdot]$ = expectation with respect to distribution $q$
- $\prod$ = product
- $\sum$ = summation
- $\log$ = natural logarithm