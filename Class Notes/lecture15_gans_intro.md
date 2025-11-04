# CS 6180 Lecture 15: Introduction to Generative Adversarial Networks (GANs)
**Date:** October 29, 2025  
**Topic:** GANs - Adversarial Training for Image Generation
**Quiz:** November 13, 2025

---

## Overview

This lecture introduces **Generative Adversarial Networks (GANs)**, a powerful approach to generative modeling that works through an adversarial game between two neural networks. GANs are particularly effective at generating realistic fake images.

**Key insight:** Once a VAE is trained, we can sample from $p(z) \approx q_\phi(z|X)$ and generate images. But can we do better without needing an encoder?

---

## 1. The Adversarial Framework

### The Core Idea

GANs consist of two neural networks working **against each other**:

1. **Generator ($G_\theta$):** Creates fake images trying to fool the discriminator
2. **Discriminator ($D_\phi$):** Tries to distinguish real images from fake ones

**Analogy: The Art Forger and the Critic**

**Generator (Xiwei):**
- Goal: Paint an exact replica of the Mona Lisa
- Wants to create paintings identical to the real thing
- Tries to fool the discriminator

**Discriminator (Julia):**
- Goal: Identify whether a painting is real or fake
- Sees both real Mona Lisas and Xiwei's forgeries
- Response: "Nop. What is that? Is that supposed to be the Mona Lisa?"

```
Generator: Creates "art" → [Discriminator] → "Real" or "Fake"?
                ↑                                    ↓
                └──────── Feedback signal ───────────┘
```

**The game:** As the generator improves, the discriminator must get better at detection. As the discriminator improves, the generator must create more realistic images.

---

## 2. Building the Discriminator

### Task Definition

**Input:** An image $x$ (either real or generated)
**Output:** A score indicating whether the image is real or fake

### Example: Simple 2×2 Images

**Real images:**
```
┌───────┬───────┐
│ 0.8   │ 0.05  │  → Strong diagonal pattern
├───────┼───────┤    Consistent with real data
│ 0.05  │ 0.9   │
└───────┴───────┘
```

**Fake images:**
```
┌───────┬───────┐
│ 0.3   │ 0.4   │  → Random noise
├───────┼───────┤    No clear pattern
│ 0.5   │ 0.2   │
└───────┴───────┘
```

### Simple Metric Approach (Naive)

For a 2×2 matrix with values $a_{11}, a_{12}, a_{21}, a_{22}$:

$$\text{score} = (a_{11} + a_{22}) - (a_{12} + a_{21})$$

**Examples:**
- Real image: score = $(0.8 + 0.9) - (0.05 + 0.05) = 1.4$ ✓
- Another real: score = $(0.9 + 0.9) - (0.05 + 0.05) = 1.7$ ✓
- Fake image: score = $(0.1 + 0.6) - (0.9 + 0.1) = -0.3$ ✓

**Problem with threshold approach:**
- Need to manually set threshold (e.g., > 1 → real, else → fake)
- Not differentiable
- Doesn't generalize

### Neural Network Approach

Instead of manual thresholds, use a neural network with **sigmoid activation**:

```
Image pixels → [Neural Network] → score → σ(score) → probability
(0.8, 0.1, 0.1, 0.8)      ↓
                       [Weights W]
                          ↓
                       D_φ(x) ∈ (0, 1)
```

**Discriminator output:**
$$D_\phi(x) = \sigma(\text{score}) = \frac{1}{1 + e^{-\text{score}}}$$

Where:
- $D_\phi(x) \approx 1$ → likely real
- $D_\phi(x) \approx 0$ → likely fake
- $\phi$ = discriminator parameters (weights)

**Advantage:** Smooth, differentiable output that can be trained with gradient descent.

---

## 3. Learning the Discriminator: Loss Function

### Goal

Learn weights $\phi$ (not just eyeballing them) to optimize discrimination.

### Loss for Real Images

**Setup:**
- Label: $y = 1$ (real image)
- Prediction: $D_\phi(x)$

**Desired behavior:**
- If prediction = 0.9 → small error (good!)
- If prediction = 0.1 → large error (bad!)

**Loss function:**
$$\mathcal{L}_{\text{real}} = -\log(D_\phi(x))$$

**Why this works:**
- When $D_\phi(x) \approx 1$: $-\log(1) \approx 0$ (small loss) ✓
- When $D_\phi(x) \approx 0$: $-\log(0) \rightarrow \infty$ (large loss) ✓

### Loss for Fake Images

**Setup:**
- Label: $y = 0$ (fake image from generator)
- Fake image: $G_\theta(z)$ where $z \sim \mathcal{N}(0, I)$
- Prediction: $D_\phi(G_\theta(z))$

**Desired behavior:**
- If prediction = 0.1 → small error (correctly identified as fake!)
- If prediction = 0.9 → large error (fooled by generator!)

**Loss function:**
$$\mathcal{L}_{\text{fake}} = -\log(1 - D_\phi(G_\theta(z)))$$

**Why this works:**
- When $D_\phi(G_\theta(z)) \approx 0$: $-\log(1-0) = -\log(1) \approx 0$ (small loss) ✓
- When $D_\phi(G_\theta(z)) \approx 1$: $-\log(1-1) = -\log(0) \rightarrow \infty$ (large loss) ✓

### Combined Discriminator Loss

**Average over real and fake images:**

$$\boxed{L_D(\phi, \theta) = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] - \mathbb{E}_{z \sim \mathcal{N}(0,I)}[\log(1 - D_\phi(G_\theta(z)))]}$$

**Interpretation:**
- **First term:** Correctly classify real images as real
- **Second term:** Correctly classify fake images as fake

**Training goal for discriminator:** Minimize $L_D(\phi, \theta)$ w.r.t. $\phi$

---

## 4. The Generator's Objective

### What the Generator Wants

**Generator's goal:** Fool the discriminator!

**Best case scenario for generator:**
- Label: $y = 0$ (this is a fake image)
- Prediction: $D_\phi(G_\theta(z)) = 0.95$ (discriminator thinks it's real!)
- This is **great** for the generator, **bad** for the discriminator

### Naive Generator Loss

**Intuition:** Generator wants discriminator to output low values for fake images.

From discriminator's perspective:
$$-\log(1 - D_\phi(G_\theta(z)))$$

**Generator wants to minimize this**, which means:

$$L_G^{(1)}(\phi, \theta) = -\mathbb{E}_{z \sim \mathcal{N}(0,I)}[\log(D_\phi(G_\theta(z)))]$$

**Equivalently, maximize:**
$$\max_\theta \mathbb{E}_{z}[\log(D_\phi(G_\theta(z)))]$$

### The Better Alternative

**Problem with naive loss:** Suffers from vanishing gradients!

**When $D_\phi(G_\theta(z)) \approx 0$ (generator is terrible):**
- $\log(D_\phi(G_\theta(z)))$ has very small gradient
- Learning is extremely slow

**Better approach:** Instead of **minimizing** $\log(1 - D_\phi(G_\theta(z)))$, **maximize** $\log(D_\phi(G_\theta(z)))$:

$$\boxed{L_G^{(2)}(\phi, \theta) = -\mathbb{E}_{z \sim \mathcal{N}(0,I)}[\log(D_\phi(G_\theta(z)))]}$$

**Why this is better:**
- When $D \approx 1$: $-\log(1) \approx 0$ → small gradient (generator already good)
- When $D \approx 0$: $-\log(0)$ → large gradient (generator needs improvement)

**Note:** This will be proven in the next homework that the alternative formulation has better gradient behavior!

---

## 5. The Complete GAN Training Objective

### The Min-Max Game

**Combined objective:**

$$\min_\theta \max_\phi \left[ \mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] + \mathbb{E}_{z}[\log(1 - D_\phi(G_\theta(z)))] \right]$$

**Interpretation:**
- **Discriminator** (maximize): Wants to correctly classify real and fake
- **Generator** (minimize): Wants to fool the discriminator

### Practical Training (Using Better Generator Loss)

**Step 1: Update Discriminator** (maximize discrimination)
$$\phi \leftarrow \phi + \alpha \nabla_\phi \left[ \mathbb{E}_{x}[\log D_\phi(x)] + \mathbb{E}_{z}[\log(1 - D_\phi(G_\theta(z)))] \right]$$

**Step 2: Update Generator** (maximize fooling ability)
$$\theta \leftarrow \theta + \alpha \nabla_\theta \mathbb{E}_{z}[\log(D_\phi(G_\theta(z)))]$$

**Alternate between these two steps.**

---

## 6. Deriving the Optimal Discriminator

### Exercise Setup

**Given fixed generator $G_\theta$, find optimal discriminator $D^*$ that maximizes:**

$$L_D(\phi, \theta) = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] - \mathbb{E}_{z}[\log(1 - D_\phi(G_\theta(z)))]$$

### Step 1: Rewrite in Terms of Distributions

**Real data distribution:** $x \sim p_{\text{data}}(x)$
**Generated distribution:** $G_\theta(z) \sim p_{\text{gen}}(x)$ where $z \sim \mathcal{N}(0, I)$

Rewrite the loss:

$$L_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] - \mathbb{E}_{x \sim p_{\text{gen}}}[\log(1 - D_\phi(x))]$$

$$= -\sum_x p_{\text{data}}(x) \log D_\phi(x) - \sum_x p_{\text{gen}}(x) \log(1 - D_\phi(x))$$

**Combine sums:**

$$L_D = -\sum_x \left[ p_{\text{data}}(x) \log D_\phi(x) + p_{\text{gen}}(x) \log(1 - D_\phi(x)) \right]$$

### Step 2: Take Derivative w.r.t. $D_\phi$

For optimal discriminator, set $\frac{\partial L_D}{\partial D_\phi} = 0$:

$$\frac{\partial L_D}{\partial D_\phi} = -\sum_x \left[ p_{\text{data}}(x) \cdot \frac{1}{D_\phi(x)} + p_{\text{gen}}(x) \cdot \frac{-1}{1 - D_\phi(x)} \right]$$

$$= -\sum_x \left[ \frac{p_{\text{data}}(x)}{D_\phi(x)} - \frac{p_{\text{gen}}(x)}{1 - D_\phi(x)} \right]$$

Setting this to zero:

$$\sum_x \left[ \frac{p_{\text{data}}(x)}{D_\phi(x)} - \frac{p_{\text{gen}}(x)}{1 - D_\phi(x)} \right] = 0$$

### Step 3: Solve for Each Term

**For optimality, each term in the sum must be zero** (equivalence to saying the entire sum is zero):

$$\frac{p_{\text{data}}(x)}{D_\phi(x)} - \frac{p_{\text{gen}}(x)}{1 - D_\phi(x)} = 0 \quad \forall x$$

**Rearrange:**

$$\frac{p_{\text{data}}(x)}{D_\phi(x)} = \frac{p_{\text{gen}}(x)}{1 - D_\phi(x)}$$

**Cross-multiply:**

$$p_{\text{data}}(x) \cdot (1 - D_\phi(x)) = p_{\text{gen}}(x) \cdot D_\phi(x)$$

$$p_{\text{data}}(x) - p_{\text{data}}(x) \cdot D_\phi(x) = p_{\text{gen}}(x) \cdot D_\phi(x)$$

$$p_{\text{data}}(x) = D_\phi(x) \cdot [p_{\text{gen}}(x) + p_{\text{data}}(x)]$$

**Solve for $D_\phi(x)$:**

$$\boxed{D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{gen}}(x) + p_{\text{data}}(x)} \quad \forall x}$$

### Interpretation of Optimal Discriminator

**When $p_{\text{data}}(x) = p_{\text{gen}}(x)$:**
$$D^*(x) = \frac{p_{\text{data}}(x)}{2p_{\text{data}}(x)} = \frac{1}{2}$$

The optimal discriminator outputs 0.5 (completely uncertain) when the generator perfectly matches the data distribution!

**When $p_{\text{gen}}(x) = 0$ (generator never produces this $x$):**
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x)} = 1$$

**When $p_{\text{data}}(x) = 0$ (this $x$ never appears in real data):**
$$D^*(x) = \frac{0}{p_{\text{gen}}(x)} = 0$$

---

## 7. Verification: Is This Really Optimal?

### Alternative Discriminators?

**Question:** Are there other discriminators that satisfy $\frac{\partial L_D}{\partial D_\phi} = 0$?

**Answer:** No! The condition:

$$\sum_x \left[ \frac{p_{\text{data}}(x)}{D_\phi(x)} - \frac{p_{\text{gen}}(x)}{1 - D_\phi(x)} \right] = 0$$

Requires **every term** in the sum to be zero, giving us the unique solution:

$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{gen}}(x) + p_{\text{data}}(x)}$$

### Why This is Optimal

This discriminator:
1. **Maximizes** the discriminator loss $L_D$
2. Provides the **strongest gradient signal** to the generator
3. At equilibrium ($p_{\text{gen}} = p_{\text{data}}$), outputs $D^* = 0.5$ everywhere

---

## Key Takeaways

### 1. GAN Framework
Two networks in adversarial competition:
- **Generator:** Creates fake images from noise
- **Discriminator:** Distinguishes real from fake

### 2. Discriminator Loss
$$L_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] - \mathbb{E}_{z}[\log(1 - D_\phi(G_\theta(z)))]$$
Wants to correctly classify both real and fake images.

### 3. Generator Loss (Better Version)
$$L_G = -\mathbb{E}_{z}[\log(D_\phi(G_\theta(z)))]$$
Maximizes discriminator's output on fake images (better gradients).

### 4. Why Not Use $\log(1 - D)$?
The naive loss $-\log(1-D_\phi(G_\theta(z)))$ suffers from vanishing gradients when generator is poor. The alternative $-\log(D_\phi(G_\theta(z)))$ provides stronger gradients for learning.

### 5. Optimal Discriminator
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{gen}}(x) + p_{\text{data}}(x)}$$
At equilibrium ($p_{\text{gen}} = p_{\text{data}}$): $D^* = 0.5$ everywhere.

### 6. Training Procedure
Alternate between:
1. Update discriminator (several steps)
2. Update generator (one step)
Repeat until convergence.

### 7. Comparison to VAEs
- **VAEs:** Encoder-decoder with probabilistic latent space, tractable training
- **GANs:** Pure generator, adversarial training, often sharper images

---

## Mathematical Notation Legend

### Model Components
- $G_\theta(z)$ = generator with parameters $\theta$
- $D_\phi(x)$ = discriminator with parameters $\phi$
- $z$ = latent noise vector, $z \sim \mathcal{N}(0, I)$
- $x$ = image (real or generated)

### Distributions
- $p_{\text{data}}(x)$ = true data distribution
- $p_{\text{gen}}(x)$ = generator's distribution
- $\mathcal{N}(0, I)$ = standard normal distribution

### Loss Functions
- $L_D(\phi, \theta)$ = discriminator loss
- $L_G(\phi, \theta)$ = generator loss
- $D^*(x)$ = optimal discriminator

### Operators
- $\mathbb{E}[\cdot]$ = expectation
- $\sim$ = "distributed as"
- $\nabla_\phi$ = gradient with respect to $\phi$
- $\sigma(\cdot)$ = sigmoid function
- $\log$ = natural logarithm

### Training
- $\alpha$ = learning rate
- $\phi$ = discriminator parameters
- $\theta$ = generator parameters