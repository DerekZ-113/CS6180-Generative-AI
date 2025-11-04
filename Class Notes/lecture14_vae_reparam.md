# CS 6180 Lecture 14: The Reparametrization Trick
**Date:** October 27, 2025  
**Topic:** Making VAE Training Practical Through Gradient Flow

---

## Overview

This lecture introduces the **reparametrization trick**, the critical technical innovation that makes VAE training practical. Without it, we cannot backpropagate through stochastic sampling operations.

**Prerequisites:** Lecture 13 (ELBO, VAE architecture, encoder/decoder specifications)

---

## 1. The Gradient Problem

### Naive Sampling Approach

**From Lecture 13, we have:**
- Encoder outputs: $\mu_\phi(X), \sigma_\phi(X)$
- Need to sample: $z \sim q_\phi(z|X) = \mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$
- Then decode to get reconstruction

**Problem:** Direct sampling breaks backpropagation!

```
X → [Encoder] → μ_φ(X), σ_φ(X) → [Sample z] → [Decoder] → X'
                                     ↑
                                 STOCHASTIC
                              (no gradients!)
```

**Why this fails:**
- Sampling operation is **not differentiable**
- Cannot compute $\frac{\partial \mathcal{L}}{\partial \mu_\phi}$ or $\frac{\partial \mathcal{L}}{\partial \sigma_\phi}$
- Gradients cannot flow back to encoder parameters $\phi$
- **Training is impossible with standard backpropagation**

**The loss function still requires this expectation:**

$\mathcal{L} = -\mathbb{E}_{z \sim q_\phi(z|X)}[\log p_\theta(X|z)] + \text{KL}(q_\phi(z|X) \| p(z))$

We need a way to compute gradients w.r.t. $\phi$!

---

## 2. The Reparametrization Trick

### The Key Insight

**Instead of sampling $z$ directly, rewrite as:**

$\boxed{z = \mu_\phi(X) + \sigma_\phi(X) \odot \varepsilon \quad \text{where } \varepsilon \sim \mathcal{N}(0, I)}$

Where $\odot$ denotes element-wise multiplication.

### Proof of Equivalence

**This transformation is mathematically equivalent to sampling from $\mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$:**

**Expected value:**
$\mathbb{E}[z] = \mathbb{E}[\mu_\phi(X) + \sigma_\phi(X) \odot \varepsilon] = \mu_\phi(X) + \sigma_\phi(X) \odot \mathbb{E}[\varepsilon] = \mu_\phi(X)$

**Variance:**
$\text{Var}(z) = \text{Var}(\sigma_\phi(X) \odot \varepsilon) = (\sigma_\phi(X))^2 \cdot \text{Var}(\varepsilon) = \sigma_\phi^2(X) \cdot 1 = \sigma_\phi^2(X)$

Therefore: $z \sim \mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$ ✓

### Why Gradients Now Flow

**New computational graph:**

```
X → [Encoder] → μ_φ(X), σ_φ(X) → [z = μ + σ⊙ε] → [Decoder] → X'
                                      ↑
                  ε ~ N(0,I)    DETERMINISTIC!
                  (sample once, no gradients needed)
```

**Critical observation:** The path from $\mu_\phi(X)$ and $\sigma_\phi(X)$ to $z$ is now **deterministic**!

**Gradient computation:**

$\frac{\partial z}{\partial \mu_\phi} = I \quad \text{(identity matrix)}$

$\frac{\partial z}{\partial \sigma_\phi} = \text{diag}(\varepsilon) \quad \text{(diagonal matrix with } \varepsilon \text{ on diagonal)}$

Both gradients are well-defined! We can now backpropagate through the entire network.

**The randomness is externalized:** $\varepsilon$ is sampled independently and doesn't need gradients.

---

## 3. Visualization: Before and After

### Before Reparametrization

```
      Encoder
         ↓
    ┌─────────┐
    │ μ_φ(X)  │
    │ σ_φ(X)  │
    └─────────┘
         ↓
    [Sample z]  ← STOCHASTIC (blocks gradients)
         ↓
         z
         ↓
      Decoder
         ↓
      Loss ℒ
         
    ∂ℒ/∂μ_φ = ❌ Cannot compute
    ∂ℒ/∂σ_φ = ❌ Cannot compute
```

### After Reparametrization

```
      Encoder
         ↓
    ┌─────────┐
    │ μ_φ(X)  │ ───┐
    │ σ_φ(X)  │ ───┼─→ z = μ_φ(X) + σ_φ(X)⊙ε
    └─────────┘    │         ↑
                   │         │
         ε ~ N(0,I)│    (external randomness)
      (no gradients needed)
                   │
              DETERMINISTIC PATH
                   ↓
                   z
                   ↓
                Decoder
                   ↓
                Loss ℒ
                   
    ∂ℒ/∂μ_φ = ✓ Computable!
    ∂ℒ/∂σ_φ = ✓ Computable!
```

**Key difference:** There's now a deterministic path from encoder outputs to loss, allowing gradient flow.

---

## 4. Closed Form KL Divergence

### General Formula for Gaussians

The second term in our loss is the KL divergence between two Gaussians:

$\text{KL}(q_\phi(z|X) \| p(z)) = \text{KL}(\mathcal{N}(\mu_\phi, \sigma_\phi^2) \| \mathcal{N}(0, I))$

**For diagonal covariance Gaussians, this has a closed form:**

$\boxed{\text{KL}(q_\phi(z|X) \| p(z)) = \frac{1}{2}\sum_{j=1}^{d} \left[\mu_{\phi,j}^2 + \sigma_{\phi,j}^2 - \log(\sigma_{\phi,j}^2) - 1\right]}$

**Critical advantage:** No sampling required for this term! It's computed deterministically.

### Interpretation of Each Term

1. **$\mu_{\phi,j}^2$:** Penalizes mean far from 0 (prior mean)
2. **$\sigma_{\phi,j}^2$:** Penalizes variance far from 1 (prior variance)
3. **$-\log(\sigma_{\phi,j}^2)$:** Prevents variance collapse to 0
4. **$-1$:** Normalization constant

**Key advantage:** This can be computed **exactly** with no sampling required!

---

## 5. MNIST Implementation Tasks

### Task 1: Reparametrization Trick (utils.py)

**Implement `sample_gaussian` function:**

```python
def sample_gaussian(mu, log_var):
    """
    Sample from N(mu, var) using reparametrization trick
    
    Args:
        mu: Mean (batch_size, latent_dim)
        log_var: Log variance (batch_size, latent_dim)
        
    Returns:
        z: Sampled latent codes (batch_size, latent_dim)
    """
    # Step 1: Sample ε ~ N(0, I)
    epsilon = torch.randn_like(mu)
    
    # Step 2: Compute σ = exp(0.5 * log_var)
    sigma = torch.exp(0.5 * log_var)
    
    # Step 3: z = μ + σ ⊙ ε
    z = mu + sigma * epsilon
    
    return z
```

**Key implementation details:**
- Use `torch.randn_like(mu)` to sample $\varepsilon$
- Work with `log_var` for numerical stability
- Compute $\sigma = \exp(0.5 \cdot \log(\sigma^2))$

### Task 2: Negative ELBO (vae.py)

**Implement the complete loss function:**

```python
def negative_elbo_bound(x, x_recon, mu, log_var):
    """
    Compute -ELBO = reconstruction_loss + KL_divergence
    
    Args:
        x: Original images (batch_size, 784)
        x_recon: Reconstructed images (batch_size, 784)
        mu: Encoder mean (batch_size, latent_dim)
        log_var: Encoder log variance (batch_size, latent_dim)
        
    Returns:
        loss: Negative ELBO (scalar)
    """
    # Reconstruction loss: Binary cross-entropy
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL divergence: Closed form
    # KL = 0.5 * sum(μ² + σ² - log(σ²) - 1)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss
```

**Reconstruction loss (binary cross-entropy):**
$\mathcal{L}_{\text{recon}} = -\sum_i [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$

**KL divergence (closed form):**
$\mathcal{L}_{\text{KL}} = \frac{1}{2}\sum_j [\mu_j^2 + e^{\log \sigma_j^2} - \log \sigma_j^2 - 1]$

### Why Log Variance?

**Numerical stability:** Working with $\log(\sigma^2)$ instead of $\sigma^2$ prevents:
- Numerical overflow when $\sigma^2$ is large
- Numerical underflow when $\sigma^2$ is small
- Ensures $\sigma^2 = e^{\log(\sigma^2)} > 0$ always

---

## 6. Limitations and Scope

### When Reparametrization Works

✓ **Normal (Gaussian) distributions**
- Clean mathematical solution
- Most common case for VAEs

✓ **Location-scale families**
- Distributions of form: $x = \mu + \sigma \cdot \varepsilon$
- Examples: Logistic, Laplace, Uniform

### When It Doesn't Work

✗ **Discrete distributions**
- Bernoulli, Categorical, Multinomial
- No continuous path for gradients
- **Alternative:** REINFORCE, Gumbel-Softmax trick

✗ **General distributions**
- Not all have location-scale form
- Requires specialized techniques

**Note:** The reparametrization trick applies to **normal distributions**, not all distributions.

---

## Key Takeaways

### 1. The Core Problem
Naive sampling $z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$ blocks gradient flow because sampling is not differentiable.

### 2. The Solution
**Reparametrization trick:** $z = \mu_\phi(X) + \sigma_\phi(X) \odot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, I)$

**Why it works:**
- Mathematically equivalent to sampling from $\mathcal{N}(\mu_\phi, \sigma_\phi^2)$
- Creates deterministic path from encoder outputs to loss
- Externalizes randomness (no gradients needed for $\varepsilon$)

### 3. Gradient Flow
$\frac{\partial z}{\partial \mu_\phi} = I, \quad \frac{\partial z}{\partial \sigma_\phi} = \text{diag}(\varepsilon)$

Both gradients are well-defined, enabling standard backpropagation.

### 4. KL Divergence Closed Form
For Gaussian prior and posterior:
$\text{KL} = \frac{1}{2}\sum_j [\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1]$

**No sampling needed** for this term!

### 5. Implementation Practices
- Use `log_var` instead of `var` for numerical stability
- Sample $\varepsilon \sim \mathcal{N}(0, I)$ using `torch.randn`
- Binary cross-entropy for reconstruction (MNIST pixels)
- Sum both losses for total negative ELBO

### 6. Scope Limitation
Reparametrization trick works for **Gaussian (and location-scale) distributions only**. Other distributions require alternative gradient estimators.

---

## Mathematical Notation Legend

### Core Variables
- $z$ = latent variable
- $\varepsilon$ = noise sample from $\mathcal{N}(0, I)$
- $\mu_\phi(X)$ = encoder mean
- $\sigma_\phi(X)$ = encoder standard deviation
- $\log \sigma_\phi^2(X)$ = log variance

### Gradients
- $\frac{\partial z}{\partial \mu_\phi}$ = gradient of $z$ w.r.t. mean
- $\frac{\partial z}{\partial \sigma_\phi}$ = gradient of $z$ w.r.t. std deviation
- $\frac{\partial \mathcal{L}}{\partial \phi}$ = gradient of loss w.r.t. encoder parameters

### Loss Components
- $\mathcal{L}_{\text{recon}}$ = reconstruction loss (binary cross-entropy)
- $\mathcal{L}_{\text{KL}}$ = KL divergence term (closed form)
- $\mathcal{L}$ = total loss (negative ELBO)

### Operators
- $\odot$ = element-wise (Hadamard) multiplication
- $\text{diag}(\cdot)$ = diagonal matrix
- $I$ = identity matrix