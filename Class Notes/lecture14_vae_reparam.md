# CS 6180 Lecture 14: VAE Reparametrization Trick and Implementation
**Date:** October 27, 2025  
**Topic:** Reparametrization Trick, Training VAEs, and MNIST Implementation

---

## Overview

This lecture continues the VAE discussion from Lecture 13, focusing on a critical technical innovation: the **reparametrization trick**. This technique enables backpropagation through stochastic sampling operations, making VAE training practical. We also cover the complete VAE training objective and implementation details for generating MNIST digits.

---

## 1. Review: VAE Framework and Loss Function

### The VAE Architecture

**Core components:**

```
Image X → [Encoder] → Latent space z → [Decoder] → Reconstructed X'
             ↓                              ↑
         p(z|X)                         p(X|z)
         (intractable)                  (reconstruction)
```

### The ELBO Loss Function

From Lecture 13, we derived the Evidence Lower Bound (ELBO):

$$\text{ELBO} = \mathbb{E}_{q_\phi(z|X)}[\log p_\theta(X|z)] - \text{KL}(q_\phi(z|X) \| p(z))$$

**Two components:**

**1. Reconstruction Loss** (maximize this):
$$\mathbb{E}_{q_\phi(z|X)}[\log p_\theta(X|z)]$$
- Measures how well decoder reconstructs original image
- Want this to be **high** (good reconstruction)

**2. KL Divergence Term** (minimize this):
$$\text{KL}(q_\phi(z|X) \| p(z))$$
- Measures difference between posterior and prior
- Want this to be **low** (posterior stays close to prior)

### Training Objective

**Minimize negative ELBO:**

$$\mathcal{L} = -\mathbb{E}_{q_\phi(z|X)}[\log p_\theta(X|z)] + \text{KL}(q_\phi(z|X) \| p(z))$$

**This is a lower bound** to the log-likelihood function, since we're using $q_\phi(z|X)$ as an approximation of the intractable $p(z|X)$.

---

## 2. Model Specification Review

### Prior Distribution (Fixed)

$$p(z) = \mathcal{N}(0, I)$$

**Why this choice?**

1. **Simple to sample from:** Standard normal is straightforward
2. **Efficient KL computation:** KL between two Gaussians has closed form
3. **Maximum entropy:** Encourages learning diverse features
4. **Ensures spread:** Having variance in latent space helps model learn different features correctly

**Key insight:** The prior creates a "regularizing" effect, preventing the latent space from collapsing to point estimates.

### Encoder (Approximate Posterior)

$$q_\phi(z|X) = \mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$$

**Why Gaussian?**

- Not the exact posterior $p(z|X)$ (which is intractable)
- Gaussian provides tractable approximation
- Both $\mu_\phi(X)$ and $\sigma_\phi(X)$ are neural network outputs
- Diagonal covariance matrix simplifies computation

**Encoder architecture:**

```
X → [Neural Network] → μ_φ(X)  (mean)
                    → σ_φ(X)  (standard deviation)
```

### Decoder (Likelihood)

$$p_\theta(X|z) = \prod_i \text{Bernoulli}(x_i | f_\theta(z))$$

Where $f_\theta(z)$ is a neural network (decoder) outputting probability for each pixel.

---

## 3. The Reparametrization Trick: The Key Innovation

### The Problem with Naive Sampling

**Naive approach:** Sample directly from encoder distribution

$$z \sim q_\phi(z|X) = \mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$$

**Problem:** Cannot backpropagate through stochastic sampling!

```
X → [Encoder] → μ_φ(X), σ_φ(X) → [Sample z] → [Decoder] → X'
                                     ↑
                                 STOCHASTIC
                              (breaks gradients!)
```

**Why this breaks training:**
- Sampling operation is not differentiable
- Cannot compute $\frac{\partial \mathcal{L}}{\partial \mu_\phi}$ or $\frac{\partial \mathcal{L}}{\partial \sigma_\phi}$
- Gradients cannot flow back to encoder parameters $\phi$

### The Reparametrization Trick Solution

**Key insight:** Separate randomness from parameters!

Instead of sampling $z$ directly, rewrite as:

$$\boxed{z = \mu_\phi(X) + \sigma_\phi(X) \odot \varepsilon \quad \text{where } \varepsilon \sim \mathcal{N}(0, I)}$$

Where $\odot$ denotes element-wise multiplication.

**This is mathematically equivalent:**

**Expected value:**
$$\mathbb{E}[z] = \mathbb{E}[\mu_\phi(X) + \sigma_\phi(X) \odot \varepsilon] = \mu_\phi(X)$$

**Variance:**
$$\text{Var}(z) = \text{Var}(\sigma_\phi(X) \odot \varepsilon) = (\sigma_\phi(X))^2 \cdot \text{Var}(\varepsilon) = \sigma_\phi^2(X)$$

Therefore: $z \sim \mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$ ✓

### Why This Works

**New computational graph:**

```
X → [Encoder] → μ_φ(X), σ_φ(X) → [z = μ + σ⊙ε] → [Decoder] → X'
                                      ↑
                  ε ~ N(0,I)    DETERMINISTIC PATH!
                  (no gradients needed)
```

**Key advantages:**

1. **Deterministic path:** From $\mu_\phi(X)$ and $\sigma_\phi(X)$ to loss $\mathcal{L}$
2. **Randomness externalized:** Sample $\varepsilon$ independently (no gradients needed)
3. **Gradients flow:** Can compute $\frac{\partial \mathcal{L}}{\partial \mu_\phi}$ and $\frac{\partial \mathcal{L}}{\partial \sigma_\phi}$

### Mathematical Derivation

Starting from the reconstruction loss:

$$\mathbb{E}_{z \sim q_\phi(z|X)}[\log p_\theta(X|z)]$$

**Without reparametrization:** Cannot differentiate w.r.t. $\phi$

**With reparametrization:**

$$\mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}[\log p_\theta(X|z)] \quad \text{where } z = \mu_\phi(X) + \sigma_\phi(X) \odot \varepsilon$$

Now we can compute:

$$\frac{\partial}{\partial \mu_\phi} \mathbb{E}_{\varepsilon}[\log p_\theta(X|z)] = \mathbb{E}_{\varepsilon}\left[\frac{\partial \log p_\theta(X|z)}{\partial z} \cdot \frac{\partial z}{\partial \mu_\phi}\right]$$

Where:
$$\frac{\partial z}{\partial \mu_\phi} = I \quad \text{(identity matrix)}$$

Similarly:
$$\frac{\partial z}{\partial \sigma_\phi} = \text{diag}(\varepsilon)$$

**Gradient flow is now possible!** ✓

---

## 4. Visualizing the Reparametrization Trick

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
```

### Latent Space Visualization

```
        N(0,1)  ← Prior p(z)
         ↓
    ┌─────────────────┐
    │                 │
    │    ○  ○  ○      │  ← Learned encoder pushes
    │  ○        ○     │     towards certain regions
    │    ○  ○  ○      │     (nose region)
    │  ○  eyes  ○     │
    │                 │  ← Different samples give
    └─────────────────┘     different features
```

**The encoder learns:**
- Where to place samples in latent space (mean $\mu_\phi$)
- How much uncertainty/spread (variance $\sigma_\phi^2$)

**The KL term ensures:**
- Latent codes don't collapse to a single point
- Posterior stays close to prior $\mathcal{N}(0, I)$
- Model learns to use the latent space effectively

---

## 5. Complete Training Algorithm

### Forward Pass

**Input:** Image $X$

**Step 1: Encode**
$$\mu_\phi(X), \sigma_\phi(X) = \text{Encoder}(X)$$

**Step 2: Sample noise**
$$\varepsilon \sim \mathcal{N}(0, I)$$

**Step 3: Reparametrize**
$$z = \mu_\phi(X) + \sigma_\phi(X) \odot \varepsilon$$

**Step 4: Decode**
$$\hat{X} = \text{Decoder}(z)$$

**Step 5: Compute loss**

Reconstruction loss:
$$\mathcal{L}_{\text{recon}} = -\log p_\theta(X|z) \approx -\sum_i x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)$$

KL divergence (closed form for Gaussians):
$$\mathcal{L}_{\text{KL}} = \text{KL}(q_\phi(z|X) \| \mathcal{N}(0,I)) = \frac{1}{2}\sum_j \left(\mu_{\phi,j}^2 + \sigma_{\phi,j}^2 - \log \sigma_{\phi,j}^2 - 1\right)$$

Total loss:
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}}$$

### Backward Pass

**Step 6: Backpropagation**

Compute gradients:
$$\frac{\partial \mathcal{L}}{\partial \theta}, \frac{\partial \mathcal{L}}{\partial \phi}$$

Thanks to reparametrization, these are well-defined!

**Step 7: Update parameters**
$$\theta \leftarrow \theta - \alpha \frac{\partial \mathcal{L}}{\partial \theta}$$
$$\phi \leftarrow \phi - \alpha \frac{\partial \mathcal{L}}{\partial \phi}$$

---

## 6. KL Divergence: Closed Form for Gaussians

### General KL Between Gaussians

For two Gaussian distributions:
- $q = \mathcal{N}(\mu_q, \Sigma_q)$
- $p = \mathcal{N}(\mu_p, \Sigma_p)$

The KL divergence is:

$$\text{KL}(q \| p) = \frac{1}{2}\left[\log\frac{|\Sigma_p|}{|\Sigma_q|} - d + \text{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p - \mu_q)^T \Sigma_p^{-1}(\mu_p - \mu_q)\right]$$

### Simplified Form for VAE

In our case:
- $q_\phi(z|X) = \mathcal{N}(\mu_\phi(X), \text{diag}(\sigma_\phi^2(X)))$
- $p(z) = \mathcal{N}(0, I)$

With diagonal covariance, the KL simplifies to:

$$\boxed{\text{KL}(q_\phi(z|X) \| p(z)) = \frac{1}{2}\sum_{j=1}^{d} \left[\mu_{\phi,j}^2 + \sigma_{\phi,j}^2 - \log(\sigma_{\phi,j}^2) - 1\right]}$$

**Interpretation of each term:**

1. $\mu_{\phi,j}^2$: Penalizes mean far from 0
2. $\sigma_{\phi,j}^2$: Penalizes variance far from 1
3. $-\log(\sigma_{\phi,j}^2)$: Prevents variance collapse to 0
4. $-1$: Normalization constant

**Key insight:** This term can be computed in closed form (no sampling needed)!

---

## 7. Practical Implementation: MNIST VAE

### Dataset: MNIST Digits

**Task:** Generate new handwritten digits (0-9)

**Data:**
- Images: 28×28 pixels
- Pixel values: $x_i \in \{0, 1\}$ (binary: 0=white, 1=black)
- Training set: 60,000 images

### Model Architecture

**Encoder:**
```python
Input: 784-dimensional vector (28×28 flattened)
↓
Dense(512) + ReLU
↓
Dense(256) + ReLU
↓
μ_φ: Dense(latent_dim)     # Mean
σ_φ: Dense(latent_dim)     # Log variance (for numerical stability)
```

**Decoder:**
```python
Input: latent_dim-dimensional vector
↓
Dense(256) + ReLU
↓
Dense(512) + ReLU
↓
Dense(784) + Sigmoid  # Output probabilities for each pixel
```

**Latent dimension:** Typically 2-20 (hyperparameter to tune)

### Implementation Tasks

From the lecture notes, two key functions to implement:

#### Task 1: Sample Gaussian Function (utils.py)

**Implement the reparametrization trick:**

```python
def sample_gaussian(mu, log_var):
    """
    Reparametrization trick: sample from N(mu, var) using N(0,1)
    
    Args:
        mu: Mean of distribution (batch_size, latent_dim)
        log_var: Log variance (batch_size, latent_dim)
        
    Returns:
        z: Sampled latent codes (batch_size, latent_dim)
    """
    # TODO: Implement
    # Hint: Use torch.randn to sample epsilon
    # z = mu + sigma * epsilon
    # Remember: sigma = exp(0.5 * log_var)
```

**Key points:**
- Use `torch.randn_like(mu)` to sample $\varepsilon$
- Compute $\sigma = \exp(0.5 \cdot \log(\sigma^2))$ for numerical stability
- Return $z = \mu + \sigma \odot \varepsilon$

#### Task 2: Negative ELBO Loss (vae.py)

**Implement the complete loss function:**

```python
def negative_elbo_bound(x, x_recon, mu, log_var):
    """
    Compute negative ELBO = reconstruction_loss + KL_divergence
    
    Args:
        x: Original images (batch_size, 784)
        x_recon: Reconstructed images (batch_size, 784)
        mu: Encoder mean (batch_size, latent_dim)
        log_var: Encoder log variance (batch_size, latent_dim)
        
    Returns:
        loss: Negative ELBO (scalar)
    """
    # TODO: Implement
    # 1. Reconstruction loss: Binary cross-entropy
    # 2. KL divergence: Closed form for Gaussian
```

**Reconstruction loss** (binary cross-entropy):
$$\mathcal{L}_{\text{recon}} = -\sum_i [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)]$$

**KL divergence** (closed form):
$$\mathcal{L}_{\text{KL}} = \frac{1}{2}\sum_j [\mu_j^2 + e^{\log \sigma_j^2} - \log \sigma_j^2 - 1]$$

---

## 8. Training Dynamics and Insights

### What the Model Learns

**Encoder learns:**
- **Mean $\mu_\phi(X)$:** Where in latent space to place this image
  - Similar digits cluster together
  - Smooth interpolation between digit types
  
- **Variance $\sigma_\phi^2(X)$:** How uncertain the encoding is
  - Clear, well-formed digits → low variance
  - Ambiguous or distorted digits → high variance

**Decoder learns:**
- How to map latent codes to pixel probabilities
- Gradual transitions between digit styles

### Balancing the Two Loss Terms

**Reconstruction loss** pushes the model to:
- Encode each image precisely
- Minimize variance (for accurate reconstruction)
- Risk: Overfitting, no diversity

**KL divergence** pushes the model to:
- Keep latent codes close to $\mathcal{N}(0, I)$
- Maintain variance in latent space
- Risk: Blurry reconstructions

**Trade-off:** These two terms compete!

**Solution:** Some implementations use a $\beta$ parameter:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

- $\beta < 1$: Prioritize reconstruction (sharper images)
- $\beta > 1$: Prioritize regularization (more diverse sampling)
- $\beta = 1$: Standard VAE (theoretically justified)

### Generating New Digits

**Once trained:**

1. Sample $z \sim \mathcal{N}(0, I)$ (no encoder needed!)
2. Pass through decoder: $\hat{X} = \text{Decoder}(z)$
3. Sample pixels: $x_i \sim \text{Bernoulli}(\hat{x}_i)$

**Latent space exploration:**
- Sample different $z$ values → different digits
- Interpolate between two $z$ values → smooth transitions
- Modify specific dimensions → control features (thickness, slant, etc.)

---

## 9. Limitations of Reparametrization Trick

### When It Works

✓ **Normal (Gaussian) distributions**
- Most common case
- Clean mathematical solution
- Efficient to implement

✓ **Other location-scale families**
- Distributions of form: $x = \mu + \sigma \cdot \varepsilon$
- Examples: Logistic, Laplace, Uniform

### When It Doesn't Work

✗ **Discrete distributions**
- Bernoulli, Categorical, Multinomial
- No continuous path for gradients
- Require alternative techniques (REINFORCE, Gumbel-Softmax)

✗ **General distributions**
- Not all distributions have location-scale form
- May need specialized reparametrizations
- Active research area

**Alternative gradient estimators:**
- **REINFORCE (score function estimator):** High variance
- **Gumbel-Softmax trick:** For categorical distributions
- **Straight-through estimators:** Biased but practical

---

## Key Takeaways

### 1. The Reparametrization Trick
**Problem:** Cannot backpropagate through stochastic sampling
**Solution:** $z = \mu_\phi(X) + \sigma_\phi(X) \odot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0,I)$
**Result:** Deterministic path for gradients, randomness externalized

### 2. Why This is Critical
- Enables training of VAEs with standard backpropagation
- No need for high-variance gradient estimators
- Makes VAEs practical and efficient

### 3. Complete VAE Training
**Loss:** $\mathcal{L} = -\mathbb{E}[\log p_\theta(X|z)] + \text{KL}(q_\phi(z|X) \| p(z))$
**Gradients:** Flow through both $\mu_\phi$ and $\sigma_\phi$
**Training:** Standard SGD/Adam with backpropagation

### 4. KL Divergence Closed Form
For Gaussian prior and posterior:
$$\text{KL} = \frac{1}{2}\sum_j [\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1]$$
No sampling needed for this term!

### 5. Trade-offs in VAE Training
- Reconstruction loss vs. KL regularization
- Sharpness vs. diversity
- Controlled by $\beta$ hyperparameter

### 6. Practical Implementation
- Use `log_var` instead of `var` for numerical stability
- Sample $\varepsilon \sim \mathcal{N}(0,I)$ using `torch.randn`
- Binary cross-entropy for pixel-level reconstruction

### 7. Generation Process
1. Sample $z \sim \mathcal{N}(0, I)$
2. Decode: $p_\theta(X|z)$
3. Sample pixels from Bernoulli distributions

---

## Mathematical Notation Legend

### Model Components
- $X$ = observed image (data)
- $z$ = latent variable
- $\varepsilon$ = noise sample from $\mathcal{N}(0, I)$
- $\theta$ = decoder parameters
- $\phi$ = encoder parameters

### Distributions
- $p(z) = \mathcal{N}(0, I)$ = prior (standard normal)
- $p_\theta(X|z)$ = decoder/likelihood
- $q_\phi(z|X) = \mathcal{N}(\mu_\phi(X), \sigma_\phi^2(X))$ = encoder (approximate posterior)
- $p(z|X)$ = true posterior (intractable)

### Network Outputs
- $\mu_\phi(X)$ = encoder mean (neural network)
- $\sigma_\phi(X)$ = encoder standard deviation (neural network)
- $\log \sigma_\phi^2(X)$ = log variance (for numerical stability)
- $f_\theta(z)$ = decoder output (pixel probabilities)

### Loss Terms
- $\mathcal{L}_{\text{recon}}$ = reconstruction loss
- $\mathcal{L}_{\text{KL}}$ = KL divergence term
- $\mathcal{L}$ = total loss (negative ELBO)

### Operators
- $\odot$ = element-wise (Hadamard) multiplication
- $\mathbb{E}[\cdot]$ = expectation
- $\sim$ = "distributed as"
- $\text{diag}(\cdot)$ = diagonal matrix
- $|\Sigma|$ = determinant of $\Sigma$
- $\text{tr}(\cdot)$ = trace

### Constants
- $d$ = latent dimension
- $I$ = identity matrix
- $\alpha$ = learning rate
- $\beta$ = KL weight (optional, default=1)