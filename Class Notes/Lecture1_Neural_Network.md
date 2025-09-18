# Gen AI Lecture Notes: Neural Networks

## Topics Covered
1. Neural Networks
2. Recurrent Neural Networks (RNNs)  
3. Review of Neural Networks
4. Linear Regression vs. Non-linear Regression

## Example Problem: Sentiment Analysis
**Task:** Determine whether a word has positive or negative connotation

### Key Challenges:
- No notion of similarity between different words
- Dimension of the vector can be very large

## Dataset Notation
Dataset: $\{(x^{(i)}, y^{(i)}), i = 1, 2, ..., n\}$
- $n$ = number of data points in dataset
- $x$ = word (input)
- $y$ = sentiment (output)

## From Linear to Non-linear Models

### Linear Combination
From $x_1$ to $x_n$:

$$z_1 = w_{11}x_1 + w_{12}x_2 + ... + w_{1d}x_d + b_1$$

**Problem:** This is still linear!

### Adding Non-linearity
**Solution:** Make $z_1 \rightarrow g(z_1)$ where $g$ is an activation function (non-linear function)

Create many z's and combine them:
$$\sigma(w_1z_1 + w_2z_2 + b)$$

### Matrix Formulation

Individual computations:
$$z_1 = w_{11}x_1 + w_{12}x_2 + ... + w_{1d}x_d + b_1$$
$$z_2 = w_{21}x_1 + w_{22}x_2 + ... + w_{2d}x_d + b_2$$

Vector notation:
$$\vec{z} = W^{[1]} \cdot \vec{x} + \vec{b}^{[1]}$$

Where:
- $\vec{z} = (z_1, ..., z_{10})$
- $\vec{b} = (b_1, ..., b_{10})$
- $W^{[1]}$ is the weight matrix

## Network Architecture

### Hidden Layer
$$\vec{z} = W^{[1]} \cdot \vec{x} + \vec{b}^{[1]}$$
$$\vec{a} = g(\vec{z})$$

### Output Layer
$$y = \sigma(W^{[2]} \cdot \vec{a} + b^{[2]})$$

## Complete Network
- **Input:** $\vec{x}$
- **Output:** $\hat{y} = \sigma(W^{[2]} \cdot \vec{a} + b^{[2]})$

## Training Objective

### Goal
Need to figure out $W^{[1]}$, $W^{[2]}$, $b^{[1]}$, $b^{[2]}$ such that the prediction of the sentiment of the word is as accurate as possible.

### Comparing Prediction to Reality
- $\hat{y}$ = prediction (y hat)
- $y$ = actual sentiment

### Loss Function
Minimize the error across all data points:

$$\text{Minimize} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2$$

**Goal:** Want a low error by adjusting the weights and biases

### Alternative Loss: Binary Cross-Entropy
For binary classification (sentiment analysis), we often use:

**Case 1: When $y = 1$ (positive sentiment)**
- If $\hat{y} = 0.99$ (close to correct)
- Loss: $-y \log \hat{y} = -1 \cdot \log(0.99) \approx 0.01$ (small loss)

**Case 2: When $y = 0$ (negative sentiment)**  
- If $\hat{y} = 0.99$ (far from correct)
- Loss: $-(1 - y) \log(1 - \hat{y}) = -1 \cdot \log(0.01) \approx 4.6$ (large loss)

**Combined Formula:**
$$\mathcal{L}(y, \hat{y}) = -[y \log \hat{y} + (1 - y) \log(1 - \hat{y})]$$

This loss function:
- Gives small loss when prediction is close to true label
- Gives large loss when prediction is far from true label
- Works better than squared error for classification problems

## Optimization: Finding the Weights

### Goal
Want to minimize the loss function to find the optimal weights: $W^{[1]}$, $W^{[2]}$, $b^{[1]}$, $b^{[2]}$

### Gradient Descent
Update rule:
$$x_{k+1} = x_k - \alpha \cdot f'(x_k)$$

Where:
- $\alpha$ = learning rate (step size)
- $f'(x_k)$ = derivative of the loss

### Computing Gradients via Chain Rule

For weight $W^{[2]}$:
$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{\partial \mathcal{L}(\hat{y})}{\partial W^{[2]}}$$

Since $\hat{y} = \sigma(W^{[2]} \cdot \vec{a} + b^{[2]})$, we apply chain rule:

$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W^{[2]}}$$

#### Step-by-step derivation:

1. **Derivative of loss w.r.t. $\hat{y}$:**
   $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

2. **Derivative of $\hat{y}$ w.r.t. $z^{[2]}$ (where $z^{[2]} = W^{[2]} \cdot \vec{a} + b^{[2]}$):**
   $$\frac{\partial \hat{y}}{\partial z^{[2]}} = \sigma(z^{[2]})(1 - \sigma(z^{[2]})) = \hat{y}(1-\hat{y})$$

3. **Combining steps 1 and 2:**
   $$\frac{\partial \mathcal{L}}{\partial z^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{[2]}} = \hat{y} - y$$

4. **Finally, derivative w.r.t. $W^{[2]}$:**
   $$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = (\hat{y} - y) \cdot \vec{a}^T$$

### Update Rules for All Parameters

$$W^{[2]} := W^{[2]} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W^{[2]}}$$

$$b^{[2]} := b^{[2]} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b^{[2]}}$$

$$W^{[1]} := W^{[1]} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial W^{[1]}}$$

$$b^{[1]} := b^{[1]} - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b^{[1]}}$$

The gradients for $W^{[1]}$ and $b^{[1]}$ require backpropagating through the activation function $g$.

---

## Key Takeaways

### 1. **Linear to Non-linear Transformation**
- Neural networks start with linear combinations but add activation functions to introduce non-linearity
- This allows the network to learn complex patterns that linear models cannot capture

### 2. **Network Components**
- **Input Layer:** Raw features $(x)$
- **Hidden Layer:** Learned representations $(z \rightarrow a$ through activation)
- **Output Layer:** Final prediction $(\hat{y})$

### 3. **Training Process**
- **Forward Pass:** Input → Hidden → Output
- **Loss Calculation:** Measure error between prediction and truth
- **Backward Pass:** Compute gradients via chain rule
- **Weight Update:** Adjust parameters using gradient descent

### 4. **Loss Functions**
- **Mean Squared Error:** Simple but less effective for classification
- **Binary Cross-Entropy:** Better for binary classification, penalizes confident wrong predictions heavily

### 5. **Optimization**
- Gradient descent iteratively adjusts weights to minimize loss
- Learning rate $(\alpha)$ controls step size
- Chain rule enables computing gradients for all parameters

---
*Note: The network learns to transform linear combinations into non-linear representations through activation functions, allowing it to capture complex patterns in sentiment analysis.*

---

## Mathematical Notation Legend

### Variables
- $x$ = input vector (features of a word)
- $y$ = true label/output (actual sentiment: 0 or 1)
- $\hat{y}$ = predicted output (predicted sentiment probability)
- $n$ = number of data points in dataset
- $d$ = dimension of input features
- $i$ = index for data points (superscript in parentheses: $x^{(i)}$)
- $k$ = iteration index in gradient descent

### Weight and Bias Parameters
- $W^{[1]}$ = weight matrix for layer 1 (hidden layer)
- $W^{[2]}$ = weight matrix for layer 2 (output layer)
- $b^{[1]}$ = bias vector for layer 1
- $b^{[2]}$ = bias scalar for layer 2
- $w_{ij}$ = individual weight connecting input $j$ to neuron $i$

### Intermediate Values
- $z$ = linear combination before activation $(z = Wx + b)$
- $a$ = activation output $(a = g(z))$
- $z^{[1]}, z^{[2]}$ = linear combinations in layer 1 and 2
- $\vec{a}$ = activation vector from hidden layer

### Functions
- $g(\cdot)$ = activation function (e.g., ReLU, tanh)
- $\sigma(\cdot)$ = sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- $\mathcal{L}(\cdot)$ = loss function
- $f'(\cdot)$ = derivative of function $f$

### Mathematical Operators
- $\cdot$ = dot product or matrix multiplication
- $\sum$ = summation
- $\partial$ = partial derivative
- $\nabla$ = gradient operator
- $:=$ = assignment operator (update rule)
- $\rightarrow$ = maps to / transformation
- $\in$ = element of
- $^T$ = transpose operation

### Greek Letters
- $\alpha$ (alpha) = learning rate / step size
- $\sigma$ (sigma) = sigmoid function
- $\mathcal{L}$ (script L) = loss function

### Indexing Notation
- **Subscript** (e.g., $x_1, w_{12}$) = component index
- **Superscript in brackets** (e.g., $W^{[1]}$) = layer number
- **Superscript in parentheses** (e.g., $x^{(i)}$) = data point index
- **Vector notation** (e.g., $\vec{x}, \vec{a}$) = indicates vector quantity