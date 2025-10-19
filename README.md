# CS 6180: Generative AI - Study Notes

My personal study repository for **CS 6180 - Generative AI** at Northeastern University (Fall 2025). These are my lecture notes as I work through understanding neural networks, NLP, and the journey to modern generative AI.

## Problem Sets  
**Please navigate to the answer using the links below. Every part inside problem sets has direct links!**
**[Problem Set 1](./Problem%20Set%201)**  
**[Problem Set 2](./Problem%20Set%202)**

## Lecture Notes

- **[Lecture 1: Neural Networks](./Class%20Notes/Lecture01_Neural_Network.md)**
  - Linear vs non-linear models, backpropagation, chain rule
  - Finally understanding how gradients flow backwards!
  
- **[Lecture 2: Representation of Words](./Class%20Notes/Lecture02_Representation_of_Words.md)** 
  - Why one-hot encoding isn't enough
  - Introduction to word embeddings
  
- **[Lecture 3: Word2Vec Deep Dive](./Class%20Notes/Lecture03_Word2Vec.md)**
  - Skip-gram model mathematics
  - Probability models and gradient computation
  - The clever negative sampling trick

- **[Lecture 4: GloVe & Intro to RNNs](./Class%20Notes/Lecture04_GloVe_Intro_to_RNNs.md)**
  - Global vectors approach
  - First steps into recurrent networks
  - Language modeling fundamentals
  
- **[Lecture 5: LSTM Part 1](./Class%20Notes/Lecture05_LSTM_Part1.md)**
  - How LSTMs solve the vanishing gradient problem
  - Understanding gates and cell states
  - Perplexity as an evaluation metric

- **[Lecture 6: LSTM Part 2](./Class%20Notes/Lecture06_LSTM_Part2.md)**
  - Eigenvalue analysis of gradient flow
  - Why orthogonal initialization helps
  - Gradient clipping and truncated BPTT (partial solutions)

- **[Lecture 7: LSTM Part 3](./Class%20Notes/Lecture07_LSTM_Part3.md)**
  - Deep dive into the three gates
  - Why tanh enables negation in cell states
  - Element-wise vs matrix multiplication for gradients

- **[Lecture 8: LSTM Backpropagation](./Class%20Notes/Lecture08_LSTM_Backpropagation.md)**
  - Computing gradients through LSTMs
  - Numerical evidence: (0.95)^8 vs (0.25)^8
  - Why LSTMs actually work for long sequences

- **[Lecture 9: Attention Mechanisms](./Class%20Notes/Lecture09_Attention_Mechanisms.md)**
  - Solving the bottleneck problem in seq2seq
  - Query-Key-Value framework
  - Cross-attention for neural machine translation

- **[Lecture 10: Self-Attention & Transformer Decoder](./Class%20Notes/Lecture10_Self_Attention_Transformer_Decoder.md)**
  - From cross-attention to self-attention
  - Position encoding (order matters!)
  - Masking for causality in language modeling

- **[Lecture 11: Complete Transformer Decoder](./Class%20Notes/Lecture11_Complete_Transformer_Decoder.md)**
  - Layer normalization and residual connections
  - Multi-head attention (different heads, different patterns)
  - Scaled dot-product attention and autoregressive models