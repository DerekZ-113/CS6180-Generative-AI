## Problem Set 2 Submission Quick Look-Up

**Please visit [Problem Set 2 (My GitHUb)](https://github.com/DerekZ-113/CS6180-Generative-AI/tree/main/Problem%20Set%202) here for the whole problem set, and working links below!**

### Part 2.1: Word Embeddings
- **[`pad_sents()` utils.py](Starter%20Code/utils.py#L20-L45)** - Padding sentences to equal length
- **[`ModelEmbeddings.__init__()` model_embeddings.py](Starter%20Code/model_embeddings.py#L51-L52)** - Initialize source/target embeddings

### Part 2.2: Bidirectional LSTM Encoder
- **[`__init__` nmt_model.py](Starter%20Code/nmt_model.py#L61-L111)** - Initializer
- **[`encode()` nmt_model.py](Starter%20Code/nmt_model.py#L185-L241)** - Encoder forward pass

### Part 2.3: Unidirectional LSTM Decoder
- **[`decode()` nmt_model.py](Starter%20Code/nmt_model.py#L274-L328)** - Decoder forward pass  
- **[`step()` nmt_model.py](Starter%20Code/nmt_model.py#L399-L441)** - Single decoder step with attention

### Part 2.4: Training Result
#### Test Performance
- **BLEU Score:** 20.34

#### Training Statistics
- **Training Time:** ~40 minutes
- **Final Iteration:** 19,200
- **Dev Perplexity:** 11.31
- **Early Stopping:** Hit after 5 patience trials

#### Architecture
- Embed size: 1024
- Hidden size: 768
- Dropout: 0.3
- Bidirectional LSTM Encoder
- Attention Mechanism

#### Actual BLEU Score Output 
```
Corpus BLEU: 20.34227161925886
```