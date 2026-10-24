Homework 1 Answers separate
Homework 4 Answers separate
Lab 8 Answers separate

- [Homework 2-3 Answers](#homework-2-3-answers)
  - [Homework 2-3](#homework-2-3)
- [Homework 6/7 RNNs including LSTM and BRNN](#homework-67-rnns-including-lstm-and-brnn)
  - [Part 1](#part-1)
  - [Part 2 T/F](#part-2-tf)
  - [Part 3 Understanding](#part-3-understanding)
- [Lab 9 Answers Transformers](#lab-9-answers-transformers)
  - [Part 1](#part-1-1)
  - [Part 2](#part-2)
  - [Part 3](#part-3)
  - [Part 4](#part-4)
- [Lab 10 Answers Bert](#lab-10-answers-bert)
  - [Part 1](#part-1-2)
  - [Part 2](#part-2-1)
  - [Part 3](#part-3-1)
- [Lab 11 Answers LLM](#lab-11-answers-llm)
  - [Part 1 T/F](#part-1-tf)
  - [Part 2 MCQ](#part-2-mcq)
  - [Part 3 SAQ](#part-3-saq)
  - [Part 4 LAQ](#part-4-laq)


# Homework 2-3 Answers
## Homework 2-3

1. For the task of recognising that an object is a dog or not, we can use the Linear Regression model. If the model is regarded as a neural network, which of the following statements are correct?

- (a) There is a single unit (neuron) in the output layer. **TRUE**.
- (b) There are exactly two units (neurons) in the output layer. **FALSE**.
- (c) The output of the model is a real number between 0 and 1, which can be interpreted as the probability that the object in input image is a dog. **TRUE** (post sigmoid activation function)
- (d) After choosing a threshold, you can convert the model's output into a category of 0 or 1. **TRUE**.

2. Which of these are terms used to refer to components of an artificial neural network?

- (a) Axon. **FALSE**. Part of a neuron.
- (b) Activation function. **TRUE**.
- (c) Layers. **TRUE**.
- (d) Neurons. **TRUE**. Represents the affine transformation followed by activation function.

3. True/False? Neural networks take inspiration from, but not very accurately mimic, how neurons in a biological brain learn.

- (a) True
- (b) False

**TRUE**

4. For the following code:
```python
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 64), nn.ReLU(),
    nn.Linear(64, 10), nn.Softmax()
)
```

This code will define a neural network with how many layers?

- (a) 6
- (b) 5
- (c) 4
- (d) 3

~~4 Layers. 1 Input, 2 hidden, 1 output.~~

5. How do you define a linear regression model for a learning problem from a dataset with three features in PyTorch?

```
nn.Linear(3,1)
```

6. Suppose our Softmax regression model has 100 nodes (neurons) in the input layer and five nodes in the output layer. If bias terms are not regarded as weights, what is the size of the input vector **x**? What is the size of the matrix **w** for weights? What is the size of the bias vector? For the second output node, is its weight vector the 2nd row or the 2nd column of the matrix **w**?

Input layer: 100
Output layer: 5

Design matrix X is (num_examples, 100)
x is (100)
b is (5)
w is (5, 100)
The second output node is the 2nd row of the matrix w.

7. Which of these is the best way to determine whether your model has underfitting for the training set?

- (a) See if the training error is high (above 20% or so).
- (b) See if the validation error is high compared to the baseline level of performance.
- (c) Compare the training error to the validation error.
- (d) Compare the training error to the baseline level of performance.

**D. Compare training error to baseline level of performanc.**

8. Compare the activation functions ReLU and Sigmoid (or Logistic function), briefly explain the advantages of ReLU over Sigmoid.

ReLU(z) = max(0, z)
Sigmoid(z) = \frac{1}{1+e^{-z}}

Advantages of ReLU:
- No vanishing gradient for positive inputs
- Computation efficiency
- Faster empirical convergence

Advantages of sigmoid:

As the gradient of Sigmoid is not 0 for $z<0$, it does not suffer from the dying ReLU problem. 

# Homework 6/7 RNNs including LSTM and BRNN

## Part 1

1. Why do standard Multi-Layer Perceptrons (MLPs) struggle with tasks like binary 
addition on sequences of varying lengths? 
- A) They require a predefined, fixed number of input units. 
- B) The weights applied at the beginning of the sequence are different from those at the end, preventing the model from generalizing a rule (like carrying a '1'). 
- C) They cannot use activation functions on binary numbers. 
- **D) Both A and B.**

2. During the training of an RNN Language Model, what is "Teacher Forcing"? 
- A) Punishing the model with a higher learning rate when it makes a mistake. 
- B) Feeding the model's own predicted word back into it as the next input. 
- **C) Feeding the actual correct next word from the training corpus as the input for the next time step, regardless of what the model predicted.**
- D) Forcing the hidden state to reset to zero after every sentence

3. In the PyTorch data construction example for Autoregressive Models, what does the tau 
parameter represent? 
- A) The learning rate of the model. 
- **B) The window size, meaning the number of previous time steps used as features to predict the next time step. **
- C) The total number of epochs the model will train for. 
- D) The threshold used for gradient clipping


4. In a Long Short-Term Memory (LSTM) network, which component is primarily 
responsible for deciding what new information should be stored in the cell state? 
A) The Forget Gate 
**B) The Input Gate combined with the Candidate Memory** 
C) The Output Gate 
D) The Softmax layer

5. How does a Bidirectional Recurrent Neural Network (BRNN) differ fundamentally from a standard, unidirectional RNN?
- **A) It processes the sequence in both forward and backward directions to utilize past and future context simultaneously.**
- B) It restricts the network to only predict the previous sequence step instead of the next one.
- C) It limits the gradient backpropagation to only move backward through the hidden layers, not through time.
- D) It requires the input data to be duplicated and fed into the network twice at the same time step

## Part 2 T/F
6. True or False: In an unrolled Recurrent Neural Network, the weight matrices (e.g., the hidden-to-hidden weights) are updated and change their values at every single time step within a sequence.

F.

7. True or False: The "vanishing gradient" problem makes it difficult for an RNN to learn long-term dependencies because the gradients become too small to meaningfully update the weights for earlier time steps

T.

8. True or False: To solve the exploding gradient problem (which can cause NaN errors in your network), we use a technique called Gradient Clipping, which shrinks the gradient vector if its norm exceeds a specific threshold.

T.

## Part 3 Understanding
9. The Netflix Example: Briefly explain the "Netflix rating system" example (Koren, 2009) shown in the lecture. Why does this highlight the importance of treating data as a sequence rather than a static dataset?

IDK what the netflix rating system is but I assume that dataset drift over time 

10. LSTM Mathematics: Look at the core equation for updating the cell state (memory) in 
an LSTM: $C_t = F_t \odot C_{t-1} + I_t \odot \tilde{C}_t$. Briefly explain what the two halves of this addition are doing in plain English

We have two gates controlling cell state in a LSTM: forget and input gate. Both are linear layers based on prior hidden state and the current timestep input passed through a sigmoid function (therefore outputting between 0 and 1). In the given cell update equation, $F_t$ controls how much of the prior cell state is preserved into the current cell state, while $I_t$ controls how much candidate cell state is merged with the prior cell state.

11. PyTorch Sequence Slicing: Look at the following Python logic based on the slides on 
RNNs for creating the data iterator:

    `features = [self.x[i: self.T-self.tau+i] for i in range(self.tau)]`

    `self.labels = self.x[self.tau:].reshape((-1, 1))`

    If our raw sequence data x is [10, 20, 30, 40, 50, 60] and tau is 3, what will be the features 
array and label for the very first training example?


```python
features = [self.x[i: self.T-self.tau+i] for i in range(self.tau)]

## given x = [10, 20, 30, 40, 50, 60], tau = 3
features = [
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50]
]

## given x = [10, 20, 30, 40, 50, 60], tau = 3
self.labels = self.x[self.tau:].reshape((-1, 1))

## self.x[self.tau:] is  [40, 50, 60]
self.labels = [[40], [50], [60]]
```

# Lab 9 Answers Transformers

## Part 1

1. In a self-attention layer, the input is copied to act as the key, the value, and the
query.

T.

2. Layer normalisation in a Transformer block calculates the mean and variances along the first batch dimension, exactly like standard batch normalisation

F.

3. In an attention layer, the output is generated by computing a weighted sum of the 
values.

T

## Part 2

1. What is the primary motivation for introducing the Attention mechanism in 
sequence-to-sequence models?

- A) To compress all source tokens into a single hidden state vector.
- B) To replace all dense layers in the encoder.
- **C) To allow each generated token to relate to different source tokens rather than 
relying on a single hidden state.**
- D) To reduce the number of parameters in the network

2. Which of the following components is used to inject sequential information into a 
Transformer model?

- A) Recurrent layers
- **B) Positional Encoding**
- C) Position-wise Feed-Forward Networks
- D) Multi-head Attention

3. How does a Position-wise Feed-Forward Network (FFN) in a Transformer reshape 
the input data?

- A) From (seq_len, fea_size) to (batch, seq_len)
- **B) From (batch, seq_len, fea_size) to (batch * seq_len, fea_size)**
- C) From (batch, fea_size) to (batch * seq_len, fea_size)
- D) From (batch * seq_len, fea_size) to (batch, seq_len, fea_size)

## Part 3

1. Calculating Attention: Briefly explain the mathematical difference between how 
"Dot Product Attention" and "Multilayer Perceptron (MLP) Attention" calculate their 
attention scores.

Dot Product Attention will use a dot product operation between the keys and queries to compute the attention of each token whereas Multilayer Perceptron (MLP) Attention contains a small MLP to calculate the attention score.

Dot product:
$$
a(\mathbf{q}, \mathbf{k}_i) = \frac{\mathbf{q}^\top \mathbf{k}_i}{\sqrt{d}}
$$

MLP:
$$
a(\mathbf q, \mathbf k) = \mathbf v^\top \textrm{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},
$$

Dot product is easier to compute but MLP works with keys and queries that are different lengths.

2. Why does a Transformer model require Positional Encoding, and mathematically, what types of functions are used to create the positional matrix $P$?

A transformer model has no inhert knowledge of the order of tokens. In each transformer layer, the attention head allows the model to compute the attention of every other token but has no mechanism of identifying order. Attention head masking does not enforce order of past tokens, it just prevents the model from cheating by looking at future tokens.

The functions used is:

$$ p_{i,k} =
\begin{cases}
\sin\left(\dfrac{i}{10000^{k/d}}\right), & k \text{ is even} \\
\cos\left(\dfrac{i}{10000^{(k-1)/d}}\right), & k \text{ is odd}
\end{cases}
$$

## Part 4

1. Architectural Differences: Compare and contrast a traditional Seq2Seq model  equipped with attention versus the pure Transformer architecture. Identify and describe the three major places where the Transformer differs from a Seq2Seq model with attention.


From the ans:
- The Transformer relies entirely on attention and positional encoding, 
dropping recurrent layers entirely.
- It utilises stacked "Transformer blocks" that contain Multi-head Attention, Add 
& Norm (Layer Normalisation), and Position-wise FFNs.
- The state from the encoder is passed to every block in the decoder, rather 
than just acting as a single hidden state passed between recurrent layers.

My (incomplete) ans:

A traditional Seq2Seq model equipped with attention uses an attention head to condition each layer of the decoder.

The pure Transformer architecture uses a self-attention head in each encoder layer (without masking) and a self-attention head (with masking) followed by a cross-attention head (without masking) in the decoder layer.

The Transformer architecture is able to process the inputs in parallel, while the Seq2Seq is required to loop the decoder (during training).

# Lab 10 Answers Bert

## Part 1

1. The "Bitter Lesson" in AI suggests that complex, hand-engineered approaches will 
always outperform simple algorithms in the long run

    False

2. When evaluating a language model, a lower perplexity score indicates better 
performance.

    Perplexity is the average cross-entropy across each timestep in a sequence. A lower score indicates better performance.

    True

3. BERT is fundamentally an autoregressive model capable of naturally generating text 
left-to-right.

    BERT stands for bidirectional encoder representations from transformers.

    BERT is trained to fill in masked tokens or classify two sentences as whether they are contiguous or unrelated.

    It is not fundamentally an autoregressive model.

    False

4. Modern pre-trained language models rely heavily on subword tokenisation instead of 
standard word-level embeddings like GloVe

    True

## Part 2

5. Which of the following best describes how BERT achieves bidirectionality?

    a. It concatenates two independent unidirectional LSTMs (forward and backward).\
    b. It uses a standard autoregressive objective but reads the text from right-to-left.\
    **c. It utilises the Transformer architecture and a Masked Language Modeling (MLM) \
    objective.** \
    d. It uses Byte Pair Encoding to reverse the order of characters

    c.

6. What is the primary purpose of the [CLS] token in BERT?

    a. To act as a boundary separating two different sentences.\
    b. To predict the masked words in the sequence.\
    **c. To represent the aggregated sequence-level information for classification tasks.**\
    d. To indicate the end of a generated sequence.

    c.

7. How does BERT handle extractive Question Answering tasks (like SQuAD)?

    a. It generates the answer sequence left-to-right.\
    **b. It predicts the start and end indices of the answer span within the provided passage.**\
    c. It classifies the entire passage as either 'True' or 'False'.\
    d. It retrieves the answer from an external knowledge base

    b.

8. Why did modern NLP models move away from strictly character-level models?

    a. Character-level models cannot handle out-of-vocabulary terms.\
    **b. They resulted in sequences that were too long, making computation extremely 
    expensive.**\
    c. They required massive softmax matrices that slowed down training.\
    d. They overfit easily on small datasets

    b.

## Part 3

9. Explain the mechanism of Byte Pair Encoding (BPE). How does it construct its
vocabulary?

    BPE splits the input sequence into its individual characters, then finds the most frequent pair and concatenates it into a new symbol.

    Better answer: ```BPE starts by treating every individual character/byte as its own symbol. It counts the co-
occurrences of adjacent symbols in the training corpus and iteratively merges
the most frequent pair into a single new subword token. It repeats this process for a specified
number of merges (e.g., 8000), effectively creating a vocabulary of frequent whole words and
common word parts```

10. Contrast the bidirectionality of ELMO with the bidirectionality of BERT. Why is
BERT's approach considered "deeply bidirectional”?

    ELMO uses two LSTMs that independently read the input sequence then whose output is concatenated.

    BERT uses a transformer architecture where every token can attend to every other token in the sequence simultaneously regardless of direction.

# Lab 11 Answers LLM

## Part 1 T/F

1. According to the Chinchilla (2022) scaling laws, researchers should scale up model parameters much faster than they scale up the dataset size.

F.

2. Rejection Fine-Tuning (RFT) involves sampling multiple responses, scoring them, keeping only the high-reward answers, and running an offline Supervised Fine-Tuning (SFT) job on them

T.

3. During the Reinforcement Learning from Human Feedback (RLHF) process, a Reward Model is trained using human comparison data, where a labeler ranks model outputs from best to worst

T.

4. One of the main reasons Rejection Fine-Tuning (RFT) is considered perfectly sufficient for all post-training is that it excels at learning from failed examples.

F.

## Part 2 MCQ

1. What was a key difference in findings between Kaplan's Scaling Laws (2020) and the Chinchilla scaling laws (2022)?

- A) Kaplan stated that data quality doesn't matter, while Chinchilla argued that only
Wikipedia should be used.
- **B) Kaplan suggested test loss is more sensitive to parameter count, whereas
Chinchilla disputed this and suggested scaling data and parameters equally.**
- C) Kaplan proved that RLHF is necessary, while Chinchilla proved that Supervised
Fine-Tuning is sufficient.
- D) Kaplan found that larger models require more data to reach the same
performance, while Chinchilla found they require less

2. Which of the following is NOT one of the three primary steps in the InstructGPT
Reinforcement Learning from Human Feedback (RLHF) pipeline?

- A) Collecting demonstration data to train a supervised policy (SFT).
- B) Collecting comparison data where human labelers rank outputs to train a Reward
Model (RM).
- C) Optimising the policy against the Reward Model using the PPO algorithm.
- **D) Utilising a high-precision programmatic grader (like a Python unit test) to assign
a strict 1 or 0 verifiable reward.**

3. Why is standard Supervised Fine-Tuning (SFT) sometimes challenging or insufficient on
its own?
- A) There might be many "OK" answers rather than a single objective right answer.
- B) The model's answer might actually be better than the human label provided.
- C) It is difficult to ensure the model actually learns the underlying reasoning process
rather than just memorising demonstrations.
- **D) All of the above**

## Part 3 SAQ

1. Reward Hacking: Briefly define "Reward Hacking" in the context of LLM training. Provide an example of how an LLM might hack a reward function designed to evaluate a chess-playing assistant.

Reward hacking is taking "improper" actions to seek the highest possible reward but are not actually desired by the model developers.

For e.g., if an LLM is only penalised for losing a chess game, it might learn to always lose a chess game, and have no incentive to win.

2. Verifiable Rewards: What is RLVR (Reinforcement Learning from Verifiable Rewards), and how does its reward mechanism fundamentally differ from standard RLHF?

RLVR uses an algorithmic grader to automatically assess model output.

## Part 4 LAQ

1. The Evals Crisis: The lecture mentions that Machine Learning is currently in a "deep evals crisis". Discuss the three specific reasons for this crisis as outlined in the slides

Crisis is due to (1) explicit hill climbing (2) emergence and (3) contamination.

(1) AI labs train models for specific benchmarks
(2) It is hard to predict what new skills may arise with dataset size, meaning scaling a model may saturate a benchmark unexpectedly
(3) Benchmarks may leak into training code.