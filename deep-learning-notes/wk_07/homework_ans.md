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