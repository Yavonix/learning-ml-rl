# 3008ICT Deep Learning
## Homework 2-3
*No submission is required.*

---

**1.** For the task of recognising that an object is a dog or not, we can use the Linear Regression model. If the model is regarded as a neural network, which of the following statements are correct?

- (a) There is a single unit (neuron) in the output layer. TRUE.
- (b) There are exactly two units (neurons) in the output layer. FALSE.
- (c) The output of the model is a real number between 0 and 1, which can be interpreted as the probability that the object in input image is a dog. TRUE (post sigmoid activation function)
- (d) After choosing a threshold, you can convert the model's output into a category of 0 or 1. TRUE.

---

**2.** Which of these are terms used to refer to components of an artificial neural network?

- (a) Axon. FALSE. Part of a neuron.
- (b) Activation function. TRUE.
- (c) Layers. TRUE.
- (d) Neurons. TRUE. Represents the affine transformation followed by activation function.

---

**3.** True/False? Neural networks take inspiration from, but not very accurately mimic, how neurons in a biological brain learn.

- (a) True
- (b) False

TRUE

---

**4.** For the following code:
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

4 Layers. 1 Input, 2 hidden, 1 output.

---

**5.** How do you define a linear regression model for a learning problem from a dataset with three features in PyTorch?

```
nn.Linear(3,1)
```

---

**6.** Suppose our Softmax regression model has 100 nodes (neurons) in the input layer and five nodes in the output layer. If bias terms are not regarded as weights, what is the size of the input vector **x**? What is the size of the matrix **w** for weights? What is the size of the bias vector? For the second output node, is its weight vector the 2nd row or the 2nd column of the matrix **w**?

Input layer: 100
Output layer: 5

Design matrix X is (num_examples, 100)
x is (100)
b is (5)
w is (5, 100)
The second output node is the 2nd row of the matrix w.

---

**7.** Which of these is the best way to determine whether your model has underfitting for the training set?

- (a) See if the training error is high (above 20% or so).
- (b) See if the validation error is high compared to the baseline level of performance.
- (c) Compare the training error to the validation error.
- (d) Compare the training error to the baseline level of performance.

D. Compare training error to baseline level of performanc.

---

**8.** Compare the activation functions ReLU and Sigmoid (or Logistic function), briefly explain the advantages of ReLU over Sigmoid.

ReLU(z) = max(0, z)
Sigmoid(z) = \frac{1}{1+e^{-z}}

Advantages of ReLU:
- No vanishing gradient for positive inputs
- Computation efficiency
- Faster empirical convergence

Advantages of sigmoid:

As the gradient of Sigmoid is not 0 for $z<0$, it does not suffer from the dying ReLU problem. 