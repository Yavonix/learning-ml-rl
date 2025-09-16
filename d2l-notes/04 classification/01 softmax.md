## Softmax

![Softmax regression is a single-layer neural network.](./img/softmaxreg.svg)

$$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$$

Now we can treat this as a vector valued regression problem but then we have No guarantee outputs:

- sum to 1
- are nonnegative
- do not exceed one

Rather, a softmax solves these issues:

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \textrm{where}\quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.$$

Similar to how we used MLE to derive the loss function for linear regression, we may use MLE to derive the loss function here.

So we have:

$$P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).$$

And $P\bigl(y^{(i)}\mid x^{(i)}\bigr)$ is:

$$
P\bigl(y^{(i)}\mid x^{(i)}\bigr)
= \prod_{j=1}^q \bigl(\hat y^{(i)}_j\bigr)^{y^{(i)}_j},
$$

(intuitively this makes sense given $y^{(i)}$ is a one hot encoded vector)

Let's combine those two to form our starting point:

$$
    P(\mathbf{Y}\mid \mathbf{X})
    = \prod_{i=1}^n 
    \Bigl(\prod_{j=1}^q (\hat y^{(i)}_j)^{y^{(i)}_j}\Bigr)
    = \prod_{i=1}^n \,\prod_{j=1}^q (\hat y^{(i)}_j)^{y^{(i)}_j}.
$$

Then take the negative log:

$$
\begin{align}
    -\log P(\mathbf{Y}\mid \mathbf{X})
    &= -\log 
    \Bigl[
        \prod_{i=1}^n \prod_{j=1}^q (\hat y^{(i)}_j)^{y^{(i)}_j}
    \Bigr] \\
    &= - \sum_{i=1}^n \sum_{j=1}^q 
         \log\bigl[(\hat y^{(i)}_j)^{y^{(i)}_j}\bigr] \\
    &= - \sum_{i=1}^n \sum_{j=1}^q 
         y^{(i)}_j \,\log \hat y^{(i)}_j
\end{align}
$$

Lo and behold for any pair of label $\mathbf{y}$ and model prediction $\mathbf{\hat y}$ over $q$ classes, the loss function $l$ is:

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j.
$$

This is commonly called *cross-entropy loss*.

Note that $l(\mathbf{y}, \hat{\mathbf{y}}) \ge 0$ as $\log \hat{y}_j \le 0$ as no entry in $\hat{y}_j$ exceeds 1.

## Problems with Softmax

Softmax as we've described it is prone to instability. Very large values from our underlying output layer $\mathbf{o}$ may cause **numerical overflow** causing infs.

### Potential Solution 1

We could subtract $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ from all entries:

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

This is still not perfect though. We may encounter **numerical underflow** where $\exp(o_j - \bar{o})$ evaluates to 0, which will causes infs when we take $\log \hat{y}_j$ as $\log 0$.

### Potential Solution 2

Rather, we could combine the softmax step into the loss step:

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o})
$$

Avoiding both overflow and underflow. Optax has this implemented as `softmax_cross_entropy_with_integer_labels`.




## Connecting SoftMax to Information Theory

### Surprisal

In Information Theory, there is a concept of "surprisal". Surprise is a way of quantifying how unexpected an outcome is.

Just thinking about it, for an event $E$ we want something that:

1. Increases as the probability of the event decreases.
2. Only applies to known events.
3. The surprise for an event $p(E)=1$ is $0$.

We might expect that the surprise related to an event is the reciprocal of the event, ie $I(E)=1/p(E)$. However, if an event is 100% likely then we have $I(E)=1$ failing (3).

Instead let's define surprise as:

$$I(E)=\log(\frac{1}{p(E)})$$

This way:
1. As $p(E)$ increases, $I(E)$ decreases.
2. For an event with $p(E)=0$, $I(E)=\log(\frac{1}{0})=\mathrm{undefined}$
3. The surprise for an event $p(E)=1$ is $0$:  $I(E)=\log(\frac{1}{1})=0$

More commonly, surprise is written as:

$$I(E)=-\log(p(E))$$

### Entropy

Then, we define a new concept *entropy* to represent the expected surprisal for a given random variable. For a distribution $P$, its entropy $H[P]$ is defined as:

$$H[P] = \sum_j - P(j) \log P(j)$$

### Cross-Entropy

Cross-entropy is the expected surprisal of an observer with subjective probabilities $Q$ upon seeing data that was actually generated according to probabilities $P$:

$$H(P, Q) \stackrel{\textrm{def}}{=} \sum_j - P(j) \log Q(j)$$

So this is effectively the weighted average of the surprise of $Q$ with weights from $P$.

Therefore we can think of the cross-entropy classification objective in two ways:
1. Maximising the likelihood of the observed data
2. Minimising our surprisal
