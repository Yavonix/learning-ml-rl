## 03.01 Linear Regression

### Model

For a dataset, $x^{(i)}$ represents the i'th sample while $x^{(i)}_j$ represents the j'th coordinate.

The design matrix $X$ contains one row for every example and one column for every feature.

Then $\hat y = X w + b$.

### Loss Function

For a single example the loss function may be: (squared error)

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

The constant $\frac{1}{2}$ makes no real difference but proves to be notationally convenient.

To measure model quality on a dataset of $n$ examples we take the MSE:

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Then we minimise the model parameters:

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### Analytic Solution

We subsume bias $b$ into $w$ (append 1s col to the design matrix) then minimise $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$. Assuming the design matrix is full rank, we have:

$$\begin{aligned}
    \partial_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 =
    2 \mathbf{X}^\top (\mathbf{X} \mathbf{w} - \mathbf{y}) = 0
    \textrm{ and hence }
    \mathbf{X}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{X} \mathbf{w}.
\end{aligned}$$

Which gives us:

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}$$

### Gradient Descent

#### Stochastic Gradient Descent

The idea of taking the derivative of the loss function computed over **all examples**, then updating the model parameter. However this requires a complete pass over the dataset which can be slow.

The opposite is to update the parameters of a single example. Reading/writing single examples however is slow and does not take advantage of parallisation.

#### Minibatch Stochastic Gradient Descent

The idea of selecting a few examples to use when updating the model. Usually something between 32 and 256 (a large power of 2).

General process:

1. For each iteration $t$, we randomly sample a minibatch $\mathcal{B}_t$. 
2. We compute the gradient of the MSE on the minibatch.
3. Multiple by $\eta$ to control the learning rate.
4. Update model parameters by subtracting the current term.

Overall:

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

In the text they also show a closed form solution but like that requires everything be linear and even then, with multiple layers it going to suffer from expression swell.

Minibatch size and learning rate are user defined. User defined values are called **hyperparameters**. Apparently they can be tuned via Bayesian Optimisation (Frazier 2018).

### Prediction

Given the model $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$, we can now make *predictions* for a new example.

### The Normal Distribution and Squared Loss

Recall:

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

Then we assume observations come from noisy measurements:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \textrm{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Then we have:

$$
p(\epsilon)
= \frac{1}{\sqrt{2\pi\sigma^2}}
  \exp\!\Bigl(-\frac{\epsilon^2}{2\sigma^2}\Bigr).
$$

And:

$$
y = w^\top x + b + \epsilon
\quad\Longrightarrow\quad
\epsilon = y - (w^\top x + b).
$$

So:

$$
P(y \mid x)
= p\bigl(\epsilon = y - w^\top x - b\bigr)
= \frac{1}{\sqrt{2\pi\sigma^2}}
  \exp\!\Bigl(-\frac{\bigl(y - w^\top x - b\bigr)^2}{2\sigma^2}\Bigr).
$$

Thus, we can now write out the *likelihood* of seeing a particular $y$ for a given $\mathbf{x}$ via

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

Then, according to *the principle of maximum likelihood*, the best values of parameters $\mathbf{w}$ and $b$ are those that maximize the *likelihood* of the entire dataset:

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)} \mid \mathbf{x}^{(i)}).$$

To make things easier we will then rewrite this using *log* as that does not change the optimisation objective and also use a negative for historical reasons. This gives us the *negative log-likelihood*:

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Then, if we assume that $\sigma$ is fixed, we can ignore the first term, because it does not depend on $\mathbf{w}$ or $b$. The second term is identical to the squared error loss introduced earlier, except for the multiplicative constant $\frac{1}{\sigma^2}$. Fortunately, the solution does not depend on $\sigma$ either.
It follows that minimizing the mean squared error is equivalent to the maximum likelihood estimation of a linear model under the assumption of additive Gaussian noise.
