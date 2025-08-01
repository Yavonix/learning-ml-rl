## Generalisation

Recall that we divide our dataset into 3:
- Training set
- Validation set
- Test set

### Examples Required in the Test Set to Guarantee Generalisation

**A priori**: to establish a theoretical bound on how many samples we need to guarantee training error will be within ε of the true error before seeing any training data. However doing so typically results in an absurd number of examples (perhaps trillions or more). Therefore we ofter forego a priori guarantees altogether.

**A posteriori**: to assess generalisation after training by measuring error on a held-out test set.

Assume:
- a fresh dataset $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$ 
- a classifier $f$

Then *empirical error* of $f$ on $\mathcal{D}$ is the fraction $f(\mathbf{x}^{(i)})$ disagrees with the true label $y^{(i)}$:

$$
\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)}).
$$

The error rate is the fraction wrong.

Assume population error rate $p$.

Let $Z\sim\mathrm{Bernoulli}(p)$

Then $\mathbf{E}[Z]=p$ and $\mathbf{Var}[Z]=p(1-p)$

Then $\displaystyle\epsilon_n = \frac{1}{n}\sum_{i=1}^{n}Z_i$.

Then $\mathbf{E}[\epsilon_n]=p$ and $\mathbf{Var}[\epsilon_n]=\frac{p(1-p)}{n}$

Then assuming $Z_i$ is i.i.d, by CLT:

$$\frac{\epsilon_n-\mathbf{E}[\epsilon_n]}{\mathbf{Var}[\epsilon_n]}\sim N(0, 1)$$

In other words:

$$\hat p \sim N\left(p, \frac {p(1-p)}{n}\right)$$

Given $p(1-p)$ is maximised when $p=0.5$, then the **asymptotic** standard deviation of our estimate $\epsilon_\mathcal{D}(f)$ of the error $\epsilon_\mathcal{D}(f)$ cannot be greater than $\sqrt{0.25/n}$.

## Hoeffding's theorem

Used when you need a **non-asymptotic** estimate of n.

Provides an upper bound on the probability that the sum of bounded independent random variables deviates from its expected value by more than a certain amount. For all $t > 0$:
$$\begin{align}
\operatorname{P} \left(S_n - \mathrm{E}\left [S_n \right] \geq t \right) &\leq \exp \left(-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2} \right) \\
\operatorname{P} \left(\left |S_n - \mathrm{E}\left [S_n \right] \right | \geq t \right) &\leq 2\exp \left(-\frac{2t^2}{\sum_{i=1}^n(b_i - a_i)^2} \right)
\end{align}$$