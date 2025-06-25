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

Where $\epsilon_\mathcal{D}(f)$ is a statistical estimate of the population error $\epsilon(f)$. Let $p=\epsilon(f)$ thus $Z_i\sim\mathrm{Bernoulli}(p)$

Then

$$
\epsilon_{\mathcal D}(f) =\frac1n\sum_{i=1}^n Z_i,
$$

So then we know that:

- $\mathbb{E}[\epsilon_{\mathcal D}(f)]=p$
- $\mathbf{Var}[\epsilon_{\mathcal D}(f)]=p(1-p)$




By contrast *population error* is the *expected* fraction of examples in the underlying population (some $P(X,Y)$ with PDF $p(x,y)$) is:

$$
\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy.
$$

We can view $\epsilon_\mathcal{D}(f)$ as a statistical estimator of $\epsilon(f)$.

  Then

$$

\qquad
\mathrm{Var}\bigl(\epsilon_{\mathcal D}(f)\bigr)
=\mathrm{Var}\!\Bigl(\tfrac1n\sum_i Z_i\Bigr)
=\frac1{n^2}\sum_i\mathrm{Var}(Z_i)
=\frac{\sigma^2}{n}.
$$

Taking the square‐root gives the standard error

$$
\sqrt{\mathrm{Var}\bigl(\epsilon_{\mathcal D}(f)\bigr)}
=\frac{\sigma}{\sqrt n}.
$$

Therefore