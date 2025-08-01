## 03.01 Linear Regression Summary

So the general process is:

1. Pick your model family (linear regression, nn, decision tree)
2. Pick a linking function
   - Map the raw predictor $z=w^T\cdot x+b$ into the domain of y.
   - Let $h(z)$ represent the transform. Then some examples are:
   - $\R$: identity ($h(z)=z$)
   - $(0, \inf)$: $\exp$ ($h(z)=e^z$)
   - Probability simplex: softmax
3. Define the model error distribution
   - Decide how the residuals are distributed (Guassian, Laplace, Poisson, Categorical)
   - Continuous targets: $y = h(z) + \epsilon,\quad \epsilon \sim p(\epsilon),$
   - Categorical targets: $\Pr\bigl(y=i \mid x\bigr) \;=\; \bigl[\mathrm{softmax}(z)\bigr]_i.$
4. Pick the appropriate loss function. See [deriving loss functions](#deriving-loss-functions)
5. Train and evaluate:
   - Per-example loss $\ell_i = -\log p\bigl(y^{(i)} \mid x^{(i)};\mathbf w,b\bigr)$.
   - Define loss $L(\mathbf w,b) = -\sum_{i=1}^n \log p\bigl(y^{(i)} \mid x^{(i)};\mathbf w,b\bigr)$.
   - Mini-batch updates: for each batch $B$ of size $m$, compute  
     $g_w = \tfrac1m\sum_{j\in B}\nabla_{\mathbf w}\,\ell_j$  
     and  
     $g_b = \tfrac1m\sum_{j\in B}\tfrac{\partial \ell_j}{\partial b}$,  
     then update  
     $\mathbf w \leftarrow \mathbf w - \eta\,g_w$,  
     $b \leftarrow b - \eta\,g_b$.
   - Validate: hold out a validation set, monitor training vs. validation loss, and inspect residuals or accuracy.

<a id="deriving-loss-functions"></a>
### Deriving loss functions

Note that at the end of each section we divide by $n$ so that we:
 - Make the loss interpretable as an average error per example.
 - Keep gradient magnitudes stable when batch size changes.
 - Allow comparing losses across datasets of different sizes.

This averaged form is usually called empirical risk ($\mathcal R$)

#### Assuming noise is normally distributed

Let's assume our data has normally distributed noise:
$$y = h(z) + \epsilon,\quad \epsilon \sim N(0, \sigma^2),\quad
z^{(i)} = \mathbf w^\top x^{(i)} + b.$$

Then the residual is $\epsilon = y - h(z)$.

The probability of any given $\epsilon$ value is:

$$p(\epsilon) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (\epsilon)^2\right).$$

And we want to maximise that probability given our data. Therefore we need an expression that shows how that probability varies with our predictions. Let's substitute $\epsilon = y - h(z)$: 

$$p_\epsilon\bigl(y - h(z)\bigr)
= \frac1{\sqrt{2\pi\sigma^2}}
  \exp\!\Bigl(-\tfrac1{2\sigma^2}\bigl(y - h(z)\bigr)^2\Bigr)$$

This gives the likelihood expression for a single example. Let's compute the likelihood across several examples, ie the joint likelihood: (given the notation $P(\mathbf y \mid \mathbf X)$)

$$
\begin{align}

P(\mathbf y \mid \mathbf X) &= L(\mathbf w,b) \\
&= \prod_{i=1}^n p\!\bigl(y^{(i)} - h(z^{(i)})\bigr) \\
&= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}
  \exp\!\Bigl(-\frac{(y^{(i)} - h(z^{(i)}))^2}{2\sigma^2}\Bigr) \\
&= (2\pi\sigma^2)^{-\frac{n}{2}}
  \exp\!\Bigl(-\frac{1}{2\sigma^2}\sum_{i=1}^n (y^{(i)} - h(z^{(i)}))^2\Bigr).

\end{align}
$$

To make things easier let's rewrite this using log. We can do this as that does not change the optimisation objective. We'll also make it negative due to historical convention. This gives us the negative log-likelihood:

$$
\begin{align}
-\log (L(\mathbf w,b)) &= -\Bigl[-\tfrac{n}{2}\log(2\pi\sigma^2)
    -\frac{1}{2\sigma^2}\sum_{i=1}^n\bigl(y^{(i)} - h(z^{(i)}) \bigr)^2\Bigr] \\

&= \frac{n}{2}\log(2\pi\sigma^2)
  + \frac{1}{2\sigma^2}\sum_{i=1}^n\bigl(y^{(i)} - h(z^{(i)})\bigr)^2.
\end{align}
$$

Recall that our primary interest in the loss function is to use it to update our model parameters - ie model weights and biases. Lets remove the variables that our solution does not depend on:

$$
\mathcal{L}(\mathbf w,b) = \sum_{i=1}^n\bigl(y^{(i)} - h(z^{(i)})\bigr)^2
$$

Averaging by $n$ we get Mean Squared Error ($\mathrm{MSE}$):

$$\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^n (y^{(i)} - \hat y^{(i)})^2.$$

#### Assuming noise is Laplacian

This one's a bit simpler. Let's assume our data has Laplacian noise:
$$
y = h(z) + \epsilon,\quad 
\epsilon \sim \mathrm{Laplace}(\mu,\sigma).
$$

The residual is
$$
\epsilon = y - h(z),
$$
so its density is
$$
p(\epsilon)
= \frac{1}{2\sigma}
  \exp\!\Bigl(-\frac{|\epsilon - \mu|}{\sigma}\Bigr).
$$

Substituting $\epsilon = y - h(z)$ gives
$$
p\bigl(y - h(z)\bigr)
= \frac{1}{2\sigma}
  \exp\!\Bigl(-\frac{|y - h(z) - \mu|}{\sigma}\Bigr).
$$

The joint likelihood over $n$ examples is
$$
L(\mathbf w,b)
= \prod_{i=1}^n p\bigl(y^{(i)} - h(z^{(i)})\bigr)
= (2\sigma)^{-n}
  \exp\!\Bigl(-\tfrac1\sigma\sum_{i=1}^n |y^{(i)} - h(z^{(i)}) - \mu|\Bigr).
$$

Taking the negative log gives
$$
-\log L(\mathbf w,b)
= n\log(2\sigma)
  + \frac{1}{\sigma}\sum_{i=1}^n \bigl|y^{(i)} - h(z^{(i)}) - \mu\bigr|.
$$

Dropping constants and setting $\mu=0$:
$$
\mathcal L(\mathbf w,b)
= \sum_{i=1}^n \bigl|y^{(i)} - h(z^{(i)})\bigr|.
$$

Averaging by $n$ we get Mean Absolute Error ($\mathrm{MAE}$):

$$\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^n \bigl|y^{(i)} - h(z^{(i)})\bigr|$$
