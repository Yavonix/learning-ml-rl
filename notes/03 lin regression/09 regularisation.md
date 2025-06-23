## Regularisation

**Regularisation**: Methods for constraining a model's capacity so it doesn't simply memorise the training data but instead learns patterns that generalise

Techniques:
- Collect more data
- Limit model complexity (ie limit the feature count of monomial funcs)
- Weight decay

### Weight Decay

More commonly called regularisation outside of dl circles. Quite popular.

The general idea is to measure function complexity by measuring the distance of parameters from 0, ie the norm.

In general we take the l2 norm of our weights and add that to our loss function we are trying to minimise.

- Using l2 norm is ridge regression.
- Using l1 norm is lasso regression.

Often we do not do this for the bias term.

So we take our original loss function:

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

And combine with the weight cost:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$

We use a nonnegative hyperparameter $\lambda$ to control the extent of weight cost.

So combining the two we get:

$$
J(\mathbf w,b)
=\underbrace{\frac{1}{n}\sum_{i=1}^n \frac12\bigl(\mathbf w^\top \mathbf x^{(i)} + b - y^{(i)}\bigr)^2}_{\displaystyle L(\mathbf w,b)}
\;+\;\underbrace{\frac{\lambda}{2}\|\mathbf w\|^2}_{\text{ridge penalty}}.
$$

And then differentiating:

$$
\begin{align}
J_{\mathcal B}(\mathbf w,b)
&=\frac{1}{\lvert \mathcal B\rvert}\sum_{i\in\mathcal B}\frac12\bigl(\mathbf w^\top\mathbf x^{(i)}+b - y^{(i)}\bigr)^2
\;+\;\frac\lambda2\|\mathbf w\|^2 \\
\nabla_{\mathbf w}J_{\mathcal B}
&= \frac1{\lvert\mathcal B\rvert}\sum_{i\in\mathcal B}
    \bigl(\mathbf w^\top\mathbf x^{(i)}+b-y^{(i)}\bigr)\,\mathbf x^{(i)}
    \;+\;\lambda\,\mathbf w.
\end{align}
$$

Then we update our SGD equation:

$$
\mathbf w \;\leftarrow\;\mathbf w
\;-\;\eta\,\nabla_{\mathbf w}J_{\mathcal B}
\;=\;
\mathbf w
\;-\;\eta\Bigl[\lambda\,\mathbf w
    + \tfrac1{\lvert\mathcal B\rvert}\sum_{i\in\mathcal B}
    \bigl(\mathbf w^\top\mathbf x^{(i)}+b-y^{(i)}\bigr)\mathbf x^{(i)}
    \Bigr].
$$

Rearranging the $\lambda\mathbf w$ term gives the familiar "weight decay" form:

$$
\mathbf w
\;\leftarrow\;
(1-\eta\lambda)\,\mathbf w
\;-\;\frac{\eta}{\lvert\mathcal B\rvert}
    \sum_{i\in\mathcal B}
    \mathbf x^{(i)}\bigl(\mathbf w^\top\mathbf x^{(i)}+b-y^{(i)}\bigr).
$$

## Qs

1. Larger weight decays reduce the maximum accuracy attained by the model.
2. I mean for our model the value that minimises error is lambda = 0.
3. The derivate of the sum of the abs of each term in w is just the vector of the signs of each term in w.
4. Frobenius norm takes the sqrt of the sum of squares of the elements.
5. A google search reveals the following:
   - Elastic net regularisation: combine L1 and L2 regularisation
   - Dropout: randomly set a fraction of neuron activations to 0
   - Early stopping: haha
   - Batch normalisation
6. I have no idea an need to go read [this](https://bjlkeng.io/posts/probabilistic-interpretation-of-regularization/).