## Perplexity
(recap from wk 6 notes)

Perplexity is the exponential of cross-entropy averaged over all timesteps, so that sequences of varying timesteps are comparable

We need some way of comparing loss between variable length sequences. This is defined as perplexity:

Let $x_{<t} = x_1, \dots, x_{t-1}$ and $V$ be all possible tokens.

$$
\text{Perplexity} = \exp\left(- \frac{1}{n} \sum_{t=1}^n \sum_{v \in V} P(v \mid x_{<t}) \cdot \ln \hat{P}(v \mid x_{<t}) \right)
$$

Which when one-hot encoded:

$$
\text{Perplexity} = \exp\left(- \frac{1}{n} \sum_{t=1}^n\cdot \ln \hat{P}(x_t \mid x_{<t}) \right)
$$

By example, suppose:
- $P(w_1 \mid w_0) = \frac{1}{4}$
- $P(w_2 \mid w_1, w_0) = \frac{1}{3}$
- $P(w_3 \mid w_2, w_1, w_0) = \frac{1}{4}$

Then:

$$
\text{Perplexity}
=
\exp\left(
-\frac{1}{3}
\left(
\log \frac{1}{4}
+
\log \frac{1}{3}
+
\log \frac{1}{4}
\right)
\right)
\approx 7.28
$$

## Definitions

- Bitter lesson: scalable general methods tend to beat hand-engineered methods
- Scaling law: Larger models trained on more data with more compute yield better results

## Pretraining

Train model on X, fine-tune on Y.

For language models, pre-training usually means learning from large text corpora using a self-supervised objective, such as:

- **GPT:** predict the next token  
  $P(x_t \mid x_{<t})$

- **BERT:** predict masked tokens using surrounding context  
  $P(x_{\text{mask}} \mid x_{\text{left}}, x_{\text{right}})$

After pre-training, the model has learned general language representations. It can then be adapted to a specific task by **fine-tuning** or by adding a task-specific head.