## Distribution Shift

So far we've been training discriminative models to directly approximate $\displaystyle \hat P_{\text{train}}(y\mid x)$ from a labelled dataset whose empirical frequencies give us $\displaystyle P_{\text{train}}(y)$, $P(x\mid y)$ (up to a constant), and $P_{\text{train}}(x)$. By invoking Bayes’ theorem

$$
P(y\mid x)
= \frac{P(x\mid y)\,P(y)}{P(x)},
$$

we can analyse exactly how the posterior depends on the prior $P(y)$, the likelihood $P(x\mid y)$ and the normalising evidence $P(x)$.

That makes explicit that even though a discriminative model doesn’t estimate $P(x)$ or $P(x\mid y)$ directly, all three quantities are implicitly defined by the training data and shape the learned $\hat P(y\mid x)$.


### Types of Distribution Shift

#### Covariate Shift
Probably the most widely studied.

- Distribution of the covariates between model training and deployment changes. 
- The way the covariates relate to the outcome is unchanged.
- $P_{train}(x)\ne P_{test}(x)$
- (as a consequence) $P_{train}(y)\ne P_{test}(y)$
- $P_{train}(y\mid x) = P_{test}(y\mid x)$

In other words, the relationship between features and outcome in the dataset is unchanged, but the covariates in the dataset have changed.

See what happens with a perfect model [here](#bayes-rule-covariate-shift-perfect-discriminative-model).

#### Label Shift
- Distribution of the labels between modeling training and deployment changes
- Conditional distribution of covariates given the labels remains constant
- $P_{train}(y)\ne P_{test}(y)$
- (as a consequence) $P_{train}(x)\ne P_{test}(x)$
- $P_{train}(x\mid y)= P_{test}(x\mid y)$

Why this matters:
- Say we train a model to discriminate between C19 and the flu.
- When training the prevalence of C19 and the flu is 10% and 90% respectively.
- Because C19 and the flu may share the same features, our predictive model will bake in the posteriors when making predictions.
- In the target distribution, C19 and the flu may have prevalence of 90% and 10% respectively (flipped).
- The model will tend to misclassify C19 as the flu because the features overlap and old probability posteriors have been baked in.

#### Concept Shift
- $P_{train}(x)= P_{test}(x)$
- $P_{train}(y)= P_{test}(y)$
- $P_{\text{train}}(y\mid x)\ne P_{\text{test}}(y\mid x)$

E.g., diagnostic criteria for mental illness, what passes for fashionable, and job titles



#### Bayes Rule Covariate Shift Perfect Discriminative Model

**"Perfect" discriminative model**

Assume the model has learned the **exact** posterior on the training domain

$$
\hat P(y\mid x)=P_{\text{train}}(y\mid x).
$$

**Bayes-rule derivation**

1. **Bayes on the training set**

   $$
   P_{\text{train}}(y\mid x)
   =\frac{P(x\mid y)\,P(y)}{P_{\text{train}}(x)},
   \tag{1}
   $$

   where $P(y)$ is the common class prior (unchanged under covariate shift).

2. **Solve (1) for the likelihood**

   $$
   P(x\mid y)
   =P_{\text{train}}(y\mid x)\;
     \frac{P_{\text{train}}(x)}{P(y)}
   =\hat P(y\mid x)\;
     \frac{P_{\text{train}}(x)}{P(y)}.
   \tag{2}
   $$

3. **Bayes on the test set**
   Use the same likelihood $P(x\mid y)$ (covariate shift keeps it fixed) but the new evidence $P_{\text{test}}(x)$:

   $$
   P_{\text{test}}(y\mid x)
   =\frac{P(x\mid y)\,P(y)}{P_{\text{test}}(x)}.
   \tag{3}
   $$

4. **Substitute (2) into (3)**

   $$
   P_{\text{test}}(y\mid x)
   =\frac{\hat P(y\mid x)\,
           \tfrac{P_{\text{train}}(x)}{P(y)}\;
           P(y)}
          {P_{\text{test}}(x)}
   =\hat P(y\mid x)\,
     \frac{P_{\text{train}}(x)}{P_{\text{test}}(x)}.
   \tag{4}
   $$

5. **Normalise over all classes**
   The scaling factor $\dfrac{P_{\text{train}}(x)}{P_{\text{test}}(x)}$ is **independent of $y$**, so when probabilities are re-normalised it cancels:

   $$
   P_{\text{test}}(y\mid x)=\hat P(y\mid x)=P_{\text{train}}(y\mid x).
   $$

Because the discriminative model already supplies the true conditional $P(y\mid x)$ and covariate shift alters only the marginal $P(x)$, its predictions remain exactly correct. A “perfectly generalising” classifier is therefore unaffected by covariate shift.

#### Bayes Rule Label Shift Perfect Feature Pattern

Suppose there’s a feature pattern $x^*$ that *only* ever occurs with COVID-19 and *never* with flu.  In symbols:

* $P(x^*\mid \mathrm{C19})>0$
* $P(x^*\mid \mathrm{Flu})=0$

Apply Bayes’ rule under *any* priors $P(Y)$:

$$
\begin{align}
P(Y=\mathrm{C19}\mid x^*) 
&= \frac{P(x^*\mid \mathrm{C19})\,P(\mathrm{C19})}
       {P(x^*\mid \mathrm{C19})\,P(\mathrm{C19})
       +P(x^*\mid \mathrm{Flu})\,P(\mathrm{Flu})} \\
&= \frac{P(x^*\mid \mathrm{C19})\,P(\mathrm{C19})}
       {P(x^*\mid \mathrm{C19})\,P(\mathrm{C19})+0}
=1
\end{align}
$$