## Overfitting/Underfitting

training error == validation error == high:
- model too simple
- underfitting

training error << validation error:
- overfitting
- not always a bad thing. We mainly care about reducing validation error.

## Model Selection

- do not rely on test data for model selection

### Cross Validation
Some bits from [here](https://machinelearningmastery.com/k-fold-cross-validation/).

In circumstances where we do not have enough training data to create separate validation and training sets for each hyper-parameter or model variant, we employ cross validation. A type of cross-validation is K-fold validation.

In K-fold validation, split the training data in $K$ subsets. Then fit a model on $K-1$ subsets and validate on the remaining subset. The procedure is as follows:

1. Shuffle the dataset randomly.
2. Split the dataset into k groups (folds)
3. For each unique fold i:
   1. Take fold i as a hold out or test data set
   2. Take the remaining folds as a training data set
   3. Fit a model on the training set and evaluate it on the test set
   4. Retain the evaluation score and discard the model
4. Summarize the skill of the model using the sample of model evaluation scores

The results of a k-fold cross-validation run are often summarized with the mean of the model skill scores. It is also good practice to include a measure of the variance of the skill scores, such as the standard deviation or standard error.

Choosing $K$:
- K should be sufficiently small so that each train/test fold has enough data samples to be statistically representative of the broader dataset.
- K is generally 5 or 10, there is no formal rule.

## Qs

1. Assuming 1 dimensional space. The order of the polynomial = number of examples - 1. Ie the design matrix is full rank.
2. What if every example was not IID?
   Dependent examples include:
   1. Time-series forecasting: Weather prediction.
       - Model's goal: given features at time t (say temperature, pressure on Monday), predict the target at time t + 1 (Tuesday's temperature).
       - Tuesday is statistically dependent on Mon.
   2. Repeated measurements on the same unit: Longitudinal medical studies
       -  Model's goal: predict a patient’s outcome (say blood pressure) at each clinic visit based on that visit’s covariates.
       -  Observations for "Patient A, Visit 1" and "Patient A, Visit 2" aren’t independent—underlying physiology, lifestyle, medication effects carry over. 
   3. Spatial data with autocorrelation: Soil nutrient sampling
       -  predict nutrient concentration at a location (x,y) from local measurements (e.g.\ pH, moisture).
       -  Two samples taken at nearby sites “know about” each other via spatial processes (nutrients diffuse, moisture patterns span metres).
   4. Related individuals: genetic studies
   5. Dataset drift: The underlying distribution might change with time.
3. Not really, we can get close. If the model is able to perfectly fit the data. Ie the model complexity exceeds the data points available or the data is highly generalisable. Zero generalisation errors occurs when no noise is in the underlying data and the training and validation datasets are truly IID.
4. Because it involves training K models.
5. I dont get the question? What?
6. It ignores the complexity of the functions in each class.
7. Show proof of overfitting? Training error decreasing while generalisation (validation) error increasing.


## Modelling dependence

Taken from GPT:
When we write $p(x,y)$, we mean “the joint probability density (or mass) of a *single* example \((X,Y)\).” Under IID you assume your dataset
$
\{(x^{(1)},y^{(1)}),\dots,(x^{(n)},y^{(n)})\}
$
is drawn so that
$
p\bigl(x^{(1)},y^{(1)},\dots,x^{(n)},y^{(n)}\bigr)
=\prod_{i=1}^n p\bigl(x^{(i)},y^{(i)}\bigr).
$

---

### When it’s *not* IID

1. **Temporal dependence (time series):**  
   Instead of factorising as above, the joint over the whole sequence is  
   $$
   p\bigl(x^{(1)},y^{(1)},\dots,x^{(n)},y^{(n)}\bigr)
   =p\bigl(x^{(1)},y^{(1)}\bigr)\,
   \prod_{t=2}^n
   p\bigl(x^{(t)},y^{(t)}\mid x^{(t-1)},y^{(t-1)},\dots\bigr).
   $$
   Here each \((x^{(t)},y^{(t)})\) depends on the history, so you can’t write it as a product of identical marginals.

2. **Dataset shift / non-identical:**  
   You might have one distribution \(P\) in training and another \(Q\) in testing, so  
   $$
   (x^{(i)},y^{(i)})\sim P,\quad
   (x_{\rm test},y_{\rm test})\sim Q,
   $$
   with \(P\neq Q\). Even if they’re independent draws, they’re not identically distributed.

3. **Spatial or relational dependence:**  
   If examples are correlated by proximity or pedigree, the full joint doesn’t factorise. You’d write something like  
   $$
   p\bigl(x^{(1)},y^{(1)},x^{(2)},y^{(2)}\bigr)
   \neq p\bigl(x^{(1)},y^{(1)}\bigr)\,p\bigl(x^{(2)},y^{(2)}\bigr),
   $$
   and instead might need a model for  
   $$
   p\bigl(x^{(2)},y^{(2)}\mid x^{(1)},y^{(1)}\bigr).
   $$

---

**In summary:**  
- \(p(x,y)\) is the joint law for *one* example.  
- IID lets you write the overall likelihood as \(\prod_i p(x^{(i)},y^{(i)})\).  
- Breaking IID means either adding conditionals (for dependence) or switching to different distributions (for shift).
