## Numerical Stability and Initialisation

### Exploding Gradients

Matrix multiplications during backpropagation may cause gradient calculations to become unreasonably unless the weights are initialised to a reasonable value.

### Symmetry

If all weights are identical, all gradient updates are identical.

Dropout regularisation helps to remedy this issue.

## Parameter Initialisation Schemes

### Xavier Initialisation

## Generalisation in Deep Learning

In deep learning, we typically choose model architectures that can achieve arbitrarily low training loss (and zero training error). Therefore, the only avenue for further gains is to reduce overfitting.

Strangely, we can mitigate overfitting by increasing the complexity of the model (e.g., adding layers, nodes or training longer).

The relationship between generalization gap and model complexity can be non-monotonic, with greater complexity hurting at first but subsequently helping in a so-called "double-descent" pattern (Nakkiran et al., 2021).


### Regularisation Techniques

- Early stopping: A new line of work (Rolnick et al., 2017) has revealed that **in the setting of label noise**, neural networks tend to fit cleanly labeled data first and only subsequently to interpolate the mislabeled data. Notably, when there is no label noise and datasets are realizable (the classes are truly separable, e.g., distinguishing cats from dogs), early stopping tends not to lead to significant improvements in generalization. 

- Weight decay: Still a commonly used technique. Doesn't work by the classic "capacity‐reduction" narrative (Zhang et al., 2021), might work due to encoding bias in the model?