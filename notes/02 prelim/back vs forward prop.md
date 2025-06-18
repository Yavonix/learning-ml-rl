## Note

Taken from [stackexchange](https://math.stackexchange.com/a/3119199)

## Ans

The chain rule states that to compute the Jacobian of an operation we should multiply the Jacobians of all sub-operations together. The difference between forward- and reverse-mode auto-differentiation is the **order** in which we multiply those Jacobians.

In your case you only have two sub-operations: \(x y\) and \(\sin()\), leading to only one matrix multiplication, so it isn’t really instructive. However, let’s consider an operation with 3 sub-operations. Take the function:

$$
y = f(x) = r\bigl(q\bigl(p(x)\bigr)\bigr)
$$

where \(x\) and \(y\) are vectors of different lengths. We can break this down into:

$$
a = p(x), \quad b = q(a), \quad y = r(b).
$$

This gives us the Jacobian

$$
\frac{\partial y}{\partial x}
=
\frac{\partial r(b)}{\partial b}
\;\frac{\partial q(a)}{\partial a}
\;\frac{\partial p(x)}{\partial x},
$$

with the size of each matrix noted below it:

- $\displaystyle \frac{\partial r(b)}{\partial b}: |y|\times|b|$  
- $\displaystyle \frac{\partial q(a)}{\partial a}: |b|\times|a|$  
- $\displaystyle \frac{\partial p(x)}{\partial x}: |a|\times|x|$  

The time taken to compute each of those intermediate Jacobians is fixed, but the **order** in which we multiply them changes the number of operations required. Forward-mode auto-differentiation would compute

$$
\frac{\partial y}{\partial x}
=
\frac{\partial r(b)}{\partial b}
\;\Bigl(
  \frac{\partial q(a)}{\partial a}
  \;\frac{\partial p(x)}{\partial x}
\Bigr),
$$

which involves 
\[
|x|\cdot|a|\cdot|b|\;+\;|x|\cdot|b|\cdot|y|
\]
multiplications*, which simplifies to 
\[
|x|\cdot|b|\cdot\bigl(|a|+|y|\bigr).
\]
In contrast, reverse-mode auto-differentiation would compute

$$
\frac{\partial y}{\partial x}
=
\Bigl(
  \frac{\partial r(b)}{\partial b}
  \;\frac{\partial q(a)}{\partial a}
\Bigr)
\;\frac{\partial p(x)}{\partial x},
$$

which involves 
$
|y|\cdot|a|\cdot|b|\;+\;|y|\cdot|a|\cdot|x|
$
multiplications, simplifying to 
$
|y|\cdot|a|\cdot\bigl(|b|+|x|\bigr).
$

Assuming for simplicity that the dimensionality of variables monotonically increases or decreases through the computation, in the case that $|y|\ge|b|\ge|a|\ge|x|$, we can see that forward-mode auto-differentiation results in the same or fewer operations, since

$$
\frac{|y|\cdot|a|}{|x|\cdot|b|}
=
\frac{|y|+|a|}{|b|+|x|},
$$

hence 
$$
|y|\cdot|a|\cdot\bigl(|x|+|b|\bigr)
\;\ge\;
|x|\cdot|b|\cdot\bigl(|a|+|y|\bigr).
$$
Similarly, if $|y|\le|b|\le|a|\le|x|$, then reverse-mode auto-differentiation results in the same or fewer operations.

This means that reverse-mode auto-differentiation (a.k.a. back-propagation) will usually be faster when
$$
f\colon \mathbb{R}^n \to \mathbb{R}^m,\quad m \ll n,
$$
i.e. the cost-function output is low-dimensional (e.g. a scalar loss), but the input is high-dimensional (e.g. pixels in an image or words in text), as is generally the case in neural-network training.

This reasoning on whether forward- or reverse-mode is preferable extends to longer chains of Jacobians. However, exceptions can occur, e.g. when the lowest dimensionality of variables occurs neither at the function input nor output, but somewhere in between. In such cases, the optimal ordering of matrix multiplications won’t be fully forward- or reverse-mode, but a **hybrid** scheme.

I’ve discussed theoretical/idealised considerations, but there are practical considerations too. For example, reverse-mode auto-differentiation requires a forward pass through the code to compute values, then a reverse pass to compute the derivatives. A trace of the values needs to be stored during the forward pass in order to compute the reverse pass. This increases the complexity of implementing and running reverse-mode auto-differentiation.

\* Multiplication counts ignore constant factors and additive terms.
