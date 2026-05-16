numpy advanced indexing triggered when selection object is non-tuple sequence object or ndarray or tuple with non-tuple sequence object or ndarray

numpy advanced indices are always broadcast to be the same shape and iterated as one
- ie numpy first makes those index arrays the same shape by broadcasting, then walks through them element-by-element together


```
x = np.array([
    [10, 11, 12],
    [20, 21, 22],
    [30, 31, 32]
])

```
x[[0, 1, 2], [2, 1, 0]] will be (0, 2), (1, 1), (2, 0)
The two index arrays are