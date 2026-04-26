## Part 1
1. D
2. C
3. B
4. B
5. A

## Part 2 T/F
6. F
7. T
8. T


## Part 3 Understanding
9. IDK what the netflix rating system is but I assume that dataset drift over time 

10. We have two gates controlling cell state in a LSTM: forget and input gate. Both are linear layers based on prior hidden state and the current timestep input passed through a sigmoid function (therefore outputting between 0 and 1). In the given cell update equation, $F_t$ controls how much of the prior cell state is preserved into the current cell state, while $I_t$ controls how much candidate cell state is merged with the prior cell state.

11.
```python
features = [self.x[i: self.T-self.tau+i] for i in range(self.tau)]

## given x = [10, 20, 30, 40, 50, 60], tau = 3
features = [
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50]
]

## given x = [10, 20, 30, 40, 50, 60], tau = 3
self.labels = self.x[self.tau:].reshape((-1, 1))

## self.x[self.tau:] is  [40, 50, 60]
self.labels = [[40], [50], [60]]
```