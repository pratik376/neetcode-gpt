import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                             x: List[float],
                             W1: List[List[float]], b1: List[float],
                             W2: List[List[float]], b2: List[float],
                             y_true: List[float]) -> dict:
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        # Forward pass
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2

        loss = np.mean((z2 - y_true) ** 2)

        # Backward pass
        m = y_true.shape[0]
        dz2 = (2.0 / m) * (z2 - y_true)

        dW2 = np.outer(dz2, a1)
        db2 = dz2

        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0).astype(float)

        dW1 = np.outer(dz1, x)
        db1 = dz1

        # Remove negative zero after rounding
        def clean(arr):
            arr = np.round(arr, 4)
            arr[np.isclose(arr, 0.0)] = 0.0
            return arr.tolist()

        loss = round(float(loss), 4)

        return {
            "loss": loss,
            "dW1": clean(dW1),
            "db1": clean(db1),
            "dW2": clean(dW2),
            "db2": clean(db2),
        }