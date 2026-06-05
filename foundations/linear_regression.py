import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        Y= X @ weights
        return np.round(Y,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        
        core= (ground_truth-model_prediction) ** 2

        MSE= np.sum(core)/ len(model_prediction)

        return np.round(MSE,5)
