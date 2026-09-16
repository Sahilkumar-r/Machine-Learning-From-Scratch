import numpy as np

class MultipleLinearRegression:
    """
    Multiple Linear Regression optimized via the Normal Equation (Closed-form solution).
    Calculates the exact optimal weights mathematically without iteration.
    """
    
    def __init__(self):
        # We will store the bias and feature weights separately for user readability,
        # but keep the combined _theta vector for fast matrix predictions.
        self.weights = None
        self.bias = None
        self._theta = None 

    def fit(self, X, y):
        """
        Fits the model to the training data using the Normal Equation.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        # 1. Add a column of 1s to the X matrix to act as the multiplier for the bias
        n_samples = X.shape[0]
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        # 2. Apply the Normal Equation: theta = (X^T * X)^-1 * X^T * y
        # Note: We use np.linalg.pinv (pseudo-inverse) instead of np.linalg.inv.
        # This is a safety measure in case X^T * X is a singular (non-invertible) 
        # matrix, which happens if features are perfectly correlated.
        self._theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        # 3. Extract the bias (first element) and the feature weights (the rest)
        self.bias = self._theta[0]
        self.weights = self._theta[1:]
        
        return self

    def predict(self, X):
        """
        Predicts target values for new data using the calculated weights.
        """
        # Add the column of 1s to the new data
        n_samples = X.shape[0]
        X_b = np.c_[np.ones((n_samples, 1)), X]
        
        # Perform matrix multiplication: y_pred = X_b * theta
        return X_b.dot(self._theta)

    def evaluate(self, X, y):
        """Calculates the Mean Squared Error (MSE)."""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)