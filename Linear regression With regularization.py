import numpy as np

class SGDLinearRegression:
    """
    Linear Regression optimized via Stochastic Gradient Descent (SGD).
    Supports L1 (Lasso), L2 (Ridge), and Elastic Net regularization.
    """
    
    def __init__(self, learning_rate=0.01, epochs=1000, penalty=None, alpha=0.01, l1_ratio=0.5):
        """
        Args:
            learning_rate (float): Step size for the gradient descent update.
            epochs (int): Number of passes over the training dataset.
            penalty (str): 'l1', 'l2', 'elasticnet', or None.
            alpha (float): Regularization strength (lambda parameter).
            l1_ratio (float): ElasticNet mixing parameter (0 <= l1_ratio <= 1). 
                              1 is pure L1, 0 is pure L2.
        """
        if penalty not in [None, 'l1', 'l2', 'elasticnet']:
            raise ValueError("Penalty must be None, 'l1', 'l2', or 'elasticnet'")
            
        self.lr = learning_rate
        self.epochs = epochs
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        # Model parameters
        self.weights = None
        self.bias = None

    def _calculate_regularization_gradient(self):
        """Computes the gradient of the regularization term."""
        if self.penalty == 'l2':
            return self.alpha * self.weights
        elif self.penalty == 'l1':
            return self.alpha * np.sign(self.weights)
        elif self.penalty == 'elasticnet':
            l1_grad = self.l1_ratio * np.sign(self.weights)
            l2_grad = (1 - self.l1_ratio) * self.weights
            return self.alpha * (l1_grad + l2_grad)
        return 0.0

    def fit(self, X, y):
        """
        Fits the model to the training data.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Shuffle the dataset at the start of each epoch for true SGD
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                # 1. Forward pass (Calculate prediction)
                y_pred = np.dot(xi, self.weights) + self.bias

                # 2. Calculate error
                error = y_pred - yi

                # 3. Calculate gradients for a single sample
                dw = error * xi + self._calculate_regularization_gradient()
                db = error  # Bias is not typically regularized

                # 4. Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
        return self

    def predict(self, X):
        """Predicts target values for new data."""
        return np.dot(X, self.weights) + self.bias