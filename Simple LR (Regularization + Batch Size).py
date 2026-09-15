import numpy as np

class MiniBatchLinearRegression:
    """
    Linear Regression optimized via Mini-Batch Gradient Descent.
    Supports L1 (Lasso), L2 (Ridge), and Elastic Net regularization.
    """
    
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32, penalty=None, alpha=0.01, l1_ratio=0.5):
        """
        Args:
            learning_rate (float): Step size for the gradient descent update.
            epochs (int): Number of passes over the training dataset.
            batch_size (int): Number of samples per gradient update.
            penalty (str): 'l1', 'l2', 'elasticnet', or None.
            alpha (float): Regularization strength (lambda parameter).
            l1_ratio (float): ElasticNet mixing parameter (0 <= l1_ratio <= 1).
        """
        if penalty not in [None, 'l1', 'l2', 'elasticnet']:
            raise ValueError("Penalty must be None, 'l1', 'l2', or 'elasticnet'")
            
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
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
        """Fits the model using Mini-Batch Gradient Descent."""
        n_samples, n_features = X.shape
        
        # Initialize parameters to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Shuffle the dataset at the start of each epoch
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Iterate over the data in mini-batches
            for i in range(0, n_samples, self.batch_size):
                # Slice the batch (handles the final smaller batch automatically)
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Number of actual samples in this batch
                m_batch = X_batch.shape[0]

                # 1. Forward pass (Matrix multiplication for the whole batch)
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # 2. Calculate error vector
                error = y_pred - y_batch

                # 3. Calculate gradients (Averaged over the batch)
                dw = (1 / m_batch) * np.dot(X_batch.T, error) + self._calculate_regularization_gradient()
                db = (1 / m_batch) * np.sum(error)

                # 4. Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
        return self

    def predict(self, X):
        """Predicts target values for new data."""
        return np.dot(X, self.weights) + self.bias