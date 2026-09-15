import numpy as np

class MiniBatchLinearRegression:
    """
    Linear Regression optimized via Mini-Batch Gradient Descent.
    Supports customizable loss functions and L1/L2/ElasticNet regularization.
    """
    
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32, 
                 loss='mse', penalty=None, alpha=0.01, l1_ratio=0.5):
        """
        Args:
            learning_rate (float): Step size for the gradient descent update.
            epochs (int): Number of passes over the training dataset.
            batch_size (int): Number of samples per gradient update.
            loss (str): The cost function to optimize ('mse' or 'mae').
            penalty (str): 'l1', 'l2', 'elasticnet', or None.
            alpha (float): Regularization strength.
            l1_ratio (float): ElasticNet mixing parameter.
        """
        if loss not in ['mse', 'mae']:
            raise ValueError("Loss must be 'mse' or 'mae'")
            
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
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
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                m_batch = X_batch.shape[0]

                y_pred = np.dot(X_batch, self.weights) + self.bias
                
                # Calculate the gradient base depending on the chosen loss function
                if self.loss == 'mse':
                    error_grad = y_pred - y_batch
                elif self.loss == 'mae':
                    error_grad = np.sign(y_pred - y_batch)

                # Calculate gradients
                dw = (1 / m_batch) * np.dot(X_batch.T, error_grad) + self._calculate_regularization_gradient()
                db = (1 / m_batch) * np.sum(error_grad)

                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
                
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y, metric='r2'):
        """
        Evaluates the model's performance using the specified metric.
        
        Args:
            X (np.ndarray): Test features.
            y (np.ndarray): True target values.
            metric (str): 'mse', 'rmse', 'mae', 'r2', or 'adjusted_r2'.
            
        Returns:
            float: The calculated evaluation score.
        """
        y_pred = self.predict(X)
        n_samples, n_features = X.shape
        
        if metric == 'mse':
            return np.mean((y - y_pred) ** 2)
            
        elif metric == 'rmse':
            return np.sqrt(np.mean((y - y_pred) ** 2))
            
        elif metric == 'mae':
            return np.mean(np.abs(y - y_pred))
            
        elif metric == 'r2':
            ss_res = np.sum((y - y_pred) ** 2)       # Residual sum of squares
            ss_tot = np.sum((y - np.mean(y)) ** 2)   # Total sum of squares
            return 1 - (ss_res / ss_tot)
            
        elif metric == 'adjusted_r2':
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            # Adjusted R2 formula
            return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
            
        else:
            raise ValueError("Metric must be 'mse', 'rmse', 'mae', 'r2', or 'adjusted_r2'")