import numpy as np

class GradientDescentLinearRegression:
    """
    A unified Multiple Linear Regression class supporting Batch, Stochastic (SGD), 
    and Mini-Batch Gradient Descent, with L1/L2/ElasticNet regularization.
    """
    
    def __init__(self, mode='mini_batch', batch_size=32, learning_rate=0.01, epochs=1000, 
                 loss='mse', penalty=None, alpha=0.01, l1_ratio=0.5):
        """
        Args:
            mode (str): 'batch', 'sgd', or 'mini_batch'.
            batch_size (int): Size of the mini-batch (ignored if mode is 'batch' or 'sgd').
            learning_rate (float): Step size for the optimizer.
            epochs (int): Number of passes over the dataset.
            loss (str): The cost function to optimize ('mse' or 'mae').
            penalty (str): Regularization type ('l1', 'l2', 'elasticnet', or None).
            alpha (float): Regularization strength.
            l1_ratio (float): Mix ratio for Elastic Net (0 to 1).
        """
        if mode not in ['batch', 'sgd', 'mini_batch']:
            raise ValueError("Mode must be 'batch', 'sgd', or 'mini_batch'")
        if loss not in ['mse', 'mae']:
            raise ValueError("Loss must be 'mse' or 'mae'")
            
        self.mode = mode
        self.batch_size = batch_size
        self.lr = learning_rate
        self.epochs = epochs
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        self.weights = None
        self.bias = None
        self.loss_history = []  # Stores the average loss per epoch for plotting

    def _calculate_regularization_gradient(self):
        """Computes the gradient of the chosen regularization penalty."""
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
        self.loss_history = []

        # Determine actual batch size based on the selected mode
        if self.mode == 'batch':
            current_batch_size = n_samples
        elif self.mode == 'sgd':
            current_batch_size = 1
        else:
            current_batch_size = self.batch_size

        for epoch in range(self.epochs):
            # Shuffle data to ensure independent and identically distributed batches
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0

            for i in range(0, n_samples, current_batch_size):
                X_batch = X_shuffled[i:i + current_batch_size]
                y_batch = y_shuffled[i:i + current_batch_size]
                m_batch = X_batch.shape[0]

                # Forward pass
                y_pred = np.dot(X_batch, self.weights) + self.bias
                
                # Compute Cost and Gradients
                if self.loss == 'mse':
                    error_grad = y_pred - y_batch
                    epoch_loss += np.sum((y_pred - y_batch) ** 2)
                elif self.loss == 'mae':
                    error_grad = np.sign(y_pred - y_batch)
                    epoch_loss += np.sum(np.abs(y_pred - y_batch))

                # Gradient calculation (averaged over the batch)
                dw = (1 / m_batch) * np.dot(X_batch.T, error_grad) + self._calculate_regularization_gradient()
                db = (1 / m_batch) * np.sum(error_grad)

                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
            
            # Record average loss for this epoch
            self.loss_history.append(epoch_loss / n_samples)
                
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y, metric='r2'):
        """
        Evaluates the model.
        Args:
            metric (str): 'mse', 'rmse', 'mae', 'r2', or 'adjusted_r2'.
        """
        y_pred = self.predict(X)
        n_samples, n_features = X.shape
        
        mse = np.mean((y - y_pred) ** 2)
        
        if metric == 'mse':
            return mse
        elif metric == 'rmse':
            return np.sqrt(mse)
        elif metric == 'mae':
            return np.mean(np.abs(y - y_pred))
        elif metric == 'r2':
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric == 'adjusted_r2':
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            return 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        else:
            raise ValueError("Invalid metric.")