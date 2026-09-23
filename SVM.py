import numpy as np

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # SVM requires labels to be strictly -1 or 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Stochastic Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Check if the point satisfies the margin condition
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Point is outside the margin; update weights by regularization only
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Point is inside the margin or misclassified; update weights and bias
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Calculate the linear approximation
        approx = np.dot(X, self.w) - self.b
        
        # Return the sign (-1 or 1) as the class prediction
        return np.sign(approx)