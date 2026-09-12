import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self, X, y):
        """Calculates the line of best fit for the training data."""
        # Calculate the means of X and y
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate the slope (m)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        self.m = numerator / denominator
        
        # Calculate the intercept (b)
        self.b = y_mean - (self.m * x_mean)
        
    def predict(self, X):
        """Returns predictions for a given set of X values."""
        return self.m * X + self.b

# --- Usage Example ---

# 1. Create some dummy data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3.5, 4, 4.5, 6])

# 2. Initialize and train the model
model = SimpleLinearRegression()
model.fit(X, y)

print(f"Slope (m): {model.m:.2f}")
print(f"Intercept (b): {model.b:.2f}")

# 3. Make predictions
predictions = model.predict(X)

# 4. Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, predictions, color='red', label='Line of Best Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()