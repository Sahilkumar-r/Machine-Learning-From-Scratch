import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0  # Initialize slope to 0
        self.b = 0  # Initialize intercept to 0
        self.history = [] # To track our line's movement
        
    def fit(self, X, y):
        N = len(X)
        
        for i in range(self.iterations):
            # 1. Calculate current predictions
            y_predicted = self.m * X + self.b
            
            # 2. Calculate gradients
            dm = (-2 / N) * np.sum(X * (y - y_predicted))
            db = (-2 / N) * np.sum(y - y_predicted)
            
            # 3. Update parameters
            self.m = self.m - (self.learning_rate * dm)
            self.b = self.b - (self.learning_rate * db)
            
            # (Optional) Save progress every 100 iterations to visualize later
            if i % 100 == 0:
                self.history.append((self.m, self.b))
                
    def predict(self, X):
        return self.m * X + self.b

# --- Usage Example ---

# 1. Create some dummy data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3.5, 4, 4.5, 6])

# 2. Initialize and train the model
# Note: Learning rate tuning is crucial! If it's too high, the model diverges.
model = LinearRegressionGD(learning_rate=0.01, iterations=1000)
model.fit(X, y)

print(f"Final Slope (m): {model.m:.2f}")
print(f"Final Intercept (b): {model.b:.2f}")

# 3. Plot the final result
predictions = model.predict(X)
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, predictions, color='red', label='Fitted Line (Gradient Descent)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()