import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3, task='classification', weights='uniform', p=2):
        """
        k: Number of neighbors to use.
        task: 'classification' or 'regression'.
        weights: 'uniform' (all neighbors equal) or 'distance' (closer neighbors weighted higher).
        p: Power parameter for the Minkowski metric (1 = Manhattan, 2 = Euclidean).
        """
        self.k = k
        self.task = task
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        """
        KNN 'training' is just memorizing the dataset. No gradients or learning rates involved.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        # Calculate Minkowski distances between the input x and all training points
        distances = np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1 / self.p)

        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        k_nearest_distances = distances[k_indices]

        if self.task == 'classification':
            if self.weights == 'distance':
                # Weight votes by inverse distance (add epsilon to avoid division by zero)
                epsilon = 1e-9
                weights = 1.0 / (k_nearest_distances + epsilon)
                classes = np.unique(self.y_train)
                weighted_votes = {c: 0.0 for c in classes}
                
                for i, label in enumerate(k_nearest_labels):
                    weighted_votes[label] += weights[i]
                return max(weighted_votes, key=weighted_votes.get)
            else:
                # Standard majority vote
                most_common = Counter(k_nearest_labels).most_common(1)
                return most_common[0][0]

        elif self.task == 'regression':
            if self.weights == 'distance':
                # Weighted average of neighbor values
                epsilon = 1e-9
                weights = 1.0 / (k_nearest_distances + epsilon)
                return np.sum(k_nearest_labels * weights) / np.sum(weights)
            else:
                # Standard mean average
                return np.mean(k_nearest_labels)

# --- Example Usage ---
if __name__ == "__main__":
    # Dummy data
    X_train = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]
    y_train = [0, 0, 1, 1, 0, 1]
    X_test = [[1, 1], [8, 9]]

    # Initialize and 'train' (store data)
    clf = KNearestNeighbors(k=3, task='classification', weights='distance', p=2)
    clf.fit(X_train, y_train)

    # Predict
    predictions = clf.predict(X_test)
    print(f"Predictions: {predictions}")