class GaussianNaiveBayes:
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self._classes = np.unique(y)
        num_classes = len(self._classes)

        # Initialize arrays for mean, variance, and prior probabilities
        self._mean = np.zeros((num_classes, num_features))
        self._var = np.zeros((num_classes, num_features))
        self._priors = np.zeros(num_classes)

        # Calculate statistics for each class
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(num_samples)

    def _gaussian_density(self, class_idx, x):
        """Calculate the probability density for a given feature using Gaussian formula."""
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        
        # Add a tiny epsilon to variance to prevent division by zero
        epsilon = 1e-4
        
        numerator = np.exp(-((x - mean) ** 2) / (2 * (var + epsilon)))
        denominator = np.sqrt(2 * np.pi * (var + epsilon))
        return numerator / denominator

    def _predict_single_sample(self, x):
        posteriors = []

        # Calculate posterior probability for each class
        for idx, c in enumerate(self._classes):
            # Use log probabilities to prevent numerical underflow
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._gaussian_density(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        # Return the class with the highest probability
        return self._classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._predict_single_sample(x) for x in X]
        return np.array(y_pred)