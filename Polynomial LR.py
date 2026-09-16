import numpy as np
from itertools import combinations_with_replacement

class PolynomialFeatures:
    """
    Generates a new feature matrix consisting of all polynomial combinations 
    of the features with degree less than or equal to the specified degree.
    """
    def __init__(self, degree=2):
        self.degree = degree

    def transform(self, X):
        """
        Transforms the input matrix X to polynomial features.
        
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Transformed polynomial feature matrix.
        """
        n_samples, n_features = X.shape
        poly_features = []
        
        # Iterate through degrees from 1 up to the target degree
        for d in range(1, self.degree + 1):
            # Generate all combinations of feature indices for this degree
            # e.g., for degree 2 and features [0, 1], combos are: (0,0), (0,1), (1,1)
            for combo in combinations_with_replacement(range(n_features), d):
                # Multiply the chosen columns together row-wise
                new_feature = np.prod(X[:, combo], axis=1)
                poly_features.append(new_feature)
                
        # Stack the generated 1D feature arrays as columns in a new 2D matrix
        return np.column_stack(poly_features)