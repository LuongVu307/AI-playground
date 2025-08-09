import numpy as np 

class LinearRegression:
    def __init__(self):
        ...

    def fit(self, X, y):
        """
            B: Batch (Number of data)
            F: Features (Number of features)
            T: Targets (Number of targets)
        """
        X = np.array(X) # Shape: (B, F)
        y = np.array(y) # Shape: (B, T)

        bias = np.ones((X.shape[0], 1))
        # print(bias.shape)

        X_bias = np.concatenate((bias, X), axis=-1)
        # print(X_bias)

        theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        # Shape: (F', B) @ (B, F') --> (F', F') @ (F', B) --> (F', B) @ (B, T) --> (F', T)

        self.coef_ = theta[1:] # Shape: (F, T)
        self.intercept_ = theta[0] # Shape: (1, T)

    def predict(self, X):
        X = np.array(X) # Shape: (B, F)
        return X@self.coef_ + self.intercept_ # Shape (B, F) @ (F, T) --> (B, T) 

