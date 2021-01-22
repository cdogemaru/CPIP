import numpy as np

class Normalizor:
    def __init__(self):
        self.max = None
        self.min = None

    def fit(self, X):
        X = np.array(X)
        self.max = np.max(X, axis=0)
        self.min = np.min(X, axis=0)

    def normalize(self, X):
        assert(self.min is not None and self.max is not None)
        return (X - self.min) / (self.max - self.min)

    def denormalize(self, X):
        assert(self.min is not None and self.max is not None)
        return X * (self.max - self.min) + self.min

if __name__ == "__main__":
    normalizor = Normalizor()
    A = [
        [0,1],
        [2,100],
        [4,5]
    ]
    normalizor.fit(A)
    A_normalized = normalizor.normalize(A)
    A_denormalized = normalizor.denormalize(A_normalized)

    print("A_normalized:\n", A_normalized)
    print("A_demornalized:\n", A_denormalized)
