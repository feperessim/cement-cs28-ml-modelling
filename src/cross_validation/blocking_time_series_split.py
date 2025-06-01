import numpy as np


class BlockingTimeSeriesSplit:
    def __init__(self, n_splits, train_size, margin=0):
        self.n_splits = n_splits
        self.train_size = train_size
        self.margin = margin

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(self.train_size * (stop - start)) + start
            yield indices[start: mid], indices[mid + self.margin: stop]


# Reference: https://goldinlocks.github.io/Time-Series-Cross-Validation/
