"""
Data utils
"""
import numpy as np
from copy import deepcopy


def randomize_tabular_data(X, mode='shuffle', shuffle_idx: list=None, correlated=False, n_samples=None):
    """
    Routine to randomize given tabular data X

    Args:
        X (np.ndarray): tabular data to randomize
        mode (str): how to shuffle the data. Accepted modes are:
                    'shuffle': shuffle the feature values across samples. If shuffle_idx is defined,
                                then only the features indicated are gonna be shuffled. In this mode,
                    'gaussian': for each feature, values are sampled from a gaussian distribution with parameters the
                                empirical means and variances of the features. If correlated is True, then the empirical covariance is
                                used as the covariance of the gaussian distribution. Otherwise, features are assumed
                                uncorrelated.
                    'uniform': feature values are sampled from indepedent uniform distributions with mean and variance
                                corresponding to the empirical ones.
                    'sample': for each feature, sample a random value in the dataset *along* that feature.
        correlated (bool): for mode 'gaussian', if true, assume that the features are uncorrelated
        n_samples (int): for all mode but 'shuffle', if defined, it indicates the number of samples to sample. Otherwise,
                         the same number of samples in X will be produced
    """
    ACCEPTED_MODES = {
        'shuffle', 'gaussian', 'uniform', 'sample'
    }
    if n_samples is None:
        n_samples = len(X)
    if mode == 'shuffle':
        if shuffle_idx is None:
            shuffle_idx = list(range(len(X.shape[1])))
        new_X = deepcopy(X)
        samples_idx = np.arange(n_samples)
        for idx in shuffle_idx:
            new_X[samples_idx, idx] = X[np.random.shuffle(samples_idx), idx]
        return new_X
    elif mode == 'sample':
        features = []
        for f in range(len(X.shape[1])):
            features.append(np.random.choice(X[:, f], size=(n_samples, 1)))
        return np.hstack(features)
    elif mode in ACCEPTED_MODES:
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        if not correlated or mode == 'uniform':
            cov = np.diag(np.diag(cov))
        if mode == 'gaussian':
            return np.random.multivariate_normal(mean, cov, size=len(X))
        if mode == 'uniform':
            a = mean - np.sqrt(3*cov)
            b = 2*mean - a
            return np.random.uniform(a, b, size=X.shape)
    else:
        raise ValueError('Accepted modes are: {}'.format(ACCEPTED_MODES))
