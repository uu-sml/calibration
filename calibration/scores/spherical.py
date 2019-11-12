import numpy as np


def spherical_score(probs, y):
    """
    Evaluate the logarithmic score of the probabilities `probs` with
    corresponding targets `y`.

    Probabilities `probs` should be of shape `(N, C)`, where `N` is the batch
    size and `C` is the number of targets. Targets `y` should be of shape
    `(N,)`.
    """
    # check dimensions
    dim = probs.ndim
    if dim != 2:
        raise ValueError('Expected 2 dimensions (got {})'.format(dim))

    if probs.shape[0] != y.shape[0]:
        raise ValueError('Expected batch_size ({}) to match batch_size ({}).'
                         .format(probs.shape[0], y.shape[0]))

    # obtain dimensions
    N = probs.shape[0]

    return -np.mean(probs[np.arange(N), y] /
                    np.linalg.norm(probs, ord=2, axis=-1))


class SphericalScore:

    def __call__(self, probs, y):
        return spherical_score(probs, y)

    def __repr__(self):
        return "SphericalScore"
