import numpy as np


def maximum_lens(probs, y):
    """
    Extract the most confident predictions from probabilities `probs` with
    corresponding targets `y`.

    Probabilities `probs` should be an array of shape `(N,)` or `(N, C)`,
    where `N` is the number of data points and `C` is the number of targets.
    If `probs` is a vector of shape `(N,)`, its entries are interpreted as
    the probabilities of the first target in a binary classification problem.
    Targets `y` should be an array of shape `(N,)` or `(N, C)`. If `y` is an
    array of shape `(N, C)`, each row represents a one-hot encoded target.

    In the reduced data set, for each data point the new target `0`
    corresponds to the old target of the most confident prediction, and target
    `1` to the set of all other targets.
    """
    # check dimensions of predictions
    dim = probs.ndim
    if dim == 1:
        N = probs.shape[0]
        C = 2
    elif dim == 2:
        N, C = probs.shape
    else:
        raise ValueError('Expected 1 or 2 dimensions (got {})'.format(dim))

    # check dimensions of targets
    if y.shape[0] != N:
        raise ValueError('Expected batch_size ({}) to match batch_size ({}).'
                         .format(y.shape[0], N))

    if y.ndim == 2 and y.shape[1] != C:
        raise ValueError(
            'Expected number of targets ({}) to match number of targets ({}).'
            .format(y.shape[1], N))

    if y.ndim > 2:
        raise ValueError('Expected 1 or 2 dimensions (got {})'.format(y.ndim))

    # compute most confident predictions
    if dim == 1:
        max_idxs = (probs < 0.5).astype(np.int)
        max_probs = np.maximum(probs, 1-probs)
    else:
        max_idxs = np.argmax(probs, axis=1)
        _max_probs = probs[np.arange(N), max_idxs]

        # expand one-dimensional probability vector
        max_probs = np.stack([_max_probs, 1-_max_probs], axis=1)

    # compute new targets:
    # if the most confident prediction matches the true outcome,
    # set the label to 0, otherwise to 1
    # (corresponding to the columns in the probability vector)
    if y.ndim == 1:
        max_y = (y != max_idxs).astype(y.dtype)
    else:
        _max_y = y[np.arange(N), max_idxs]
        max_y = np.stack([_max_y, 1-_max_y], axis=1)

    return max_probs, max_y


class MaximumLens:

    def __call__(self, probs, y):
        return maximum_lens(probs, y)

    def __repr__(self):
        return "MaximumLens()"
