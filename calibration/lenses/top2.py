import numpy as np


def top2_lens(probs, y):
    """
    Extract the two most confident predictions from probabilities `probs` with
    corresponding targets `y`.

    Probabilities `probs` should be an array of shape `(N, C)`, where `N` is
    the number of data points and `C` is the number of targets. Targets `y`
    should be an array of shape `(N,)` or `(N, C)`. If `y` is an array of
    shape `(N, C)` each row represents a one-hot encoded target.

    In the reduced data set, for each data point the new target `0`
    corresponds to the old target of the most confident prediction, the new
    target `1` to the old target of the second most confident prediction,
    and target `2` to the set of all other targets.
    """
    # check dimensions of predictions
    dim = probs.ndim
    if dim != 2:
        raise ValueError('Expected 2 dimensions (got {})'.format(dim))

    # obtain batch size and number of targets
    N, C = probs.shape

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

    # compute the two most confident predictions
    top_idxs = np.argpartition(probs, -2, axis=1)[:, -1:-3:-1]
    _top_probs = probs[np.repeat(
        np.arange(N).reshape(-1, 1), 2, axis=1), top_idxs]

    # expand array
    top_probs = np.concatenate(
        [_top_probs, 1 - np.sum(_top_probs, axis=1, keepdims=True)], axis=1)

    # compute new targets:
    # if the most confident prediction matches the true outcome,
    # set the label to 0, if the second most confident prediction matches
    # the true outcome to 1, and otherwise to 2
    if y.ndim == 1:
        top_y = 2 - np.matmul(
            (y[:, np.newaxis] == top_idxs).astype(y.dtype), [2, 1])

    else:
        _top_y = y[np.repeat(np.arange(N).reshape(-1, 1), 2, axis=1), top_idxs]
        top_y = np.concatenate(
            [_top_y, 1 - np.sum(_top_y, axis=1, keepdims=True)], axis=1)

    return top_probs, top_y


class Top2Lens:

    def __call__(self, probs, y):
        return top2_lens(probs, y)

    def __repr__(self):
        return "Top2Lens()"
