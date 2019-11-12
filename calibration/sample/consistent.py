import numpy as np


def consistent_targets(probs):
    """
    Sample consistent targets with probabilities `probs`.

    Probabilities have to be specified in shape `(N, C)`, where `N` is the
    batch size and `C` is the number of targets. Targets are sampled from
    `(0, ..., C-1)` and returned in shape `(N)`.
    """
    dim = probs.ndim
    if dim != 2:
        raise ValueError('Expected 2 dimensions (got {})'.format(dim))

    N, C = probs.shape
    if C < 2:
        raise ValueError('Expected 2 or more classes (got {})'.format(C))

    if C == 2:
        return (np.random.uniform(size=N) < probs[:, 1]).astype(int)

    return (np.random.uniform(size=(N, 1)) < np.cumsum(probs, axis=1)).argmax(
        axis=1)


class ConsistentTargets:

    def __call__(self, probs):
        return consistent_targets(probs)

    def __repr__(self):
        return "ConsistentTargets()"
