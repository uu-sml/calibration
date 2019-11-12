import numpy as np


def group_lens(probs, y, groups, check_groups=True):
    """
    Group probabilities `probs` and corresponding targets `y` into different
    `groups`.

    Probabilities `probs` should be an array of shape `(N, C)`, where `N` is
    the number of data points and `C` is the number of targets. Targets `y`
    should be an array of shape `(N,)` or `(N, C)`. If `y` is an array of
    shape `(N, C)` each row represents a one-hot encoded target. Groups
    `groups` should be a list of arrays that represent non-overlapping groups
    of targets.

    If `check_groups` is `True` (the default), it is checked if the
    provided `groups` cover all targets exactly.
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

    # check if provided groups cover all targets
    if check_groups:
        group_sum = sum(x.size for x in groups)
        if group_sum != C:
            raise ValueError('Expected number of targets in groups ({}) to '
                             'match total number of targets ({}).'
                             .format(group_sum, C))

        notcovered = np.setdiff1d(np.arange(C), np.concatenate(groups))
        if notcovered.size > 0:
            raise ValueError('Expected all targets to be covered by groups '
                             '({} not covered).'
                             .format(notcovered.size, C))

    # compute probabilities of different groups
    probs_groups = np.stack([np.sum(probs[:, group], axis=1)
                             for group in groups], axis=1)

    # compute outcomes in each group: if the true outcome
    # is in the kth group, set the new label to k,
    # corresponding to the columns in the
    # probability vector.
    if y.ndim == 1:
        y_groups = np.zeros_like(y)
        for i, group in enumerate(groups):
            y_groups[np.isin(y, group)] = i
    else:
        y_groups = np.stack([np.sum(y[:, group], axis=1)
                             for group in groups], axis=1)

    return probs_groups, y_groups


class GroupLens:

    def __init__(self, groups=None, nclasses=None, ngroups=None,
                 check_groups=True):
        if nclasses and ngroups:
            self.groups = np.array_split(np.arange(nclasses), ngroups)
        elif groups:
            self.groups = groups
        else:
            raise ValueError(
                'you have to provide either groups or number of classes and '
                'numbers of groups')

        self.check_groups = check_groups

    def __call__(self, probs, y):
        return group_lens(probs, y, self.groups,
                          check_groups=self.check_groups)

    def __repr__(self):
        return "GroupLens(groups=%r, check_groups=%r)" % (self.groups,
                                                          self.check_groups)
