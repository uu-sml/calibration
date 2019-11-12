import numpy as np

from calibration.utils import distances
from calibration.sample import consistent_targets
from calibration.binning import BinningTree, UniformBinning


def ece(probs, y, distance=distances.tvdistance, binning=None):
    """
    Estimate ECE (expected calibration error) of probabilities `probs`
    with corresponding targets `y` with respect to binning scheme `binning`
    and distance measure `distance`.

    If the data is binned, use `ece_binned` instead of this function.

    Probabilities `probs` should be an array of shape `(N,)` or `(N, C)`,
    where `N` is the number of data points and `C` is the number of targets.
    If `probs` is a vector of shape `(N,)`, its entries are interpreted as
    the probabilities of the first target in a binary classification problem.
    Targets `y` should be `None` or an array of shape `(N, C)` with one-hot
    encoded rows. If `y` is `None`, consistent targets are sampled from
    probabilities `probs`.

    If `binning` is `None` (the default), a binning scheme with 10 bins of
    uniform size along each dimension is used.
    """
    if binning is None:
        binning = UniformBinning(bins=10)

    binning_tree = BinningTree(binning).fit(probs)
    binned_probs = binning_tree.bin_data()
    binned_y = None if y is None else binning_tree.bin_data(y)

    return ece_binned(binned_probs, binned_y, distance)


def ece_binned(binned_probs, binned_y, distance=distances.tvdistance):
    """
    Estimate ECE (expected calibration error) of binned probabilities
    `binned_probs` with corresponding targets `binned_y` with respect to
    distance measure `distance`.

    Probabilities `binned_probs` should be a list of arrays of shape `(N, C)`,
    where `N` is the batch size and `C` is the number of targets. Every element
    of the list `binned_probs` should correspond to a set of probabilties in a
    different region of the probability simplex. Targets `binned_y` should be
    `None` or a corresponding list of the same length as `binned_probs` with
    arrays of shape `(N, C)` with one-hot encoded rows. If `binned_y` is
    `None`, consistent targets are sampled from probabilities `binned_probs`.
    """
    # create consistent targets
    if binned_y is None:
        nclasses = binned_probs[0].shape[-1]
        onehot = np.eye(nclasses)
        binned_y = [onehot[consistent_targets(x)] for x in binned_probs]

    # obtain proportion of different bins
    proportions = np.array([x.shape[0] for x in binned_y])
    proportions = proportions / np.sum(proportions)

    # sum distances of average predictions to outcomes in each bin,
    # weighted by the proportion of predictions
    return np.dot(proportions,
                  np.array([distance(x.mean(axis=0),
                                     y.mean(axis=0))
                            for x, y in zip(binned_probs, binned_y)]))


class ECE:

    def __init__(self, distance=distances.tvdistance, binning=None):
        self.distance = distance
        self.binning = binning

    def __call__(self, probs, y):
        return ece(probs, y, distance=self.distance, binning=self.binning)

    def __repr__(self):
        return "ECE(distance=%r)" % self.distance
