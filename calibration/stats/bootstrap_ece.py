import numpy as np

from calibration.sample import ResampleStats
from ..stats import ECE

from calibration.utils import distances


def bootstrap_ece(probs, y, n=1000, distance=distances.tvdistance,
                  binning=None):
    """
    Evaluate the estimator of the ECE (expected calibration error) and
    estimate the standard deviation of its sampling distribution with `n`
    bootstrap samples for probabilities `probs` with corresponding targets
    `y` with respect to the binning scheme `binning` and the distance measure
    `distance`.

    Probabilities `probs` should be an array of shape `(N, C)`, where `N` is
    the batch size and `C` is the number of targets. Targets `y` should be a
    an array of shape `(N, C)` with one-hot encoded rows.

    If `binning` is `None` (the default), the default binning scheme of
    function `ece` is used.
    """
    # define ECE statistic
    ece = ECE(distance, binning)

    # evaluate ECE of original data set
    orig_ece = ece(probs, y)

    # compute estimate of the standard deviation of ECE by bootstrapping
    resample = ResampleStats(ece, n)
    bootstrap_ece_std = np.std(resample(probs, y))

    return orig_ece, bootstrap_ece_std


class BootstrapECE:

    def __init__(self, n=1000, distance=distances.tvdistance, binning=None):
        self.n = n
        self.distance = distance
        self.binning = binning

    def __call__(self, probs, y):
        return bootstrap_ece(probs, y, n=self.n, distance=self.distance,
                             binning=self.binning)

    def __repr__(self):
        return "BootstrapECE(n=%r, distance=%r, binning=%r)" % (
            self.n, self.distance, self.binning)
