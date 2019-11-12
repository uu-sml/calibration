import numpy as np

from calibration.sample import ResampleStats
from ..stats import ece

from calibration.utils import distances


def consistency_ece(probs, n=1000, distance=distances.tvdistance,
                    binning=None):
    """
    Estimate the mean and standard deviation of the estimator of the ECE
    (expected calibration error) with respect to the binning scheme `binning`
    and the distance measure `distance` under the assumption that the model
    with predicted probabilities `prob` is calibrated by consistency resampling
    of `n` samples.

    Probabilities `probs` should be an array of shape `(N, C)`, where `N` is
    the number of data points and `C` is the number of targets.

    If `binning` is `None` (the default), the default binning scheme of
    function `ece` is used.
    """
    # define ece resampling
    resample = ResampleStats(lambda x: ece(x, None, distance, binning), n)

    # generate samples with consistency resampling
    consistency_ece_samples = resample(probs)

    # compute mean and standard deviation of the empirical distribution
    consistency_ece_mean = np.mean(consistency_ece_samples)
    consistency_ece_std = np.std(consistency_ece_samples)

    return consistency_ece_mean, consistency_ece_std


class ConsistencyECE:

    def __init__(self, n=1000, distance=distances.tvdistance, binning=None):
        self.n = n
        self.distance = distance
        self.binning = binning

    def __call__(self, probs):
        return consistency_ece(probs, n=self.n, distance=self.distance,
                               binning=self.binning)

    def __repr__(self):
        return "ConsistencyECE(n=%r, distance=%r, binning=%r)" % (
            self.n, self.distance, self.binning)
