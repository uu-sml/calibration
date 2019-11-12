import unittest

import numpy as np
import calibration.stats as stats
import calibration.binning as binning


class TestStats(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.probs = np.random.dirichlet(np.ones(5), 20)
        self.y = np.eye(self.probs.shape[1])[np.random.randint(0, 5, 20)]

    def test_ece(self):
        # use binning scheme with only one bin
        binning_scheme = binning.UniformBinning(bins=1)

        # standard estimates
        ece_binned_standard = stats.ece_binned([self.probs], [self.y])
        self.assertEqual(ece_binned_standard,
                         np.linalg.norm(
                             self.probs.mean(axis=0) - self.y.mean(axis=0),
                             ord=1) / 2)

        ece_standard = stats.ece(self.probs, self.y, binning=binning_scheme)
        self.assertEqual(ece_standard, ece_binned_standard)

        # estimates with consistent targets
        np.random.seed(1234)
        ece_binned_consistent = stats.ece_binned([self.probs], None)
        self.assertLess(ece_binned_consistent, ece_binned_standard)

        np.random.seed(1234)
        ece_consistent = stats.ece(self.probs, None, binning=binning_scheme)
        self.assertEqual(ece_consistent, ece_binned_consistent)

    def test_bootstrap_ece(self):
        # for different numbers of bins
        for nbins in [1, 5]:
            # define binning scheme
            binning_scheme = binning.UniformBinning(bins=nbins)

            # estimate ECE
            orig = stats.ece(self.probs, self.y, binning=binning_scheme)

            # estimate sampling distribution of ECE by bootstrapping
            bootstrap_orig, bootstrap_std = stats.bootstrap_ece(
                self.probs, self.y, binning=binning_scheme)
            self.assertEqual(bootstrap_orig, orig)
            self.assertLess(bootstrap_std, bootstrap_orig)

    def test_consistency_ece(self):
        # for different numbers of bins
        for nbins in [1, 5]:
            # define binning scheme
            binning_scheme = binning.UniformBinning(bins=nbins)

            # estimate ECE
            orig = stats.ece(self.probs, self.y, binning=binning_scheme)

            # estimate sampling distribution of ECE under the assumption
            # that the model is calibrated by consistency resampling
            consistency_mean, consistency_std = stats.consistency_ece(
                self.probs, binning=binning_scheme)
            self.assertLess(consistency_mean, orig)
            self.assertLess(consistency_std, consistency_mean)


if __name__ == '__main__':
    unittest.main()
