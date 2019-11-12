import unittest

import numpy as np
import calibration.binning as binning


class TestBinning(unittest.TestCase):

    def test_uniform(self):
        # example data set
        probs = np.array([0.0, 0.5, 0.2, 0.7, 1.0, 0.95])
        y = np.eye(2)[[0, 1, 1, 0, 1, 1]]

        # tree with 5 bins along each dimension
        tree = binning.BinningTree(binning.UniformBinning(5))

        # indices of data points in each bin
        bins = [np.array([0]), np.array([2]), np.array([1]),
                np.array([3]), np.array([4, 5])]

        # index of bin for each data point
        binnumbers = np.array([0, 2, 1, 3, 4, 4])

        # one-dimensional probabilities
        tree.fit(probs)

        # check indices of data points in each bin
        for a, b in zip(tree.bins, bins):
            self.assertTrue(np.all(np.equal(a, b)))

        # check index of bin for each data point
        self.assertTrue(np.all(np.equal(tree.binnumbers, binnumbers)))

        # check binning of probabilities
        probs = np.stack([probs, 1-probs], axis=-1)
        binned_probs = [probs[x] for x in bins]
        for a, b in zip(tree.bin_data(), binned_probs):
            self.assertTrue(np.all(np.equal(a, b)))

        # check binning of targets
        binned_y = [y[x] for x in bins]
        for a, b in zip(tree.bin_data(y), binned_y):
            self.assertTrue(np.all(np.equal(a, b)))

        # two-dimensional probabilities
        tree.fit(probs)

        # check indices of data points in each bin
        for a, b in zip(tree.bins, bins):
            self.assertTrue(np.all(np.equal(a, b)))

        # check index of bin for each data point
        self.assertTrue(np.all(np.equal(tree.binnumbers, binnumbers)))

        # check binning of probabilities
        for a, b in zip(tree.bin_data(), binned_probs):
            self.assertTrue(np.all(np.equal(a, b)))

        # check binning of targets
        for a, b in zip(tree.bin_data(y), binned_y):
            self.assertTrue(np.all(np.equal(a, b)))

    def test_dependent(self):
        # example data set
        np.random.seed(1234)
        probs1D = np.random.rand(10000)
        probs2D = np.stack([probs1D, 1 - probs1D], axis=-1)

        # data dependent binning tree with at most 100 samples per bin
        tree = binning.BinningTree(binning.DataDependentBinning(
            min_size=100, threshold="mean"))

        # one-dimensional probabilities
        tree.fit(probs1D)

        # check indices of data points in each bin
        bins = tree.bins
        self.assertEqual(sum(b.size for b in bins), 10000)
        for b in bins:
            self.assertTrue(b.size <= 100)

        # two-dimensional probabilities
        tree.fit(probs2D)

        # check indices of data points in each bin
        for a, b in zip(tree.bins, bins):
            self.assertTrue(np.all(np.equal(a, b)))


if __name__ == '__main__':
    unittest.main()
