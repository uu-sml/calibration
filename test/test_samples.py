import unittest

import numpy as np
import scipy.stats as ss
import calibration.sample as sample


class TestSamples(unittest.TestCase):

    def test_consistent_targets(self):
        N = 10000  # number of target samples
        batch_size = 100  # batch size

        # for probability vectors of different sizes
        for C in (2, 5):
            # generate random probabilities
            np.random.seed(1234)
            probs = np.random.dirichlet(np.ones(C), batch_size)

            # sample N consistent targets using functional implementation
            np.random.seed(1234)
            targets = np.stack([sample.consistent_targets(probs)
                                for _ in range(N)])

            # check shape of samples
            self.assertTrue(np.shape(targets) == (N, batch_size))

            # check proportions of targets
            proportions = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=C) / N,
                arr=targets, axis=0).swapaxes(0, 1)
            self.assertTrue(
                np.all(ss.chisquare(proportions, probs).pvalue > 0.99))

            # check state-full implementation
            np.random.seed(1234)
            sampler = sample.ConsistentTargets()
            targets2 = np.stack([sampler(probs) for _ in range(N)])
            self.assertTrue(np.array_equal(targets, targets2))

    def test_resample_stats(self):
        N = 10000  # number of resample steps
        batch_size = 100  # batch size

        # use identity function as statistic
        def stats(x, y):
            return x, y

        # initialize inputs
        x = np.arange(batch_size)
        y = -np.arange(batch_size)

        # bootstrap targets using functional implementation
        np.random.seed(1234)
        results_x, results_y = [np.stack(list(x))
                                for x in zip(*sample.resample_stats(
                                    stats, x, y, n=N))]

        # check shape
        self.assertTrue(results_x.shape == (N, batch_size))
        self.assertTrue(results_y.shape == (N, batch_size))

        # check values
        self.assertTrue(np.all(np.isin(results_x, x)))
        self.assertTrue(np.all(np.isin(results_y, y)))

        # check proportion of targets
        proportions_x = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=batch_size),
            arr=results_x, axis=0) / N
        proportions_y = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=batch_size),
            arr=-results_y, axis=0) / N
        self.assertTrue(np.all(ss.chisquare(proportions_x).pvalue > 0.99))
        self.assertTrue(np.all(ss.chisquare(proportions_y).pvalue > 0.99))

        # check state-full implementation
        np.random.seed(1234)
        sampler = sample.ResampleStats(stats, n=N)
        results2_x, results2_y = [np.stack(list(x))
                                  for x in zip(*sampler(x, y))]
        self.assertTrue(np.array_equal(results_x, results2_x))
        self.assertTrue(np.array_equal(results_y, results2_y))


if __name__ == '__main__':
    unittest.main()
