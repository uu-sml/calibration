import unittest

import numpy as np
import calibration.lenses as lenses


class TestLenses(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.probs = np.random.dirichlet(np.ones(10), 10000)
        self.y = np.random.randint(0, 10, 10000)
        self.y_onehot = np.eye(10)[self.y]

    def test_group(self):
        groups = np.array_split(np.arange(10), 3)

        # functional implementation
        groups_probs, groups_y = lenses.group_lens(self.probs, self.y, groups)

        # check that each row is a probability vector
        self.assertTrue(np.allclose(np.sum(groups_probs, axis=1), 1))

        # check that the new targets correspond to the provided groups of the
        # old targets
        for i, g in enumerate(groups):
            self.assertTrue(
                np.array_equal(groups_probs[:, i],
                               np.sum(self.probs[:, g], axis=-1)))
            self.assertTrue(np.array_equal(groups_y == i,
                                           np.isin(self.y, g)))

        # check that one-hot encoded targets yield same results
        groups_probs_onehot, groups_y_onehot = lenses.group_lens(
            self.probs, self.y_onehot, groups)

        self.assertTrue(np.array_equal(groups_probs_onehot, groups_probs))
        self.assertTrue(np.array_equal(groups_y_onehot, np.eye(3)[groups_y]))

        # state-full implementation
        lens = lenses.GroupLens(groups=groups)
        groups_probs2, groups_y2 = lens(self.probs, self.y)

        self.assertTrue(np.array_equal(groups_probs2, groups_probs))
        self.assertTrue(np.array_equal(groups_y2, groups_y))

        lens = lenses.GroupLens(nclasses=10, ngroups=3)
        groups_probs3, groups_y3 = lens(self.probs, self.y)

        self.assertTrue(np.array_equal(groups_probs3, groups_probs))
        self.assertTrue(np.array_equal(groups_y3, groups_y))

    def test_maximum(self):
        # functional implementation
        max_probs, max_y = lenses.maximum_lens(self.probs, self.y)

        # check that each row is a probability vector
        self.assertTrue(np.allclose(np.sum(max_probs, axis=1), 1))

        # check that the new target `0` corresponds to the old target of
        # the most confident prediction
        self.assertTrue(
            np.array_equal(max_probs[:, 0], np.max(self.probs, axis=1)))
        self.assertTrue(np.array_equal(max_y == 0,
                                       np.argmax(self.probs, axis=1) ==
                                       self.y))

        # check that one-hot encoded targets yield same results
        max_probs_onehot, max_y_onehot = lenses.maximum_lens(
            self.probs, self.y_onehot)

        self.assertTrue(np.array_equal(max_probs_onehot, max_probs))
        self.assertTrue(np.array_equal(max_y_onehot, np.eye(2)[max_y]))

        # check that predictions can be provided as a vector
        probs0 = self.probs[:, 0]
        probs0_full = np.stack([probs0, 1-probs0], axis=1)
        y0 = (self.y != 0).astype(self.y.dtype)

        max_probs0, max_y0 = lenses.maximum_lens(probs0, y0)
        max_probs0_full, max_y0_full = lenses.maximum_lens(probs0_full, y0)

        self.assertTrue(np.array_equal(max_probs0_full[:, 0], max_probs0))
        self.assertTrue(np.array_equal(max_y0_full, max_y0))

        # state-full implementation
        lens = lenses.MaximumLens()
        max_probs2, max_y2 = lens(self.probs, self.y)

        self.assertTrue(np.array_equal(max_probs, max_probs2))
        self.assertTrue(np.array_equal(max_y, max_y2))

    def test_top2(self):
        # functional implementation
        top_probs, top_y = lenses.top2_lens(self.probs, self.y)

        # check that each row is a probability vector
        self.assertTrue(np.allclose(np.sum(top_probs, axis=1), 1))

        # check that the new target `0` corresponds to the old target of
        # the most confident prediction
        self.assertTrue(
            np.array_equal(top_probs[:, 0],
                           np.max(self.probs, axis=1)))
        self.assertTrue(np.array_equal(top_y == 0,
                                       np.argmax(self.probs,
                                                 axis=1) == self.y))

        # check that one-hot encoded targets yield same results
        top_probs_onehot, top_y_onehot = lenses.top2_lens(
            self.probs, self.y_onehot)

        self.assertTrue(np.array_equal(top_probs_onehot, top_probs))
        self.assertTrue(np.array_equal(top_y_onehot, np.eye(3)[top_y]))

        # state-full implementation
        lens = lenses.Top2Lens()
        top_probs2, top_y2 = lens(self.probs, self.y)

        self.assertTrue(np.array_equal(top_probs2, top_probs))
        self.assertTrue(np.array_equal(top_y2, top_y))


if __name__ == '__main__':
    unittest.main()
