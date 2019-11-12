import unittest

import numpy as np
import calibration.scores as scores


class TestScores(unittest.TestCase):

    def setUp(self):
        np.random.seed(1234)
        self.y = np.random.randint(0, 5, 20)
        self.probs_best = np.eye(5)[self.y]
        self.probs_worst = np.eye(5)[(self.y + 1) % 5]

    def test_logarithmic(self):
        # functional implementation
        score_best = scores.logarithmic_score(self.probs_best, self.y)
        score_worst = scores.logarithmic_score(self.probs_worst, self.y)

        self.assertEqual(score_best, 0)
        self.assertEqual(score_worst, np.inf)

        # state-full implementation
        score = scores.LogarithmicScore()

        self.assertEqual(score_best, score(self.probs_best, self.y))
        self.assertEqual(score_worst, score(self.probs_worst, self.y))

    def test_quadratic(self):
        # functional implementation
        score_best = scores.quadratic_score(self.probs_best, self.y)
        score_worst = scores.quadratic_score(self.probs_worst, self.y)

        self.assertEqual(score_best, -1)
        self.assertEqual(score_worst, 1)

        # state-full implementation
        score = scores.QuadraticScore()

        self.assertEqual(score_best, score(self.probs_best, self.y))
        self.assertEqual(score_worst, score(self.probs_worst, self.y))

    def test_spherical(self):
        # functional implementation
        score_best = scores.spherical_score(self.probs_best, self.y)
        score_worst = scores.spherical_score(self.probs_worst, self.y)

        self.assertEqual(score_best, -1)
        self.assertEqual(score_worst, 0)

        # state-full implementation
        score = scores.SphericalScore()

        self.assertEqual(score_best, score(self.probs_best, self.y))
        self.assertEqual(score_worst, score(self.probs_worst, self.y))


if __name__ == '__main__':
    unittest.main()
