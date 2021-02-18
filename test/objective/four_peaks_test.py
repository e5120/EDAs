import unittest
from unittest import TestCase, main

import numpy as np

from eda.objective import FourPeaks
from eda.utils import idx2one_hot


class TestFourPeaks(TestCase):
    def test_optimal(self):
        objective = FourPeaks(7, 2, minimize=True)
        optimum = objective.get_optimum(one_hot=True)
        evals, info = objective(optimum)
        self.assertEqual(evals.item(), objective.optimal_value)
        optimum = np.logical_not(optimum)[::-1]
        evals, _ = objective(optimum)
        self.assertEqual(evals.item(), objective.optimal_value)

    def test_evaluation(self):
        dim = 6
        th = 2
        objective = FourPeaks(dim, th, minimize=True)

        sample = idx2one_hot(np.array([1, 1, 1, 0, 0, 0]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals - objective.optimal_value) < 1e-8), True)

        sample = idx2one_hot(np.array([1, 1, 1, 1, 0, 0]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + 4) < 1e-8), True)

        sample = idx2one_hot(np.array([[1, 1, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0],
                                       [0, 0, 0, 1, 1, 1]]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + np.array([4, 1, 0])) < 1e-8), True)

    def test_assert(self):
        with self.assertRaises(AssertionError):
            dim = 6
            th = 5
            self.assertRaises(FourPeaks(dim, th))


if __name__ == "__main__":
    unittest.main()
