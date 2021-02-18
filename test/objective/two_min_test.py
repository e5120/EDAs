import unittest
from unittest import TestCase, main

import numpy as np

from eda.objective import TwoMin
from eda.utils import idx2one_hot


class TestFourPeaks(TestCase):
    def test_optimal(self):
        dim = 6
        objective = TwoMin(dim, minimize=True)
        optimum = objective.get_optimum(one_hot=True)
        evals, _ = objective(optimum)
        self.assertEqual(evals.item(), objective.optimal_value)
        optimum = np.logical_not(optimum)
        evals, _ = objective(optimum)
        self.assertEqual(evals.item(), objective.optimal_value)

    def test_evaluate(self):
        np.random.seed(1)
        dim = 6
        objective = TwoMin(dim, minimize=True)
        # target = [0, 1, 0, 0, 0, 0]

        sample = idx2one_hot(np.array([0, 1, 0, 0, 0, 0]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals - objective.optimal_value) < 1e-8), True)

        sample = idx2one_hot(np.array([1, 1, 1, 1, 0, 0]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals - 3) < 1e-8), True)

        sample = idx2one_hot(np.array([[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]]), 2)
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals - np.array([1, 2, 2])) < 1e-8), True)


if __name__ == "__main__":
    unittest.main()
