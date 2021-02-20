import unittest
from unittest import TestCase, main

import numpy as np

from eda.objective import DeceptiveTrap
from eda.utils import idx2one_hot


class TestDeceptiveTrap(TestCase):
    def test_optimum(self):
        objective = DeceptiveTrap(6, minimize=True)
        evals, _ = objective(objective.get_optimum(one_hot=True))
        self.assertEqual(evals.item(), objective.optimal_value)

    def test_sub_optimum(self):
        dim, k, d = 6, 3, 0.1
        objective = DeceptiveTrap(dim, k=k, d=d, minimize=True)
        sub_optimum = np.zeros(dim, dtype=np.int)
        evals, _ = objective(idx2one_hot(sub_optimum, 2))
        self.assertEqual(evals.item(), -(dim / k) * (1 - d))

    def test_evaluate(self):
        objective = DeceptiveTrap(6, minimize=True)

        sample = np.array([[[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + np.array([1.8] )) < 1e-8), True)

        sample = np.array([[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1]],
                           [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + np.array([1.7, 2.0] )) < 1e-8), True)

    def test_d(self):
        objective = DeceptiveTrap(6, d=0.2, minimize=True)

        sample = np.array([[[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + np.array([1.6] )) < 1e-8), True)

        sample = np.array([[[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]],
                           [[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + np.array([1.6, 0.8] )) < 1e-8), True)

    def test_k(self):
        objective = DeceptiveTrap(8, k=4, minimize=True)

        sample = np.array([[[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + np.array([0.7] )) < 1e-8), True)

        sample = np.array([[[1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0]],
                           [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0]]])
        evals, _ = objective(sample)
        self.assertEqual(np.all(np.abs(evals + np.array([0.7, 1.0] )) < 1e-8), True)

    def test_assert(self):
        with self.assertRaises(AssertionError):
            objective = DeceptiveTrap(7, minimize=False)
        with self.assertRaises(AssertionError):
            objective = DeceptiveTrap(6, d=0.5, minimize=False)


if __name__ == "__main__":
    unittest.main()
