import unittest
from unittest import TestCase, main

import numpy as np

from eda.objective import NKLandscape
from eda.utils import idx2one_hot


class TestNKLandscape(TestCase):
    def test_evaluate(self):
        np.random.seed(0)
        '''
        Lookup table
                 0          1          2           3         4           5          6          7
        0 : [0.85714286 0.28571429 0.14285714 1.         0.42857143 0.         0.71428571 0.57142857]
        1 : [0.14285714 0.85714286 0.42857143 1.         0.         0.57142857 0.28571429 0.71428571]
        2 : [0.71428571 0.28571429 0.42857143 0.57142857 1.         0.         0.85714286 0.14285714]
        3 : [0.28571429 0.71428571 0.85714286 1.         0.42857143 0.57142857 0.14285714 0.        ]
        4 : [0.57142857 0.71428571 0.14285714 0.85714286 0.42857143 0.28571429 0.         1.        ]
        Dependency table for each bit
        [[False  True False  True False]
        [ True False  True False False]
        [False False False  True  True]
        [False False  True False  True]
        [ True False  True False False]]
        '''
        objective = NKLandscape(5, k=2, seed=None, minimize=True)

        sample = idx2one_hot(np.array([0, 1, 1, 0, 0]), max_size=2)
        evals, feas = objective(sample)
        expected = np.array([np.mean([0.14285714, 0.57142857, 1, 0.85714286, 0.71428571])])
        self.assertEqual(np.all(np.abs(expected + evals) < 1e-8), True)

        sample = idx2one_hot(np.array([[0, 1, 1, 0, 0], [1, 1, 1, 1, 1]]), max_size=2)
        evals, feas = objective(sample)
        expected = np.array([np.mean([0.14285714, 0.57142857, 1, 0.85714286, 0.71428571]),
                              np.mean([0.57142857, 0.71428571, 0.14285714, 0, 1 ])])
        self.assertEqual(np.all(np.abs(expected + evals) < 1e-8), True)

    def test_assert(self):
        dim = 10
        with self.assertRaises(AssertionError):
            self.assertRaises(NKLandscape(dim, k=0))
        with self.assertRaises(AssertionError):
            self.assertRaises(NKLandscape(dim, k=dim+1))


if __name__ == "__main__":
    unittest.main()
