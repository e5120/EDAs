import unittest
from unittest import TestCase, main

import numpy as np

from eda.optimizer.replacement import RestrictedTournament


class TestRestrictedTournament(TestCase):
    def test_replacement(self):
        dim = 5
        ws = 2
        rate = 0.4
        replacement = RestrictedTournament(dim, replace_rate=rate, window_size=ws, fix_size=True)

        parent = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1],
                           [1, 0, 0, 0, 0]])
        p_evals = np.array([0, -2, -1])
        candidate = np.array([[0, 0, 1, 1, 1]])
        c_evals = np.array([-3])

        for _ in range(1000):
            p, p_e = replacement(parent, p_evals, candidate, c_evals)
            self.assertEqual(np.all(p[0] == candidate[0]) or np.all(p[1] == candidate[0]), True)
            self.assertEqual(p_e[0] == c_evals[0] or p_e[1] == c_evals[0], True)

        c_evals = np.array([0])
        p, p_e = replacement(parent, p_evals, candidate, c_evals)
        self.assertEqual(np.all(p == parent), True)
        self.assertEqual(np.all(p_e == p_evals), True)

    def test_assert(self):
        with self.assertRaises(AssertionError):
            RestrictedTournament(-1)


if __name__ == "__main__":
    unittest.main()
