import unittest
from unittest import TestCase, main

import numpy as np

from eda.optimizer.selection import Top


class TestTop(TestCase):
    def test_selection(self):
        lam = 10
        population = np.arange(lam)
        evals = np.arange(lam)[::-1]

        selection = Top(0.5)
        p, e = selection(population, evals, sort=True)
        self.assertEqual(np.all(population[5:][::-1] == p), True)
        self.assertEqual(np.all(evals[5:][::-1] == e), True)

        p, e = selection(population, evals, sort=True)
        self.assertEqual(np.all(population[5:][::-1] == p), True)
        self.assertEqual(np.all(evals[5:][::-1] == e), True)

    def test_assert(self):
        with self.assertRaises(AssertionError):
            self.assertRaises(Top(-1))
        with self.assertRaises(AssertionError):
            self.assertRaises(Top(1.1))


if __name__ == "__main__":
    unittest.main()
