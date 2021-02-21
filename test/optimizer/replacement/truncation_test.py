import unittest
from unittest import TestCase, main

import numpy as np
from numpy.core.defchararray import replace

from eda.optimizer.replacement import Truncation


class TestTruncation(TestCase):
    def test_replacement(self):
        rate = 0.2
        replacement = Truncation(rate, fix_size=True)

        lam = 10
        parent = np.arange(lam)
        p_evals = np.arange(lam)[::-1]
        candidate = np.arange(lam+1, lam+1+round(lam*rate))
        c_evals = np.arange(lam+1, lam+1+round(lam*rate))[::-1]

        p, p_e = replacement(parent, p_evals, candidate, c_evals)
        self.assertListEqual(list(p[:round(lam*(1-rate))]), list(parent[round(lam*rate):][::-1]))
        self.assertListEqual(list(p[round(lam*(1-rate)):]), list(candidate))
        self.assertListEqual(list(p_e[:round(lam*(1-rate))]), list(p_evals[round(lam*rate):][::-1]))
        self.assertListEqual(list(p_e[round(lam*(1-rate)):]), list(c_evals))

        rate = 0.9
        replacement = Truncation(rate, fix_size=True)

        lam = 10
        parent = np.arange(lam)
        p_evals = np.arange(lam)[::-1]
        candidate = np.arange(lam+1, lam+1+round(lam*rate))
        c_evals = np.arange(lam+1, lam+1+round(lam*rate))[::-1]

        p, p_e = replacement(parent, p_evals, candidate, c_evals)
        self.assertListEqual(list(p[:round(lam*(1-rate))]), list(parent[round(lam*rate):][::-1]))
        self.assertListEqual(list(p[round(lam*(1-rate)):]), list(candidate))
        self.assertListEqual(list(p_e[:round(lam*(1-rate))]), list(p_evals[round(lam*rate):][::-1]))
        self.assertListEqual(list(p_e[round(lam*(1-rate)):]), list(c_evals))

        rate = 0.54
        replacement = Truncation(rate, fix_size=True)

        lam = 10
        parent = np.arange(lam)
        p_evals = np.arange(lam)[::-1]
        candidate = np.arange(lam+1, lam+1+round(lam*rate))
        c_evals = np.arange(lam+1, lam+1+round(lam*rate))[::-1]

        p, p_e = replacement(parent, p_evals, candidate, c_evals)
        self.assertListEqual(list(p[:round(lam*(1-rate))]), list(parent[round(lam*rate):][::-1]))
        self.assertListEqual(list(p[round(lam*(1-rate)):]), list(candidate))
        self.assertListEqual(list(p_e[:round(lam*(1-rate))]), list(p_evals[round(lam*rate):][::-1]))
        self.assertListEqual(list(p_e[round(lam*(1-rate)):]), list(c_evals))

    def test_fix_size(self):
        rate = 0.3
        replacement = Truncation(rate, fix_size=False)

        lam = 10
        parent = np.arange(lam)
        p_evals = np.arange(lam)[::-1]
        candidate = np.arange(lam, lam+5)
        c_evals = np.arange(lam, lam+5)[::-1]

        p, p_e = replacement(parent, p_evals, candidate, c_evals)
        self.assertListEqual(list(p[:round(lam*(1-rate))]), list(parent[round(lam*rate):][::-1]))
        self.assertListEqual(list(p[round(lam*(1-rate)):]), list(candidate))
        self.assertListEqual(list(p_e[:round(lam*(1-rate))]), list(p_evals[round(lam*rate):][::-1]))
        self.assertListEqual(list(p_e[round(lam*(1-rate)):]), list(c_evals))

    def test_assert(self):
        with self.assertRaises(AssertionError):
            Truncation(replace_rate=-0.1)
        with self.assertRaises(AssertionError):
            Truncation(replace_rate=1.1)


if __name__ == "__main__":
    unittest.main()
