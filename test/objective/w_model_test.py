import unittest
from unittest import TestCase, main

import numpy as np

from eda.objective import WModel
from eda.utils import idx2one_hot


class TestWModel(TestCase):
    def test_neutrality(self):
        print("Neutrality")
        lam = 3
        dim = 10
        mu = 2
        objective = WModel(dim, mu=mu, minimize=True)

        sample = np.random.randint(0, 2, size=(lam, dim))
        transformed_sample = objective.neutrality_layer(sample)
        print(sample)
        print(transformed_sample)

        lam = 3
        dim = 9
        mu = 4
        objective = WModel(dim, mu=mu, minimize=True)

        sample = np.random.randint(0, 2, size=(lam, dim))
        transformed_sample = objective.neutrality_layer(sample)
        print(sample)
        print(transformed_sample)

        lam = 3
        dim = 8
        mu = 3
        objective = WModel(dim, mu=mu, minimize=True)

        sample = np.random.randint(0, 2, size=(lam, dim))
        transformed_sample = objective.neutrality_layer(sample)
        print(sample)
        print(transformed_sample)

    def test_epistasis(self):
        print("Epistasis")
        lam = 3
        dim = 10
        v = 4
        objective = WModel(dim, v=v, minimize=True)

        sample = np.random.randint(0, 2, size=(lam, dim))
        print(sample)
        transformed_sample = objective.epistasis_layer(sample)
        print(transformed_sample)

    def test_multi_objective_layer(self):
        print("Multi-objectivity")
        lam = 3
        dim = 10
        m = 2
        n = 5
        objective = WModel(dim, m=m, n=n, minimize=True)

        sample = np.random.randint(0, 2, size=(lam, dim))
        print(sample)
        transformed_sample = objective.multi_objective_layer(sample)
        print(transformed_sample)

        lam = 3
        dim = 10
        m = 3
        n = 5
        objective = WModel(dim, m=m, n=n, minimize=True)

        sample = np.random.randint(0, 2, size=(lam, dim))
        print(sample)
        transformed_sample = objective.multi_objective_layer(sample)
        print(transformed_sample)

    def test_evaluation_layer(self):
        print("Evaluation")
        lam = 3
        dim = 10
        m = 2
        n = 5
        objective = WModel(dim, m=m, n=n, minimize=True)

        sample = np.random.randint(0, 2, size=(lam, m, n))
        print(sample)
        transformed_sample = objective.evaluation_layer(sample)
        print(transformed_sample)

    def test_ruggedness_and_deceptiveness_layer(self):
        print("ruggedness and deceptivenss")
        lam = 3
        dim = 10
        m = 2
        n = 6
        gamma = 12
        objective = WModel(dim, m=m, n=n, gamma=gamma, minimize=True)

        evals_sample = np.array([[3, 6]])
        evals = objective.ruggedness_and_deceptiveness_layer(evals_sample)
        print(evals)
        print(objective.gamma_prime)
        print(objective.perm)

    def test_evaluation(self):
        print("Evaluation method")
        lam = 3
        dim = 10
        mu = 2
        v = 4
        m = 1
        n = 5
        gamma = 3
        objective = WModel(dim, mu=mu, v=v, m=m, n=n, gamma=gamma, minimize=True)

        sample = idx2one_hot(np.random.randint(0, 2, size=(lam, dim)), 2)
        print(np.argmax(sample, axis=-1))
        evals, _ = objective(sample)
        print(evals)
        print(objective.perm)


if __name__ == "__main__":
    unittest.main()
