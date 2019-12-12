import numpy as np

from optimizer.crossover.crossover_base import CrossoverBase


class Uniform(CrossoverBase):
    def __init__(self):
        pass

    def apply(self, parent1, parent2, fitness1, fitness2):
        rand_value = np.tile(np.random.rand(parent1.shape[0])[:, np.newaxis], (1, 2))
        child1 = np.where(rand_value <= 0.5, parent1, parent2)
        child2 = np.where(rand_value > 0.5, parent1, parent2)
        return child1, child2

    def __str__(self):
        return "Uniform Crossover(\n)"
