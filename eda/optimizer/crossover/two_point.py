import numpy as np

from optimizer.crossover.crossover_base import CrossoverBase


class TwoPoint(CrossoverBase):
    def __init__(self):
        pass

    def apply(self, parent1, parent2, fitness1, fitness2):
        dim = parent1.shape[0]
        p1, p2 = np.random.randint(0, dim, 2)
        p1, p2 = (p1, p2) if p1 < p2 else (p2, p1)
        child1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]], axis=0)
        child2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]], axis=0)
        return child1, child2

    def __str__(self):
        return "2-Point Crossover(\n)"
