import numpy as np

from objective.objective_base import ObjectiveBase


class FourPeaks(ObjectiveBase):
    def __init__(self, D, T, target, K=2, noise=False):
        super(FourPeaks, self).__init__(D, K, target, noise=noise)
        self.T = T * D

    def forward(self, c):
        c = np.argmax(c, axis=2)
        c_inv = c[:, ::-1]
        start = np.argmax(c, axis=1)
        end = np.argmin(c_inv, axis=1)
        loss = np.where(start > end, start, end)
        loss = -np.where((start > self.T) & (end > self.T), loss + self.D, loss)
        general_loss = loss
        info = {"start": start[np.argmin(loss)], "end": end[np.argmin(loss)]}
        return loss, general_loss, info

    def __str__(self):
        sup_str = "    " + super(FourPeaks, self).__str__().replace("\n", "\n    ")
        return 'Foue Peaks(\n' \
               '{}\n' \
               '    threshold: {}\n' \
               ')\n'.format(sup_str, self.T)
