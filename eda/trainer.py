import numpy as np
import torch

from optimizer import MIMIC, AffEDA, ECGA


class Trainer(object):
    def __init__(self, objective, optim, train_steps=1e5, logger=None):
        self.objective = objective
        self.optim = optim
        self.stop_condition = lambda x, o: x > train_steps or o.is_convergence()
        self.logger = logger

    def train(self, trial):
        step = 1
        gloss = np.inf

        while not self.stop_condition(step, self.optim):
            lam = self.optim.lam
            losses = np.array([])
            general_losses = np.array([])

            c = np.array([self.optim.sampling() for _ in range(lam)], dtype=np.int32)
            loss, gloss, info = self.objective(c)
            losses = np.append(losses, loss)
            general_losses = np.append(general_losses, gloss)

            self.optim.update(c, losses)

            gloss = np.min(general_losses)
            # logging
            self.log(c, general_losses, gloss, info, trial, step)

            if gloss <= self.objective.target:
                self.logger.info("{} steps, gloss: {:.3f}, theta_convergence: {:.4f}".format(
                                 step,
                                 gloss,
                                 self.optim.theta.max(axis=1).mean()),
                                 step)
                break
            step += 1
        return gloss <= self.objective.target, step

    def log(self, c, general_losses, gloss, info, trial, step):
        if self.logger is None:
            return
        self.logger.info("{} steps, gloss: {:.3f}, theta_convergence: {:.4f}".format(
                         step,
                         gloss,
                         self.optim.theta.max(axis=1).mean()),
                         step)
        self.logger.write("general_loss", gloss, trial, step)
        self.logger.write("best_discrete_vec", torch.argmax(torch.Tensor(c), dim=2)[np.argmin(general_losses)], trial, step)
        self.logger.save("theta", self.optim.theta, trial, step)

        if isinstance(self.optim, MIMIC):
            self.logger.write("order", self.optim.order, trial, step)
        if isinstance(self.optim, ECGA) or isinstance(self.optim, AffEDA):
            self.logger.write("clustering", " - ".join(map(str, [c.idx_set for c in self.optim.cluster])), trial, step)
        if "start" in info:
            self.logger.write("start", info["start"], trial, step)
        if "end" in info:
            self.logger.write("end", info["end"], trial, step)
