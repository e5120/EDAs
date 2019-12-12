import numpy as np
import torch


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
            if self.logger is not None:
                self.logger.info("{} steps, gloss: {:.3f}, theta_convergence: {:.4f}".format(
                                 step,
                                 gloss,
                                 self.optim.theta.max(axis=1).mean()),
                                 step)
                self.logger.write("general_loss", gloss, trial, step)
                self.logger.write("best_discrete_vec", torch.argmax(torch.Tensor(c), dim=2)[np.argmin(general_losses)], trial, step)
                # print("clustering", " - ".join(map(str, [c.idx_set for c in self.optim.cluster])))
                # self.logger.write("clustering", " - ".join(map(str, [c.idx_set for c in self.optim.cluster])), trial, step)
                # self.logger.write("eps", self.optim.eps, trial, step)
                # self.logger.write("delta", self.optim.delta, trial, step)
                # self.logger.save("theta", self.optim.theta, trial, step)
                # self.logger.write("start", info["start"], trial, step)
                # self.logger.write("end", info["end"], trial, step)
                # self.logger.write("gamma_value", self.optim.gamma, trial, step)
                # self.logger.save("s_value", self.optim.s, trial, step)
                # self.logger.save("weight", self.objective.x.detach().numpy(), trial, step)

            if gloss <= self.objective.target:
                if self.logger is not None:
                    self.logger.info("{} steps, gloss: {:.3f}, theta_convergence: {:.4f}".format(
                                     step,
                                     gloss,
                                     self.optim.theta.max(axis=1).mean()),
                                     step)
                break
            step += 1
        return gloss <= self.objective.target, step
