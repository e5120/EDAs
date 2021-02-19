import numpy as np

from eda.optimizer import *


class Experimenter(object):
    """
    A class that executes an independent experimental trial.
    """
    def __init__(self, objective, optim, max_num_evals=1e5, logger=None):
        """
        Parameters
        ----------
        objective : eda.objective.objective_base.ObjectiveBase
            A objective function.
        optim : eda.optimizer.eda_base.EDABase
            A optimizer.
        max_num_evals : int, default 1e5
            Maximum number of evaluations.
        logger : eda.logger.Logger, default None
            A logger.
        """
        self.objective = objective
        self.optim = optim
        self.stop_condition = lambda o: o.num_evals > max_num_evals or o.is_convergence()
        self.logger = logger
        columns = ["step", "num_evals", "best_eval_gen", "best_indiv_gen",
                   "best_eval_all", "best_indiv_all", "convergence", "probability",
                   "order", "uni_freq", "bi_freq", "cluster",
                   "network_structure", "network_score"]
        self.logger.set_log_columns(columns)
        self.logger.set_display_columns(columns)

    def execute(self):
        is_success = False
        step = 0
        while not self.stop_condition(self.optim):
            step += 1
            lam = self.optim.lam

            c = np.array([self.optim.sampling() for _ in range(lam)])
            evals, obj_info = self.objective(c)

            optim_info = self.optim.update(c, evals)

            if self.logger:
                self.log(c, evals, obj_info, optim_info, step)
            best_eval = np.min(evals)
            # a optimum was found
            if np.abs(best_eval - self.objective.optimal_value) < 1e-8:
                is_success = True
                if self.logger and step % self.logger.logging_step:
                    self.log(c, evals, obj_info, optim_info, step, force=True)
                break
        return is_success, step

    def log(self, c, evals, obj_info, optim_info, step, force=False):
        best_idx = np.argmin(evals)
        self.logger.add("step", step, step, force=force)
        self.logger.add("num_evals", self.optim.num_evals, step, force=force)
        self.logger.add("best_eval_gen", evals[best_idx], step, force=force)
        self.logger.add("best_indiv_gen", c[best_idx], step, force=force)
        self.logger.add("best_eval_all", self.optim.best_eval, step, force=force)
        self.logger.add("best_indiv_all", self.optim.best_indiv, step, force=force)
        self.logger.add("convergence", self.optim.convergence(), step, force=force)
        if isinstance(self.optim, (PBIL, UMDA, CGA)):
            self.logger.add("probability", self.optim.theta, step, force=force)
        if isinstance(self.optim, MIMIC):
            self.logger.add("order", self.optim.order, step, force=force)
            self.logger.add("uni_freq", self.optim.uni_freq, step, force=force)
            self.logger.add("bi_freq", self.optim.bi_freq, step, force=force)
        if isinstance(self.optim, (ECGA, AffEDA)):
            # ToDo: implement
            pass
            # self.logger.add("cluster", , step, force=force)
        if isinstance(self.optim, BOA):
            self.logger.add("network_structure", optim_info["structure"], step, force=force)
            self.logger.add("network_score", optim_info["k2_algorithm_score"], step, force=force)

        self.logger.output(step, force=force)
