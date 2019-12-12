import os.path
import json
import logging
import datetime

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


class Logger(object):
    def __init__(self, dir_name, logging_step, display_step, save_model_step, args):
        if dir_name is not None:
            dir_name = "{}/{}_{}_{}".format(dir_name,
                                            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                                            args.theta_optim_type,
                                            args.theta_lam)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        self.dir_name = dir_name
        self.logging_step = logging_step
        self.display_step = display_step
        self.save_model_step = save_model_step
        self.args = args
        self.logger = logging.getLogger()

    # save user parameter in directory
    def save_setting(self):
        if self.dir_name is None:
            return
        self.info("save settings in {}".format(self.dir_name))
        if self.args is not None:
            with open("{}/settings.json".format(self.dir_name), "w", encoding="utf-8") as f:
                json.dump(self.args.__dict__, f, ensure_ascii=True, indent=4)

    # write the value to file
    def write(self, key, val, trial, step=0):
        if self.dir_name is None:
            return
        if step % self.logging_step == 0:
            with open("{}/{}_trial_{}.log".format(self.dir_name, key, trial), "a", encoding="utf-8") as f:
                f.write("{}\t{}\n".format(step, val))

    # write a message to file and std output
    def output(self, message, print=True):
        if print:
            self.info(message)
        if self.dir_name is None:
            return
        with open("{}/output.log".format(self.dir_name), "a", encoding="utf-8") as f:
            f.write("{}\n".format(message))

    # write a message about failure information to file
    def failure_output(self, message, print=True):
        if print:
            self.info(message)
        if self.dir_name is None:
            return
        with open("{}/failure_output.log".format(self.dir_name), "a", encoding="utf-8") as f:
            f.write("{}\n".format(message))

    # save a parameter of ndarray in directory
    def save(self, key, val, trial, step=0):
        if self.dir_name is None:
            return
        np_file = "{}/{}_trial_{}_step_{}".format(self.dir_name, key, trial, step)
        if step % self.save_model_step == 0:
            np.save(np_file, val)
            with open("{}/{}_trial_{}.log".format(self.dir_name, key, trial), "a", encoding="utf-8") as f:
                f.write(np_file)

    # write a message to std output as general information
    def info(self, message, step=0):
        if step % self.display_step == 0:
            self.logger.info(message)

    # write a message to std output as debug information
    def debug(self, message):
        self.logger.debug(message)
