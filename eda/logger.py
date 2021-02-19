import os
import csv
import json
import logging
import datetime
from collections import OrderedDict
from types import MappingProxyType

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s %(levelname)s] %(message)s")


class Logger(object):
    """
    A class to log a optimization process.
    """
    def __init__(self, dir_path, args, logging_step=10, display_step=10):
        """
        Parameters
        ----------
        dir_path : str
            Directory path to output logs.
        logging_step : int, default 10
            Interval of outputting logs to directory.
        display_step : int, default 10
            Interval of displaying logs to stdout.
        """
        if dir_path is not None:
            dir_path = "{}_{}".format(dir_path,
                                      datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            os.makedirs(dir_path, exist_ok=False)
        self.dir_path = dir_path
        self.trial_path = None
        self.logging_step = logging_step
        self.display_step = display_step
        self.args = args
        self.logger = logging.getLogger()
        self.log = OrderedDict()
        self.display = OrderedDict()
        # save arguments
        if self.dir_path and args:
            args.log_dir = self.dir_path
            with open("{}/settings.json".format(self.dir_path), "w", encoding="utf-8") as f:
                json.dump(args.__dict__, f, cls=JsonEncoder, ensure_ascii=True, indent=4)

    def set_log_columns(self, columns):
        """
        Set a column name of each log to be output in log file.

        Parameters
        ----------
        columns : array-like
            List of column names.
        """
        self.log = self._set_columns(columns)
        if self.trial_path:
            self.csv_file.writerow(columns)

    def set_display_columns(self, columns):
        """
        Set a column name of each log to be displayed in stdout.

        Parameters
        ----------
        columns : array-like
            List of column names.
        """
        self.display = self._set_columns(columns)

    def _set_columns(self, columns):
        """
        Set columns.

        Parameters
        ----------
        columns : array-like
            List of column names.

        Returns
        -------
        collections.OrderedDict
            The key-value data, where each of key is a column name and each of value is a observed value.
        """
        dic = OrderedDict({column: None for column in columns})
        return dic

    def add(self, key, val, step, force=False):
        """
        Add a log.

        Parameters
        ----------
        key : str
            Column name.
        val : any
            Observed value such as scalar, vector, and matrix.
        step : int
            Iteration.
        force : bool, default False
            If True, force to add logs.
        """
        if key in self.log and (step % self.logging_step == 0 or force):
            self.log[key] = val
        if key in self.display and (step % self.display_step == 0 or force):
            self.display[key] = val

    def output(self, step, force=False):
        """
        Output logs.

        Parameters
        ----------
        step : int
            Iteration.
        force : bool, default False
            If True, force to output logs.
        """
        if (step % self.logging_step == 0 or force) and self.trial_path:
            for key, val in self.log.items():
                if isinstance(val, (list, tuple, np.ndarray)):
                    val = np.array(val)
                    np_dir = "{}/{}".format(self.trial_path, key)
                    os.makedirs(np_dir, exist_ok=True)
                    np_file = "{}/{}_step".format(np_dir, step)
                    np.save(np_file, val)
                    self.log[key] = np_file
            self.csv_file.writerow(self.log.values())
        if step % self.display_step == 0 or force:
            msg = ", ".join(["{}: {}".format(key, val) for key, val in self.display.items()
                                if isinstance(val, (int, float, str, bool, *np.typeDict.values()))])
            self.logger.info(msg)

    def result(self, info, filename="results.csv"):
        """
        Output results.

        Parameters
        ----------
        info : dict
            Information.
        filename : str, default "result.csv"
            Filename to which the information will be output.
        """
        if self.trial_path:
            with open("{}/{}".format(self.trial_path, filename), "w") as f:
                result_file = csv.writer(f)
                result_file.writerow(info.keys())
                result_file.writerow(info.values())

    def open(self, trial, filename="logs.csv"):
        """
        Start logging of each independent trial.

        Parameters
        ----------
        trial : int
            The number of trials.
        filename : str, default "logs.csv"
            Filename which is output logs.
        """
        if self.dir_path:
            self.trial_path = "{}/{}".format(self.dir_path, trial)
            os.makedirs(self.trial_path, exist_ok=False)
            self.f = open("{}/{}".format(self.trial_path, filename), "w")
            self.csv_file = csv.writer(self.f)

    def close(self):
        """
        Finish logging of each independent trial.
        """
        if self.trial_path:
            self.trial_path = None
            self.f.close()

    def info(self, msg, step=0):
        if step % self.display_step == 0:
            self.logger.info(msg)


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return "shape of numpy.ndarray: {}".format(obj.shape)
        elif isinstance(obj, MappingProxyType):
            return obj["__module__"]
        elif isinstance(obj, object):
            return obj.__dict__
        else:
            return super(JsonEncoder, self).default(obj)
