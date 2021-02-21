import argparse
import time
import random

import numpy as np

from eda.builder import build_logger, build_objective, build_optimizer
from eda.experimenter import Experimenter
from eda import opts


def main(args):
    # generate a random seed of each trial
    if args.seeds is None:
        if args.seed < 0:
            args.seed = np.random.randint(2**31)
        np.random.seed(args.seed)
        args.seeds = list(np.random.randint(0, 2**31, args.trials))
    assert args.trials == len(args.seeds), \
        "The length of the seeds ({}) must match the number of trials ({}).\n".format(len(args.seeds), args.trials)
    # build each component
    logger = build_logger(args)
    logger.info("building now...")
    objective = build_objective(args)
    logger.info("built an objective function: {}".format(objective))
    optim_dummy = build_optimizer(args, objective)
    logger.info("built an optimizer: {}".format(optim_dummy))

    # independent trials
    iters, best_evals, num_evals, times = [], [], [], []
    for i, seed in enumerate(args.seeds, 1):
        logger.open(i)
        # set a random seed
        np.random.seed(seed)
        random.seed(seed)

        optim = build_optimizer(args, objective)

        start = time.time()

        exp = Experimenter(objective, optim,
                           max_num_evals=args.max_num_evals,
                           logger=logger)
        success, iteration = exp.execute()
        if success:
            iters.append(iteration)
            num_evals.append(optim.num_evals)
            times.append(time.time() - start)
        best_evals.append(optim.best_eval)
        info = {
            "success": success,
            "iters": iteration,
            "num_evals": optim.num_evals,
            "elapsed_time": time.time() - start,
            "best_eval": optim.best_eval,
            "best_indiv": np.argmax(optim.best_indiv, axis=-1),
        }
        logger.result(info)
        logger.info("Finished {}/{}".format(i, args.trials))
        logger.close()

    if len(iters):
        logger.info("Success rate\t{:.2f}  ({}/{})".format(len(iters) / args.trials, len(iters), args.trials))
        logger.info("Iterations\t{:.2f}±{:.2f}".format(np.mean(iters), np.std(iters)))
        logger.info("Number of evaluations\t{:.2f}±{:.2f}".format(np.mean(num_evals), np.std(num_evals)))
        logger.info("Elapsed time\t{:.2f}±{:.2f}".format(np.mean(times), np.std(times)))
    logger.info("Best evaluation\t{:.2f}±{:.2f}".format(np.mean(best_evals), np.std(best_evals)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")

    opts.objective_opts(parser)
    opts.optimizer_opts(parser)
    opts.utils_opts(parser)
    opts.log_opts(parser)

    args = parser.parse_args()

    main(args)
