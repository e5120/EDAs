import argparse
import time

import numpy as np

from builder import build_logger, build_objective, build_optimizer
from trainer import Trainer
import opts


def main(args):
    # generate random seed of each trial
    if args.seeds is None:
        if args.seed < 0:
            args.seed = np.random.randint(2**31)
        np.random.seed(args.seed)
        args.seeds = list(map(str, np.random.randint(0, 2**31, args.trials)))
    else:
        args.trials = len(args.seeds)
    # build each component
    logger = build_logger(args)
    logger.info("building now...")
    objective = build_objective(args, noise=False)
    logger.info("built a objective function: {}".format(objective))
    logger.save_setting()

    iter_list, best_eval_list, eval_num_list, time_list = [], [], [], []
    for i, seed in enumerate(args.seeds):
        logger.info("{} trial".format(i))
        # set random seed
        seed = int(seed)
        np.random.seed(seed)

        optim = build_optimizer(args, objective)
        logger.info("built a optimizer: {}".format(optim))

        start = time.time()

        trainer = Trainer(objective, optim,
                          train_steps=args.train_steps,
                          logger=logger)
        success, iteration = trainer.train(i)
        logger.info("success: {}, total iteration: {}\n".format(success, iteration))
        if success:
            iter_list.append(iteration)
            eval_num_list.append(optim.eval_count)
            time_list.append(time.time() - start)
            logger.output("{} trial,\titer:  {},\telapsed time: {}".format(i, iter_list[-1], time_list[-1]))
        else:
            logger.failure_output("{} trial,\tbest evaluation: {},\tbest individual: {}".format(i, optim.best_eval, optim.best_indiv), print=False)
        best_eval_list.append(optim.best_eval)
    if len(iter_list) > 0:
        logger.output("\nsuccess_rate\t{:.2f}  ({}/{})".format(
                                                   len(iter_list) / args.trials,
                                                   len(iter_list),
                                                   args.trials))
        logger.output("average iteration\t{:.2f}".format(np.mean(iter_list)))
        logger.output("std iteration\t{:.2f}".format(np.std(iter_list)))
        logger.output("average eval_num\t{:.2f}".format(np.mean(eval_num_list)))
        logger.output("std eval_num\t{:.2f}".format(np.std(eval_num_list)))
        logger.output("average elapsed time\t{:.2f}".format(np.mean(time_list)))
        logger.output("std elapsed time\t{:.2f}".format(np.std(time_list)))
    logger.output("average best_eval\t{:.2f}".format(np.mean(best_eval_list)))
    logger.output("std best_eval\t{:.2f}".format(np.std(best_eval_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")

    opts.objective_opts(parser)
    opts.optimizer_opts(parser)
    opts.utils_opts(parser)
    opts.log_opts(parser)

    args = parser.parse_args()

    main(args)
