from eda.logger import Logger
from eda.objective import *
from eda.optimizer import *
from eda.optimizer.selection import *
from eda.optimizer.replacement import *


def build_logger(args):
    logger = Logger(args.log_dir, args,
                    logging_step=args.logging_step,
                    display_step=args.display_step)
    args.log_dir = logger.dir_path
    return logger


def build_objective(args):
    if args.objective_type == "one_max":
        return OneMax(args.dim,
                      minimize=args.minimize)
    elif args.objective_type == "two_min":
        return TwoMin(args.dim,
                      minimize=args.minimize)
    elif args.objective_type == "four_peaks":
        return FourPeaks(args.dim,
                         t=args.t,
                         minimize=args.minimize)
    elif args.objective_type == "deceptive_trap":
        return DeceptiveTrap(args.dim,
                             k=args.k,
                             d=args.d,
                             minimize=args.minimize)
    elif args.objective_type == "nk_landscape":
        return NKLandscape(args.dim,
                           k=args.k,
                           seed=args.nk_seed,
                           minimize=args.minimize)
    elif args.objective_type == "w_model":
        return WModel(args.dim,
                      mu=args.mu,
                      v=args.v,
                      m=args.m,
                      n=args.n,
                      gamma=args.gamma,
                      minimize=args.minimize)
    else:
        raise NotImplementedError


def build_optimizer(args, objective):
    categories = objective.categories
    selection = build_selection(args)
    replacement = build_replacement(args, len(categories))
    if args.optim_type == "umda":
        selection = build_selection(args)
        return UMDA(categories, args.lr, selection,
                    lam=args.lam)
    elif args.optim_type == "pbil":
        return PBIL(categories, args.lr,
                    lam=args.lam,
                    negative_lr=args.negative_lr,
                    mut_prob=args.mutation_prob,
                    mut_shift=args.mutation_shift)
    elif args.optim_type == "mimic":
        return MIMIC(categories, replacement,
                     lam=args.lam)
    elif args.optim_type == "cga":
        return CGA(categories,
                   lam=args.lam)
    elif args.optim_type == "ecga":
        return ECGA(categories, replacement,
                    lam=args.lam,
                    selection=selection)
    elif args.optim_type == "aff_eda":
        return AffEDA(categories, replacement,
                      lam=args.lam,
                      selection=selection)
    elif args.optim_type == "boa":
        return BOA(categories, selection, replacement,
                   lam=args.lam,
                   k=args.constraint_k,
                   criterion=args.metric)
    else:
        raise NotImplementedError


def build_selection(args):
    if args.selection == "block":
        return Block(selection_rate=args.selection_rate)
    elif args.selection == "tournament":
        return Tournament(selection_rate=args.selection_rate, k=args.tournament_size, replace=args.with_replacement)
    elif args.selection == "roulette":
        return Roulette(selection_rate=args.selection_rate)
    elif args.selection == "top":
        return Top(selection_rate=args.selection_rate)
    elif args.selection == "none":
        return None
    else:
        raise NotImplementedError


def build_replacement(args, dim):
    if args.replacement == "truncation":
        return Truncation(replace_rate=args.replace_rate)
    elif args.replacement == "restricted":
        return RestrictedTournament(dim,
                                    replace_rate=args.replace_rate,
                                    window_size=args.window_size)
    else:
        raise NotImplementedError
