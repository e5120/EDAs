from logger import Logger
from objective import Knapsack, OneMax, FourPeaks, DeceptiveOrder3, DeceptiveOrder4
from objective.util import Item
from optimizer import UMDA, PBIL, MIMIC, ECGA, AffEDA, SimpleGA, CGA
from optimizer.selection import Block, Tournament, Roulette, Top
from optimizer.crossover import Uniform, TwoPoint
from optimizer.mutation import Mutation
from optimizer.replacement import RestrictedReplacement, TruncatedReplacement


def build_logger(args):
    logger = Logger(args.log_dir, args.logging_step, args.display_step,
                    args.save_param_step, args)
    args.log_dir = logger.dir_name
    return logger


def build_objective(args, noise=False):
    with open(args.setting_path) as f:
        if args.objective_type == "knapsack":
            target = -int(f.readline().split(",")[1])
            K = int(f.readline().split(",")[1])
            C = int(f.readline().split(",")[1])
            items = []
            for line in f:
                n, v, w = line.split(",")
                items.append(Item(n, int(v), int(w)))
            D = len(items)
            return Knapsack(K, D, C, items, target, noise=noise)
        elif args.objective_type == "one_max":
            target = -int(f.readline().split(",")[1])
            D = int(f.readline().split(",")[1])
            return OneMax(D, target, noise=noise)
        elif args.objective_type == "four_peaks":
            target = -int(f.readline().split(",")[1])
            D = int(f.readline().split(",")[1])
            T = float(f.readline().split(",")[1])
            return FourPeaks(D, T, target, noise=noise)
        elif args.objective_type == "deceptive_order_3":
            target = -int(f.readline().split(",")[1])
            D = int(f.readline().split(",")[1])
            d = float(f.readline().split(",")[1])
            return DeceptiveOrder3(D, target, d=d)
        elif args.objective_type == "deceptive_order_4":
            target = -int(f.readline().split(",")[1])
            D = int(f.readline().split(",")[1])
            return DeceptiveOrder4(D, target)
        else:
            return NotImplementedError


def build_optimizer(args, objective):
    categories = objective.categories
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
        replacement = build_replacement(args, len(categories))
        return MIMIC(categories, replacement,
                     lam=args.lam,
                     replace_rate=args.replace_rate)
    elif args.optim_type == "cga":
        return CGA(categories,
                   lam=args.lam)
    elif args.optim_type == "ecga":
        selection = build_selection(args)
        replacement = build_replacement(args, len(categories))
        return ECGA(categories, replacement,
                    lam=args.lam,
                    selection=selection)
    elif args.optim_type == "aff_eda":
        selection = build_selection(args)
        replacement = build_replacement(args, len(categories))
        return AffEDA(categories, replacement,
                      lam=args.lam,
                      selection=selection)
    elif args.optim_type == "ga":
        selection = build_selection(args)
        crossover = build_crossover(args)
        mutation = build_mutation(args)
        replacement = build_replacement(args, len(categories))
        return SimpleGA(categories, replacement,
                        lam=args.lam,
                        selection=selection,
                        crossover=crossover,
                        mutation=mutation,
                        crossover_prob=args.crossover_prob)
    else:
        return NotImplementedError


def build_selection(args):
    if args.selection == "block":
        return Block(sampling_rate=args.sampling_rate)
    elif args.selection == "tournament":
        return Tournament(sampling_rate=args.sampling_rate)
    elif args.selection == "roulette":
        return Roulette(selection_rate=args.selection_rate)
    elif args.selection == "top":
        return Top(selection_rate=args.selection_rate)
    elif args.selection == "none":
        return None
    else:
        return NotImplementedError


def build_crossover(args):
    if args.crossover == "uniform":
        return Uniform()
    elif args.crossover == "two_point":
        return TwoPoint()
    elif args.crossover == "none":
        return None
    else:
        return NotImplementedError


def build_mutation(args):
    if args.mutation == "mutation":
        return Mutation(args.mutation_prob)
    elif args.mutation == "none":
        return None
    else:
        return NotImplementedError


def build_replacement(args, dim):
    if args.replacement == "restricted":
        return RestrictedReplacement(args.window_size, dim)
    elif args.replacement == "trunc":
        return TruncatedReplacement()
    else:
        return NotImplementedError
