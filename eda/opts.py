def objective_opts(parser):
    parser.add_argument("--objective-type", type=str, required=True,
                        choices=["one_max",  "two_min", "four_peaks",
                                 "deceptive_trap", "nk_landscape", "w_model"],
                        help="specify a objective function.")
    parser.add_argument("--dim", type=int, required=True,
                        help="dimension of the objective function.")
    parser.add_argument("--minimize", action="store_false",
                        help="the problem is whether minimization problem or maximization problem.")
    parser.add_argument("-t", type=int, default=3,
                        help="a user parameter which determines the difficulty of the four-peaks function.")
    parser.add_argument("-k", type=int, default=3,
                        help="a user parameter, which determines the ruggedness of the fitness landscape, of deceptive-k trap function and NK-landscape.")
    parser.add_argument("-d", type=float, default=0.1,
                        help="a user parameter, which determines the deceptiveness of the fitness landscape, of deceptive-k trap function.")
    parser.add_argument("--nk-seed", type=int, default=1,
                        help="a random number seed for NK-landscape where the seed is used to generate k bits and a lookup table randomly.")
    parser.add_argument("--mu", type=int, default=2,
                        help="a user parameter, which determines neutrality, of w-model function.")
    parser.add_argument("-v", type=int, default=4,
                        help="a user parameter, which determines epistasis, of w-model function.")
    parser.add_argument("-m", type=int, default=2,
                        help="a user parameter, which determines mullti-objectivity, of w-model function.")
    parser.add_argument("-n", type=int, default=5,
                        help="a user parameter, which determines the dimension of each objective function, of w-model function.")
    parser.add_argument("--gamma", type=int, default=0,
                        help="a user parameter, which determines the ruggedness and deceptiveness of the fitness landscape, of w-model function.")


def optimizer_opts(parser):
    parser.add_argument("--optim-type", type=str, required=True,
                        choices=["umda", "pbil", "mimic",
                                 "cga", "ecga", "aff_eda", "boa"],
                        help="specify a optimizer.")
    parser.add_argument("--lam", type=int, required=True,
                        help="population size.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate in PBIL and UMDA.")
    parser.add_argument("--negative-lr", type=float, default=None,
                        help="learning rate for negative example in PBIL.")
    parser.add_argument("--selection", type=str, default="tournament",
                        choices=["none", "block", "tournament", "roulette", "top"],
                        help="selection method which chooses individuals from a population based on the evaluation value.")
    parser.add_argument("--selection-rate", type=float, default=0.5,
                        help="selection rate, i.e., how many individuals are chosen when the selection method is applied to a population.")
    parser.add_argument("--tournament-size", type=int, default=2,
                        help="tournament size in the tournament selection.")
    parser.add_argument("--with-replacement", action="store_true",
                        help="sampling with replacement or not")
    parser.add_argument("--mutation-prob", type=float, default=0.01,
                        help="mutation probability in PBIL.")
    parser.add_argument("--mutation-shift", type=float, default=0.05,
                        help="amount of shift for mutation in PBIL.")
    parser.add_argument("--replacement", type=str, default="restricted",
                        choices=["truncation", "restricted"],
                        help="replacement method which replaces individuals in parent population with ones in candidate population.")
    parser.add_argument("--replace-rate", type=float, default=0.5,
                        help="replacement rate, i.e., how many individuals are replaced when the replacement method is applied to a population.")
    parser.add_argument("--constraint-k", type=int, default=None,
                        help="maximum number of parents of each node in BOA.")
    parser.add_argument("--metric", type=str, default="bic",
                        choices=["aic", "bic", "k2"],
                        help="evaluation criterion to measure the degree of dependencies among variables in BOA.")
    parser.add_argument("--window-size", type=int, default=2,
                        help="a user parameter, which determines trade-off between the goodness and the diversity in the population, in the restricted tournament replacement.")
    parser.add_argument("--max-num-evals", type=int, default=1e5,
                        help="Maximum number of evaluations.")


def utils_opts(parser):
    parser.add_argument("--seed", type=int, default=-1,
                        help="a random number seed for trials.")
    parser.add_argument("--seeds", type=str, nargs="+", default=None,
                        help="a random number seed for each trials. the length of the seeds must match the number of trials.")
    parser.add_argument("--trials", type=int, default=1,
                        help="how many independent trials.")


def log_opts(parser):
    parser.add_argument("--log-dir", type=str, default=None,
                        help="directory path to output logs.")
    parser.add_argument("--logging-step", type=int, default=10,
                        help="interval of outputting logs to directory.")
    parser.add_argument("--display-step", type=int, default=10,
                        help="interval of displaying logs to stdout.")
