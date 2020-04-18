def objective_opts(parser):
    parser.add_argument("--objective-type", type=str, required=True,
                        choices=["knapsack", "one_max", "four_peaks",
                                 "deceptive_order_3", "deceptive_order_4"])
    parser.add_argument("--setting-path", type=str, required=True)


def optimizer_opts(parser):
    parser.add_argument("--optim-type", type=str, default="pbil",
                        choices=["umda", "pbil", "mimic",
                                 "ga", "cga", "ecga", "aff_eda"])
    parser.add_argument("--lam", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--negative-lr", type=float, default=None)
    parser.add_argument("--replace-rate", type=float, default=0.1)
    parser.add_argument("--selection", type=str, default="tournament",
                        choices=["none", "block", "tournament", "roulette", "top"])
    parser.add_argument("--selection-rate", type=float, default=0.5)
    parser.add_argument("--sampling-rate", type=float, default=0.5)
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--with-replacement", action="store_true")
    parser.add_argument("--crossover", type=str, default="uniform",
                        choices=["none", "uniform", "two_point"])
    parser.add_argument("--crossover-prob", type=float, default=0.8)
    parser.add_argument("--mutation", type=str, default="none",
                        choices=["mutation", "none"])
    parser.add_argument("--mutation-prob", type=float, default=0.01)
    parser.add_argument("--mutation-shift", type=float, default=0.05)
    parser.add_argument("--replacement", type=str, default="restricted",
                        choices=["restricted", "trunc"])
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=1e5)


def utils_opts(parser):
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--seeds", type=str, nargs="+", default=None)
    parser.add_argument("--trials", type=int, default=1)


def log_opts(parser):
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--logging-step", type=int, default=1)
    parser.add_argument("--display-step", type=int, default=10)
    parser.add_argument("--save-param-step", type=int, default=10)


def plot_opts(parser):
    parser.add_argument("--path", type=str, required=True,
                        help="対象のログが存在するディレクトリを指定．")
    parser.add_argument("--save-path", type=str, default=None,
                        help="プロット結果を保存するディレクトリ．\
                        Noneの場合，--pathと同じディレクトリに保存される．")
    parser.add_argument("--target", type=str, default="general_loss")
    parser.add_argument("--xaxis", type=str, default="train steps")
    parser.add_argument("--yaxis", type=str, default=None)
