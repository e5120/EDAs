import os
import glob
import argparse

from matplotlib import pyplot as plt
import numpy as np

import opts


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    filenames = sorted(glob.glob("{}/{}_trial_*.log".format(args.path, args.target)))
    x_all = []
    y_all = []
    max_x = None
    max_len = 0
    for filename in filenames:
        with open(filename) as f:
            x, y = [], []
            for line in f:
                step, val = line.strip().split("\t")
                x.append(int(step))
                if val.startswith("result"):
                    y.append(np.load("{}.npy".format(val)))
                else:
                    y.append(float(val))
            x = np.array(x)
            y = np.array(y)
        x_all.append(x)
        y_all.append(y)
        if max_len < len(x):
            max_len = len(x)
            max_x = x
        # 値がスカラーの場合
        if len(y.shape) == 1:
            plt.plot(x, y)
        # 値がベクトルの場合
        else:
            # plt.plot(x, y.reshape(y.shape[0], -1))
            # print(y[0, :, 3])
            # print(y[-1, :, 3])
            # plt.plot(x, y[:, :, 3])
            # print(y.shape)
            for i in range(y.shape[1]):
                for j in range(y.shape[2]):
                    plt.plot(x, y[:, i, j], label="{}th".format(j))
                plt.xlabel(args.xaxis)
                plt.ylabel(args.yaxis)
                plt.legend(loc='upper left')
                plt.savefig("{}/{}_D_{}.pdf".format(args.save_path, os.path.splitext(os.path.basename(filename))[0], i+1))
                plt.clf()
        plt.xlabel(args.xaxis)
        plt.ylabel(args.yaxis)

        plt.savefig("{}/{}.pdf".format(args.save_path, os.path.splitext(os.path.basename(filename))[0]))
        plt.clf()

    if len(y_all[0].shape) == 1:
        for i, (x, y) in enumerate(zip(x_all, y_all)):
            plt.plot(x, y, label="trial_{}".format(i+1))
            y_all[i] = np.append(y, (max_len - len(y)) * [np.nan])
        plt.xlabel(args.xaxis)
        plt.ylabel(args.yaxis)
        plt.legend()
        plt.savefig("{}/{}_all.pdf".format(args.save_path, args.target))
        plt.clf()

    if len(y_all[0].shape) == 1:
        y_all = np.array(y_all)
        y_ave = np.nanmean(y_all, axis=0)
        y_std = np.nanstd(y_all, axis=0)
        plt.plot(max_x, y_ave)
        plt.fill_between(max_x, y_ave - y_std, y_ave + y_std, facecolor="r", alpha=0.5)
        plt.xlabel(args.xaxis)
        plt.ylabel(args.yaxis)
        plt.savefig("{}/{}_mean_std.pdf".format(args.save_path, args.target))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plot.py")

    opts.plot_opts(parser)

    args = parser.parse_args()

    if args.path[-1] == "/":
        args.path = args.path[:-1]
    if args.save_path is None:
        args.save_path = args.path
    if args.yaxis is None:
        args.yaxis = args.target
    main(args)
