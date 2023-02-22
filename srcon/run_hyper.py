import os
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from SRCON_simulator.utils import BlackBoxSRCONObjective
from hypersolution import HyperSolution, set_seed, bound_point
from logger import configure, dump_params


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="korea",
        choices=["korea", "chengdu", "hanjiang", "nandong"],
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n-exp", type=int, default=3)
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    parser.add_argument("--time-period", type=int, default=int(1e6))
    parser.add_argument("--d-index", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--xi-coef", type=float, default=0.01)
    parser.add_argument("--norm-coef", type=float, default=0.01)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--update-num", type=int, default=100)
    parser.add_argument("--learning-start", type=int, default=2000)
    parser.add_argument("--action-num", type=int, default=10)
    parser.add_argument("--action-lr", type=float, default=0.001)
    parser.add_argument("--action-max-update", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="./results/srcon")
    args = parser.parse_known_args()[0]
    return args

def plotLoss(title, path, log=False):
    x_data, y_data = [], []
    for file in os.listdir(path):
        if "progress" in file:  
            data = pd.read_csv(os.path.join(path, file))
            x_data.append(data['step'].to_numpy())
            y_data.append(data['loss'].to_numpy())
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x = x_data[0]
    y = y_data.mean(0)
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(x, y, label="hypermodel")
    if y_data.shape[0] > 1:
        low_CI_bound, high_CI_bound = st.t.interval(
            0.95, len(y_data[0]) - 1, loc=y, scale=st.sem(y_data)
        )
        plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.2)
    if log:
        plt.yscale("log")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.ylabel("Step Loss")
    plt.xlabel("Time period")
    plt.legend(loc="best")
    plt.savefig(path + "/loss.pdf")


def plotReward(title, path, log=False):
    x_data, y_data = [], []
    for file in os.listdir(path):
        if "progress" in file:  
            data = pd.read_csv(os.path.join(path, file))
            x_data.append(data['step'].to_numpy())
            y_data.append(data['reward'].to_numpy())
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x = x_data[0]
    y = y_data.mean(0)
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(x, y, label="hypermodel")
    if y_data.shape[0] > 1:
        low_CI_bound, high_CI_bound = st.t.interval(
            0.95, len(y_data[0]) - 1, loc=y, scale=st.sem(y_data)
        )
        plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.2)
    if log:
        plt.yscale("log")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.ylabel("Step Reward")
    plt.xlabel("Time period")
    plt.legend(loc="best")
    plt.savefig(path + "/reward.pdf")


def debug(args, model, objective, logger):
    # collect data
    for _ in range(args.time_period):
        x = np.random.rand(args.d_theta)
        x = bound_point(x)
        y_true = objective.call(x)
        data = {"x": x, "y": y_true / 1000}
        model.put(data)

    # train model
    for t in range(args.time_period * 10):
        update_results = model.update()
        if t % args.log_freq == 0 or t == 1:
            logger.record("step", t)
            for key, val in update_results.items():
                logger.record(key, val)
            logger.dump(t)


def run(args, model, objective, logger):
    update_num = 0
    update_results = {}
    reward_list, error_list = [], []
    for t in range(1, args.time_period + 1):
        start_time = time.time()
        # collect data
        if t >= args.learning_start:
            x = model.select_action()
        else:
            x = np.random.rand(args.d_theta)
            x = bound_point(x)
        y_true = objective.call(x)
        y_pred = model.predict(np.array([x]))
        y_pred = y_pred[0][0] * 1000
        reward_list.append(y_true)
        error_list.append(abs(y_true - y_pred))
        data = {"x": x, "y": y_true / 1000}
        model.put(data)
        # update model
        if t % args.update_freq == 0 and t >= args.learning_start:
            for _ in range(args.update_num):
                update_results = model.update()
                update_num += 1
        end_time = time.time()
        # log data
        if t % args.log_freq == 0 or t == 1:
            logger.record("step", t)
            logger.record("time", end_time - start_time)
            logger.record("error", error_list[-1])
            logger.record("reward", reward_list[-1])
            logger.record("update_num", update_num)
            for key, val in update_results.items():
                logger.record(key, val)
            logger.dump(t)


def main(args):
    log_file = f"{args.dataset}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    for i in range(args.n_exp):
        seed = args.seed + i
        set_seed(seed)

        # create logger
        path = os.path.expanduser(os.path.join(args.log_dir, args.dataset, log_file))
        os.makedirs(path, exist_ok=True)
        logger = configure(path, log_suffix=str(seed))
        if i == 0:
            dump_params(logger, vars(args))

        # create environment
        objective = BlackBoxSRCONObjective(args.dataset)
        args.d_theta = objective.num_para

        # create hypermodel
        model = HyperSolution(
            args.d_index,
            args.d_theta,
            hidden_sizes=args.hidden_sizes,
            lr=args.lr,
            batch_size=args.batch_size,
            norm_coef=args.norm_coef,
            target_noise_coef=args.xi_coef,
            buffer_size=args.time_period,
            action_num=args.action_num,
            action_lr=args.action_lr,
            action_max_update=args.action_max_update,
        )

        if args.debug:
            debug(args, model, objective, logger)
        else:
            run(args, model, objective, logger)

    if args.debug:
        plotLoss(title="SRCON", path=logger.get_dir(), log=False)
    else:
        plotReward(title="SRCON", path=logger.get_dir(), log=False)



if __name__ == "__main__":
    main(args=get_args())