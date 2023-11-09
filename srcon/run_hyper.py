import os, sys

sys.path.append(os.getcwd())
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

from hypersolution import HyperSolution
from utils import set_seed
from logger import configure, dump_params

MODELS = {"Hyper": "hyper", "Ensemble": "ensemble", "EpiNet": "epinet"}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="blackbox", choices=["srcon", "blackbox"]
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Branin",
        help="[korea, chengdu, hanjiang, nandong] for scron, [Branin, Schwefel, Ackley, Hartmann] for blackbox",
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--time-period", type=int, default=int(1e3))
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1])
    # algorithm config
    parser.add_argument("--d-theta", type=int, default=10)
    parser.add_argument("--d-index", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--z-coef", type=float, default=0.01)
    parser.add_argument("--NpS", type=int, default=16)
    parser.add_argument("--action-noise", type=str, default="sps")
    parser.add_argument("--update-noise", type=str, default="sps")
    parser.add_argument("--buffer-noise", type=str, default="sp")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--update-num", type=int, default=2)
    parser.add_argument("--learning-start", type=int, default=128)
    # network config
    parser.add_argument("--method", type=str, default="Hyper")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layer", type=int, default=2)
    parser.add_argument("--prior-scale", type=float, default=1)
    parser.add_argument("--posterior-scale", type=float, default=1)
    # action config
    parser.add_argument("--action-num", type=int, default=10)
    parser.add_argument("--action-single-noise", type=int, default=1, choices=[0, 1])
    parser.add_argument("--action-lr", type=float, default=0.001)
    parser.add_argument("--action-max-update", type=int, default=10)
    # other config
    parser.add_argument("--n-expe", type=int, default=3)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="./results/srcon")
    parser.add_argument("--save-model", type=int, default=0, choices=[0, 1])
    args = parser.parse_known_args()[0]
    args.model_type = MODELS[args.method]
    args.hidden_sizes = [args.hidden_size] * args.hidden_layer
    if args.method == "Ensemble":
        args.action_noise = "oh"
        args.update_noise = "oh"
        args.buffer_noise = "gs"
    elif args.method == "EpiNet":
        args.action_noise = "gs"
        args.update_noise = "gs"
    return args


def plotLoss(title, path, label, log=False):
    x_data, y_data = [], []
    for file in os.listdir(path):
        if "progress" in file:
            data = pd.read_csv(os.path.join(path, file))
            x_data.append(data["step"].to_numpy())
            y_data.append(data["loss"].to_numpy())
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x = x_data[0]
    y = y_data.mean(0)
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(x, y, label=label)
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


def plotResult(title, path, label, log=False):
    x_data, y_data = [], []
    for file in os.listdir(path):
        if "progress" in file:
            data = pd.read_csv(os.path.join(path, file))
            x_data.append(data["step"].values)
            y_data.append(data["best_reward"].values)
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x = x_data[0]
    y = y_data.mean(0)
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(x, y, label=label)
    if y_data.shape[0] > 1:
        low_CI_bound, high_CI_bound = st.t.interval(
            0.95, len(y_data[0]) - 1, loc=y, scale=st.sem(y_data)
        )
        plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.2)
    if log:
        plt.yscale("log")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.ylabel("Best Reward")
    plt.xlabel("Time period")
    plt.legend(loc="best")
    plt.savefig(path + "/reward.pdf")


def debug(args, model, objective, logger):
    # collect data
    for _ in range(args.time_period):
        x = objective.sample_action(1)
        x = objective.bound_point(x)
        y_true = objective.call(x)
        data = {"x": x, "y": y_true}
        model.put(data)

    # train model
    for t in range(args.time_period):
        update_results = model.update()
        if t % args.log_freq == 0 or t == 1:
            logger.record("step", t)
            for key, val in update_results.items():
                logger.record(key, val)
            logger.dump(t)


def run(args, model, objective, logger, run_id):
    update_num = 0
    update_results = {}
    reward_list, error_list = [], []
    maximize = True if args.env == "srcon" else False
    best_reward = -np.inf if maximize else np.inf
    for t in range(1, args.time_period + 1):
        start_time = time.time()
        # collect data
        if t >= args.learning_start:
            xs = objective.sample_action(args.action_num)
            x = model.select_action(
                xs, objective.bound_point, args.action_single_noise, maximize
            )
        else:
            x = objective.sample_action(1)
        x = objective.bound_point(x)
        y_true = objective.call(x)[0]
        y_pred = model.predict(x)[0]
        reward_list.append(y_true)
        error_list.append(abs(y_true - y_pred))
        if (maximize and y_true > best_reward) or (
            not maximize and y_true < best_reward
        ):
            best_reward = y_true
            best_action = x
            logger.info(f"Step {t}, Best Reward {best_reward}.")
            if args.save_model:
                np.save(os.path.join(logger.get_dir(), f"best_action{run_id}"), best_action)
                model.save_model(os.path.join(logger.get_dir(), f"best_model{run_id}.pkl"))
        data = {"x": x, "y": y_true}
        model.put(data)
        # update model
        if t % args.update_freq == 0 and t >= args.learning_start:
            for _ in range(args.update_num):
                update_results = model.update()
                update_num += 1
        end_time = time.time()
        # log data
        if t % args.log_freq == 0 or t == 1:
            logger.record("run_id", run_id)
            logger.record("step", t)
            logger.record("time", end_time - start_time)
            logger.record("error", error_list[-1])
            logger.record("reward", reward_list[-1])
            logger.record("best_reward", best_reward)
            logger.record("update_num", update_num)
            for key, val in update_results.items():
                logger.record(key, val)
            logger.dump(t)


def main(args):
    log_file = (
        f"{args.dataset}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    )
    for i in range(args.n_expe):
        seed = args.seed + i
        set_seed(seed, use_torch=True)

        # create logger
        path = os.path.expanduser(os.path.join(args.log_dir, args.dataset, log_file))
        os.makedirs(path, exist_ok=True)
        logger = configure(path, log_suffix=str(seed))
        if i == 0:
            dump_params(logger, vars(args))

        # create environment
        if args.env == "srcon":
            from blacksrcon import SRCON

            objective = SRCON(args.dataset)
            args.d_theta = objective.input_dim
        elif args.env == "blackbox":
            from blackbox import BlackBox, D_THETA

            args.d_theta = D_THETA.get(args.dataset, args.d_theta)
            objective = BlackBox(args.dataset, args.d_theta)

        # create hypermodel
        model = HyperSolution(
            args.d_index,
            args.d_theta,
            hidden_sizes=args.hidden_sizes,
            prior_scale=args.prior_scale,
            posterior_scale=args.posterior_scale,
            batch_size=args.batch_size,
            lr=args.lr,
            optim=args.optim,
            weight_decay=args.weight_decay,
            noise_coef=args.z_coef,
            buffer_size=args.time_period,
            buffer_noise=args.buffer_noise,
            NpS=args.NpS,
            action_noise=args.action_noise,
            update_noise=args.update_noise,
            action_num=args.action_num,
            action_lr=args.action_lr,
            action_max_update=args.action_max_update,
            model_type=args.model_type,
        )
        logger.info(f"Network structure:\n{str(model.model)}")

        if args.debug:
            debug(args, model, objective, logger)
        else:
            run(args, model, objective, logger, seed)

    title = "SRCON" if args.env == "srcon" else f"BlackBox: {args.dataset}"
    if args.debug:
        plotLoss(title=title, path=logger.get_dir(), label=args.method, log=False)
    else:
        plotResult(title=title, path=logger.get_dir(), label=args.method, log=False)


if __name__ == "__main__":
    main(args=get_args())