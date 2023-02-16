import os
import time
import argparse
import numpy as np

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
    parser.add_argument("--time-period", type=int, default=1000)
    parser.add_argument("--d-index", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--xi-coef", type=float, default=0.01)
    parser.add_argument("--norm-coef", type=float, default=0.01)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--update-freq", type=int, default=10)
    parser.add_argument("--update-num", type=int, default=10)
    parser.add_argument("--learning-start", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="./results/srcon")
    args = parser.parse_known_args()[0]
    return args


args = get_args()
set_seed(args.seed)

# create logger
dir = f"{args.dataset}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.logdir, args.dataset, dir))
os.makedirs(path, exist_ok=True)
logger = configure(path)
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
)

update_num = 0
update_results = {}
for t in range(1, args.time_period + 1):
    # collect data
    x = np.random.rand(args.d_theta)
    x = bound_point(x)
    y_true = objective.call(x)
    data = {"x": x, "y": y_true / 1000}
    model.put(data)
    # update model
    if t % args.update_freq == 0 or t >= args.learning_start:
        for _ in range(args.update_num):
            update_results = model.update()
            update_num += 1
    # evaluate model
    if t % args.eval_freq == 0 or t == 1:
        error_list = []
        for _ in range(10):
            x = np.random.rand(args.d_theta)
            x = bound_point(x)
            y_true = objective.call(x)
            y_pred = model.predict(np.array([x]))
            y_pred = y_pred[0][0] * 1000
            error_list.append(abs(y_true - y_pred))
        error = np.mean(error_list)
        # log data
        logger.record("time_step", t)
        logger.record("error", error)
        logger.record(f"update_num", update_num)
        for key, val in update_results.items():
            logger.record(key, val)
        logger.dump(t)
