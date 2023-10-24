# %%
""" Packages import """
import os, sys

sys.path.append(os.getcwd())
import json
import argparse
import expe as exp
import numpy as np

import pickle as pkl
import utils
import time

# random number generation setup
np.random.seed(46)

# configurations
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument("--game", type=str, default="Synthetic-v3")
    parser.add_argument("--time-period", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=50)
    parser.add_argument("--n-arms", type=int, default=20)
    parser.add_argument("--n-context", type=int, default=1)
    # algorithm config
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--z-coef", type=float, default=None)
    parser.add_argument("--NpS", type=int, default=20)
    parser.add_argument("--action-noise", type=str, default="gs")
    parser.add_argument("--update-noise", type=str, default="pn")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layer", type=int, default=2)
    parser.add_argument("--update-start", type=int, default=128)
    parser.add_argument("--update-num", type=int, default=2)
    # other config
    parser.add_argument("--n-expe", type=int, default=3)
    parser.add_argument("--logdir", type=str, default="./results/bandit")
    args = parser.parse_known_args()[0]
    return args


args = get_args()
game = args.game
dir = f"{game.lower()}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.logdir, game, dir))
os.makedirs(path, exist_ok=True)

args.hidden_sizes = [args.hidden_size] * args.hidden_layer
based_param = {
    "noise_dim": args.noise_dim,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "z_coef": args.z_coef,
    "optim": args.optim,
    "update_start": args.update_start,
    "update_num": args.update_num,
    "batch_size": args.batch_size,
    "hidden_sizes": args.hidden_sizes,
    "NpS": args.NpS,
    "action_noise": args.action_noise,
    "update_noise": args.update_noise,
}
param = {
    "Hyper": {
        **based_param,
        "action_noise": args.action_noise,
        "update_noise": args.update_noise,
    },
    "EpiNet": {
        **based_param,
        "action_noise": "gs",
        "update_noise": "gs",
        "NpS": 1,
    },
    "Ensemble": {
        **based_param,
        "action_noise": "oh",
        "update_noise": "oh",
    },
}

methods = [
    "Hyper",
    "EpiNet",
    "Ensemble",
]

base_config = {
    "n_features": args.n_features,
    "n_arms": args.n_arms,
    "T": args.time_period,
}
game_config = {
    "Synthetic-v1": base_config,
    "Synthetic-v2": base_config,
    "Synthetic-v3": base_config,
}

with open(os.path.join(path, "config.json"), "wt") as f:
    methods_param = {method: param.get(method, "") for method in methods}
    f.write(
        json.dumps(
            {
                "methods_param": methods_param,
                "game_config": game_config[game],
                "user_config": vars(args),
                "methods": methods,
                "labels": utils.mapping_methods_labels,
                "colors": utils.mapping_methods_colors,
            },
            indent=4,
        )
        + "\n"
    )
    f.flush()
    f.close()

"""Kind of Bandit problem"""
check_Linear = True
store = False  # if you want to store the results
check_time = False


# %%
# Regret
labels, colors = utils.labelColor(methods)
expe_params = {
    "n_expe": args.n_expe,
    "methods": methods,
    "param_dic": param,
    "labels": labels,
    "colors": colors,
    "path": path,
    "problem": game,
    **game_config[game],
}
if args.n_context > 0:
    lin = exp.FiniteContextHyperMAB_expe(n_context=args.n_context, **expe_params)
elif args.n_context < 0:
    lin = exp.InfiniteContextHyperMAB_expe(**expe_params)
else:
    lin = exp.HyperMAB_expe(**expe_params)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
