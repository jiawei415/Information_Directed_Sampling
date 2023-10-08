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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fg-lambda", type=float, default=1.0)
    parser.add_argument("--NpS", type=int, default=20)
    parser.add_argument("--action-noise", type=str, default="sgs")
    parser.add_argument("--update-noise", type=str, default="sgs")
    parser.add_argument("--noise-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layer", type=int, default=0)
    parser.add_argument("--update-num", type=int, default=100)
    parser.add_argument("--repeat-num", type=int, default=500)
    parser.add_argument("--time-period", type=int, default=50)
    parser.add_argument("--n-expe", type=int, default=3)
    parser.add_argument("--n-context", type=int, default=1)
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--logdir", type=str, default="./results/bandit")
    args = parser.parse_known_args()[0]
    return args


args = get_args()
game = args.game
now = datetime.now()
dir = f"{game.lower()}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.logdir, game, dir))
os.makedirs(path, exist_ok=True)

args.hidden_sizes = [args.hidden_size] * args.hidden_layer
hyper_params = {
    "noise_dim": args.noise_dim,
    "lr": args.lr,
    "optim": args.optim,
    "update_num": args.update_num,
    "batch_size": args.batch_size,
    "hidden_sizes": args.hidden_sizes,
    "NpS": args.NpS,
    "action_noise": args.action_noise,
    "update_noise": args.update_noise,
    "fg_lambda": args.fg_lambda,
    "fg_decay": True,
    "reset": False,
}

param = {
    "TS": {},
    "TS_hyper": {
        **hyper_params,
        "fg_lambda": 0.0,
    },
    "TS_hyper:Reset": {
        **hyper_params,
        "fg_lambda": 0.0,
        "reset": True,
        "update_num": args.repeat_num,
    },
    "TS_hyper:FG": {**hyper_params, "fg_decay": False},
    "TS_hyper:FG Decay": {**hyper_params},
    "VIDS_action": {"M": 10000, "optim_action": True},
    "VIDS_action_hyper": {
        "M": 10000,
        "optim_action": True,
        **hyper_params,
        "fg_lambda": 0.0,
    },
    "VIDS_action_hyper:Reset": {
        "M": 10000,
        "optim_action": True,
        **hyper_params,
        "fg_lambda": 0.0,
        "reset": True,
        "update_num": args.repeat_num,
    },
    "VIDS_action_hyper:FG": {
        "M": 10000,
        "optim_action": True,
        **hyper_params,
        "fg_decay": False,
    },
    "VIDS_action_hyper:FG Decay": {"M": 10000, "optim_action": True, **hyper_params},
    "VIDS_action:theta": {"M": 10000, "optim_action": False},
    "VIDS_action_hyper:theta": {
        "M": 10000,
        "optim_action": False,
        **hyper_params,
        "fg_lambda": 0.0,
    },
    "VIDS_policy": {"M": 10000, "optim_action": True},
    "VIDS_policy_hyper": {
        "M": 10000,
        "optim_action": True,
        **hyper_params,
        "fg_lambda": 0.0,
    },
    "VIDS_policy_hyper:Reset": {
        "M": 10000,
        "optim_action": True,
        **hyper_params,
        "fg_lambda": 0.0,
        "reset": True,
        "update_num": args.repeat_num,
    },
    "VIDS_policy_hyper:FG": {
        "M": 10000,
        "optim_action": True,
        **hyper_params,
        "fg_decay": False,
    },
    "VIDS_policy_hyper:FG Decay": {"M": 10000, "optim_action": True, **hyper_params},
    "VIDS_policy:theta": {"M": 10000, "optim_action": False},
    "VIDS_policy_hyper:theta": {
        "M": 10000,
        "optim_action": False,
        **hyper_params,
        "fg_lambda": 0.0,
    },
}

methods = [
    # "TS",
    # "VIDS_action",
    # "VIDS_policy",
    # "VIDS_action:theta",
    # "VIDS_policy:theta",
    "TS_hyper",
    # "VIDS_action_hyper",
    # "VIDS_policy_hyper",
    # "VIDS_action_hyper:theta",
    # "VIDS_policy_hyper:theta",
    # "TS_hyper:Reset",
    # "TS_hyper:FG",
    # "TS_hyper:FG Decay",
    # "VIDS_action_hyper:Reset",
    # "VIDS_action_hyper:FG",
    # "VIDS_action_hyper:FG Decay",
    # "VIDS_policy_hyper:Reset",
    # "VIDS_policy_hyper:FG",
    # "VIDS_policy_hyper:FG Decay",
]

game_config = {
    "FreqRusso": {"n_features": 5, "n_arms": 30, "T": args.time_period},
    "movieLens": {"n_features": 30, "n_arms": 207, "T": args.time_period},
    "Russo": {"n_features": 5, "n_arms": 30, "T": args.time_period},
    "Zhang": {"n_features": 100, "n_arms": 10, "T": args.time_period},
    "Synthetic-v1": {"n_features": 50, "n_arms": 20, "T": args.time_period},
    "Synthetic-v2": {"n_features": 50, "n_arms": 20, "T": args.time_period},
    "Synthetic-v3": {"n_features": 50, "n_arms": 20, "T": args.time_period},
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
