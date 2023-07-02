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
    parser.add_argument(
        "--game",
        type=str,
        default="Russo",
        choices=["Russo", "FreqRusso", "Zhang", "movieLens"],
    )
    parser.add_argument("--time-period", type=int, default=50)
    parser.add_argument("--n-expe", type=int, default=3)
    parser.add_argument("--d-index", type=int, default=10)
    parser.add_argument("--n-arms", type=int, default=30)
    parser.add_argument("--d-theta", type=int, default=10)
    parser.add_argument("--scheme", type=str, default="ts", choices={"ts", "ots"})
    parser.add_argument("--logdir", type=str, default="./results/bandit")
    args = parser.parse_known_args()[0]
    return args


args = get_args()
game = args.game
now = datetime.now()
dir = f"{game.lower()}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.logdir, game, dir))
os.makedirs(path, exist_ok=True)


param = {
    "TS": {
        "scheme": args.scheme,
    },
    "ES": {"M": args.d_index},
    "IS:Normal_Sphere": {
        "M": args.d_index,
        "haar": False,
        "index": "gaussian",
        "perturbed_noise": "sphere",
        "scheme": args.scheme,
    },
    "IS:Normal_Gaussian": {
        "M": args.d_index,
        "haar": False,
        "index": "gaussian",
        "perturbed_noise": "gaussian",
        "scheme": args.scheme,
    },
    "IS:Normal_PMCoord": {
        "M": args.d_index,
        "haar": False,
        "index": "gaussian",
        "perturbed_noise": "pm_coordinate",
        "scheme": args.scheme,
    },
    "IS:Normal_UnifGrid": {
        "M": args.d_index,
        "haar": False,
        "index": "gaussian",
        "perturbed_noise": "unif_grid",
        "scheme": args.scheme,
    },
    "IS:PMCoord_Gaussian": {
        "M": args.d_index,
        "haar": False,
        "index": "pm_coordinate",
        "perturbed_noise": "gaussian",
        "scheme": args.scheme,
    },
    "IS:PMCoord_Sphere": {
        "M": args.d_index,
        "haar": False,
        "index": "pm_coordinate",
        "perturbed_noise": "sphere",
        "scheme": args.scheme,
    },
    "IS:PMCoord_PMCoord": {
        "M": args.d_index,
        "haar": False,
        "index": "pm_coordinate",
        "perturbed_noise": "pm_coordinate",
        "scheme": args.scheme,
    },
    "IS:PMCoord_UnifGrid": {
        "M": args.d_index,
        "haar": False,
        "index": "pm_coordinate",
        "perturbed_noise": "unif_grid",
        "scheme": args.scheme,
    },
    "IS:Sphere_Gaussian": {
        "M": args.d_index,
        "haar": False,
        "index": "sphere",
        "perturbed_noise": "gaussian",
        "scheme": args.scheme,
    },
    "IS:Sphere_Sphere": {
        "M": args.d_index,
        "haar": False,
        "index": "sphere",
        "perturbed_noise": "sphere",
        "scheme": args.scheme,
    },
    "IS:Sphere_PMCoord": {
        "M": args.d_index,
        "haar": False,
        "index": "sphere",
        "perturbed_noise": "pm_coordinate",
        "scheme": args.scheme,
    },
    "IS:Sphere_UnifGrid": {
        "M": args.d_index,
        "haar": False,
        "index": "sphere",
        "perturbed_noise": "unif_grid",
        "scheme": args.scheme,
    },
    "IS:UnifGrid_Gaussian": {
        "M": args.d_index,
        "haar": False,
        "index": "unif_grid",
        "perturbed_noise": "gaussian",
        "scheme": args.scheme,
    },
    "IS:UnifGrid_Sphere": {
        "M": args.d_index,
        "haar": False,
        "index": "unif_grid",
        "perturbed_noise": "sphere",
        "scheme": args.scheme,
    },
    "IS:UnifGrid_PMCoord": {
        "M": args.d_index,
        "haar": False,
        "index": "unif_grid",
        "perturbed_noise": "pm_coordinate",
        "scheme": args.scheme,
    },
    "IS:UnifGrid_UnifGrid": {
        "M": args.d_index,
        "haar": False,
        "index": "unif_grid",
        "perturbed_noise": "unif_grid",
        "scheme": args.scheme,
    },
    # "LinUCB": {"lbda": 10e-4, "alpha": 10e-1},
    # "BayesUCB": {},
    # "GPUCB": {},
    # "Tuned_GPUCB": {"c": 0.9},
    # "VIDS_sample": {"M": 10000},
}

methods = [
    "TS",
    # "ES",
    "IS:PMCoord_Gaussian",
    "IS:PMCoord_Sphere",
    # "IS:PMCoord_PMCoord",
    "IS:PMCoord_UnifGrid",
    "IS:Normal_Sphere",
    "IS:Normal_Gaussian",
    # "IS:Normal_PMCoord",
    "IS:Normal_UnifGrid",
    # "IS:UnifGrid_Sphere",
    # "IS:UnifGrid_Gaussian",
    # "IS:UnifGrid_PMCoord",
    # "IS:UnifGrid_UnifGrid",
    # "IS:Sphere_Gaussian",
    # "IS:Sphere_Sphere",
    # "IS:Sphere_PMCoord",
    # "IS:Sphere_UnifGrid",
    # "IS:Haar",
    # "LinUCB",
    # "BayesUCB",
    # "GPUCB",
    # "Tuned_GPUCB",
    # "VIDS_sample",
]

game_config = {
    "FreqRusso": {
        "n_features": args.d_theta,
        "n_arms": args.n_arms,
        "T": args.time_period,
    },
    "movieLens": {"n_features": 30, "n_arms": 207, "T": args.time_period},
    "Russo": {"n_features": args.d_theta, "n_arms": args.n_arms, "T": args.time_period},
    "Zhang": {"n_features": args.d_theta, "n_arms": args.n_arms, "T": args.time_period},
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
store = True  # if you want to store the results
check_time = False


# %%
# Regret
labels, colors, markers = utils.labelColor(methods)
expe_params = {
    "n_expe": args.n_expe,
    "methods": methods,
    "param_dic": param,
    "labels": labels,
    "colors": colors,
    "markers": markers,
    "path": path,
    "problem": game,
    **game_config[game],
}
lin = exp.LinMAB_expe(**expe_params)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
