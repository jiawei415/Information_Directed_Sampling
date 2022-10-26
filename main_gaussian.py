# %%
""" Packages import """
import json
import argparse
import os
import expe as exp
import numpy as np

# import jax.numpy as np
import pickle as pkl
import utils
import time

import jax

# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=RuntimeWarning)
#     mean = np.mean([])
#     print(mean)

# %%
# Global flag to set a specific platform, must be used at startup.
jax.config.update("jax_platform_name", "cpu")
# random number generation setup
np.random.seed(46)

# configurations
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument("--game", type=str, default="Gussian")
    parser.add_argument("--time-period", type=int, default=50)
    parser.add_argument("--n-expe", type=int, default=3)
    parser.add_argument("--n-arms", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="~/results/bandit")
    args = parser.parse_known_args()[0]
    return args


args = get_args()
game = args.game
now = datetime.now()
dir = f"{game.lower()}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.logdir, game, dir))
os.makedirs(path, exist_ok=True)


param = {
    "TS": {},
    "BayesUCB": {},
    "GPUCB": {},
    "Tuned_GPUCB": {"c": 0.9},
}

methods = [
    "TS",
    "BayesUCB",
    "GPUCB",
    "Tuned_GPUCB",
]

with open(os.path.join(path, "config.json"), "wt") as f:
    methods_param = {method: param.get(method, "") for method in methods}
    f.write(
        json.dumps(
            {
                "methods_param": methods_param,
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
labels, colors = utils.labelColor(methods)
expe_params = {
    "T": args.time_period,
    "n_expe": args.n_expe,
    "n_arms": args.n_arms,
    "methods": methods,
    "param_dic": param,
    "labels": labels,
    "colors": colors,
    "path": path,
}
lin = exp.gaussian_expe(**expe_params)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
