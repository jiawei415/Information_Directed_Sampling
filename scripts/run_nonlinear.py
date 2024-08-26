# %%
""" Packages import """
import os, sys

sys.path.append(os.getcwd())
import json
import time
import numpy as np
import argparse
import expe as exp
import utils

np.random.seed(2024)

def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument("--game", type=str, default="Synthetic-v1")
    parser.add_argument("--time-period", type=int, default=1000)
    parser.add_argument("--n-context", type=int, default=1)
    parser.add_argument("--n-features", type=int, default=50)
    parser.add_argument("--n-arms", type=int, default=20)
    parser.add_argument("--all-arms", type=int, default=1000)
    parser.add_argument("--freq-task", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eta", type=float, default=0.1)
    # algorithm config
    parser.add_argument("--method", type=str, default="Hyper")
    parser.add_argument("--noise-dim", type=int, default=4)
    parser.add_argument("--NpS", type=int, default=16)
    parser.add_argument("--z-coef", type=float, default=0.01)
    parser.add_argument("--action-noise", type=str, default="gs")
    parser.add_argument("--update-noise", type=str, default="pn")
    parser.add_argument("--buffer-noise", type=str, default="sp")
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--posterior-scale", type=float, default=1.0)
    parser.add_argument("--feature-sg", type=int, default=1, choices=[0, 1])
    # model config
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layer", type=int, default=2)
    # optimizer config
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    # buffer config
    parser.add_argument("--buffer-size", type=int, default=None)
    # update config
    parser.add_argument("--update-start", type=int, default=128)
    parser.add_argument("--update-num", type=int, default=4)
    parser.add_argument("--update-freq", type=int, default=1)
    # other config
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n-expe", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="./results/bandit")
    args = parser.parse_known_args()[0]
    # if args.game == "Synthetic-v1":
    #     args.prior_scale = 10.0
    #     args.lr = 0.0001
    # elif args.game == "Synthetic-v4":
    #     args.prior_scale = 1.0
    #     args.lr = 0.00001
    return args


args = get_args()

tag = f"{args.game.lower()}_{args.method}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.log_dir, args.game, tag))
os.makedirs(path, exist_ok=True)

args.hidden_sizes = [args.hidden_size] * args.hidden_layer
based_param = {
    "noise_dim": args.noise_dim,
    "NpS": args.NpS,
    "z_coef": args.z_coef,
    "action_noise": args.action_noise,
    "update_noise": args.update_noise,
    "buffer_noise": args.buffer_noise,
    "prior_scale": args.prior_scale,
    "posterior_scale": args.posterior_scale,
    "feature_sg": args.feature_sg,
    "hidden_sizes": args.hidden_sizes,
    "optim": args.optim,
    "lr": args.lr,
    "batch_size": args.batch_size,
    "weight_decay": args.weight_decay,
    "update_start": args.update_start,
    "update_num": args.update_num,
    "update_freq": args.update_freq,
    "buffer_size": args.buffer_size,
}

param = {
    "TS": {},
    "Hyper": {
        **based_param,
    },
    "EpiNet": {
        **based_param,
        "action_noise": "gs",
        "update_noise": "gs",
        "class_num": 2 if args.game.endswith("v3") else 1,
    },
    "Ensemble": {
        **based_param,
        "action_noise": "oh",
        "update_noise": "oh",
        "buffer_noise": "gs",
    },
    "LMCTS": {
        **based_param,
        "prior_scale": 0.0,
    },
}

methods = [args.method]

base_config = {
    "n_features": args.n_features,
    "n_arms": args.n_arms,
    "T": args.time_period,
    "freq_task": args.freq_task,
}

game_config = {
    "Synthetic-v1": {**base_config, "all_arms": args.all_arms, "eta": args.eta},
    "Synthetic-v2": {**base_config, "all_arms": args.all_arms, "eta": args.eta},
    "Synthetic-v3": {**base_config, "all_arms": args.all_arms, "eta": 0.0},
    "Synthetic-v4": {**base_config, "all_arms": args.all_arms, "eta": args.eta},
    "Synthetic-v5": {**base_config, "all_arms": args.all_arms, "eta": args.eta},
    "Synthetic-v6": {**base_config, "all_arms": args.all_arms, "eta": args.eta},
    "RealData-v1": {**base_config},
    "RealData-v2": {**base_config},
    "RealData-v3": {**base_config},
    "RealData-v4": {**base_config},
}

with open(os.path.join(path, "config.json"), "wt") as f:
    methods_param = {method: param.get(method, "") for method in methods}
    f.write(
        json.dumps(
            {
                "methods_param": methods_param,
                "game_config": game_config[args.game],
                "user_config": vars(args),
                "methods": methods,
            },
            indent=4,
        )
        + "\n"
    )
    f.flush()
    f.close()

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
    "problem": args.game,
    "seed": args.seed,
    **game_config[args.game],
}
if args.n_context > 0:
    lin = exp.FiniteContextHyperMAB_expe(n_context=args.n_context, **expe_params)
elif args.n_context < 0:
    lin = exp.InfiniteContextHyperMAB_expe(**expe_params)
else:
    lin = exp.HyperMAB_expe(**expe_params)

# %%
