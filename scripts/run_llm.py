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
    parser.add_argument("--game", type=str, default="hatespeech")
    parser.add_argument("--time-period", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=1024)
    parser.add_argument("--n-arms", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    # algorithm config
    parser.add_argument("--method", type=str, default="LLM")
    parser.add_argument("--noise-dim", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--based-weight-decay", type=float, default=0.0)
    parser.add_argument("--hyper-weight-decay", type=float, default=0.01)
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--z-coef", type=float, default=0.01)
    parser.add_argument("--NpS", type=int, default=16)
    parser.add_argument("--action-noise", type=str, default="pm")
    parser.add_argument("--update-noise", type=str, default="pm")
    parser.add_argument("--buffer-noise", type=str, default="sp")
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--update-start", type=int, default=2)
    parser.add_argument("--update-num", type=int, default=1)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--prior-scale", type=float, default=5.0)
    parser.add_argument("--model-type", type=str, default="hyper")
    parser.add_argument("--llm-name", type=str, default="pythia14m", choices=["gpt2", "pythia14m"])
    parser.add_argument("--use-lora", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fine-tune", type=int, default=0, choices=[0, 1])
    parser.add_argument("--out-bias", type=int, default=1, choices=[0, 1])
    # other config
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n-expe", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--log-dir", type=str, default="./results/bandit")
    args = parser.parse_known_args()[0]
    return args


args = get_args()
game = args.game
dir = f"{game.lower()}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
path = os.path.expanduser(os.path.join(args.log_dir, game, dir))
os.makedirs(path, exist_ok=True)

noise_param = {
    "hyper": {
        "action_noise": args.action_noise,
        "update_noise": args.update_noise,
        "buffer_noise": args.buffer_noise,
    },
    "ensemble": {
        "action_noise": "oh",
        "update_noise": "oh",
        "buffer_noise": "gs",
    }
}

param = {
    "LLM": {
        "log_interval": args.log_interval,
        "noise_dim": args.noise_dim,
        "lr": args.lr,
        "based_weight_decay": args.based_weight_decay,
        "hyper_weight_decay": args.hyper_weight_decay,
        "z_coef": args.z_coef,
        "optim": args.optim,
        "update_start": args.update_start,
        "update_num": args.update_num,
        "update_freq": args.update_freq,
        "batch_size": args.batch_size,
        "prior_scale": args.prior_scale,
        "NpS": args.NpS,
        "action_noise": args.action_noise,
        "update_noise": args.update_noise,
        "buffer_noise": args.buffer_noise,
        "buffer_size": args.buffer_size,
        "model_type": args.model_type,
        "llm_name": args.llm_name,
        "use_lora": args.use_lora,
        "fine_tune": args.fine_tune,
        "out_bias": args.out_bias,
        **noise_param[args.model_type],
    }
}

methods = [args.method]

game_config = {
    "hatespeech": {
        "n_features": args.n_features,
        "n_arms": args.n_arms,
        "llm_name": args.llm_name,
    },
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
    "T": args.time_period,
    "methods": methods,
    "param_dic": param,
    "labels": labels,
    "colors": colors,
    "path": path,
    "problem": game,
    "seed": args.seed,
    **game_config[game],
}
lin = exp.Textual_expe(**expe_params)

if store:
    pkl.dump(lin, open(os.path.join(path, "results.pkl"), "wb"))
# %%
