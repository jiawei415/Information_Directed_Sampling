# %%
""" Packages import """
import os, sys

sys.path.append(os.getcwd())
import torch
import numpy as np
import argparse
import itertools as it
from functools import partial

from fvcore.nn import FlopCountAnalysis
from calflops import calculate_flops

from network import HyperNet, EnsembleNet, EpiNet
from utils import sample_action_noise, sample_update_noise

np.random.seed(2024)

def set_action_noise(action_noise, params):
    if action_noise == "gs":
        gen_action_noise = partial(sample_action_noise, "Gaussian", **params)
    elif action_noise == "sp":
        gen_action_noise = partial(sample_action_noise, "Sphere", **params)
    elif action_noise == "pn":
        gen_action_noise = partial(sample_action_noise, "UnifCube", **params)
    elif action_noise == "pm":
        gen_action_noise = partial(sample_action_noise, "PMCoord", **params)
    elif action_noise == "oh":
        gen_action_noise = partial(sample_action_noise, "OH", **params)
    elif action_noise == "hoh":
        gen_action_noise = partial(sample_action_noise, "HOH", **params)
    elif action_noise == "sps":
        gen_action_noise = partial(sample_action_noise, "Sparse", **params)
    elif action_noise == "spc":
        gen_action_noise = partial(
            sample_action_noise, "SparseConsistent", **params
        )
    return gen_action_noise

def set_update_noise(update_noise, params):
    if update_noise == "gs":
        gen_update_noise = partial(sample_update_noise, "Gaussian", **params)
    elif update_noise == "sp":
        gen_update_noise = partial(sample_update_noise, "Sphere", **params)
    elif update_noise == "pn":
        gen_update_noise = partial(sample_update_noise, "UnifCube", **params)
    elif update_noise == "pm":
        gen_update_noise = partial(sample_update_noise, "PMCoord", **params)
    elif update_noise == "oh":
        gen_update_noise = partial(sample_update_noise, "OH", **params)
    elif update_noise == "hoh":
        gen_update_noise = partial(sample_update_noise, "HOH", **params)
    elif update_noise == "sps":
        gen_update_noise = partial(sample_update_noise, "Sparse", **params)
    elif update_noise == "spc":
        gen_update_noise = partial(
            sample_update_noise, "SparseConsistent", **params
        )
    return gen_update_noise

def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument("--game", type=str, default="Synthetic-v1")
    parser.add_argument("--time-period", type=int, default=1000)
    parser.add_argument("--n-context", type=int, default=1)
    parser.add_argument("--n-features", type=int, default=100)
    parser.add_argument("--n-arms", type=int, default=50)
    parser.add_argument("--all-arms", type=int, default=1000)
    parser.add_argument("--freq-task", type=int, default=1, choices=[0, 1])
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    # algorithm config
    parser.add_argument("--method", type=str, default="Hyper",
                        choices=["TS", "Hyper", "EpiNet", "Ensemble", "LMCTS", "NeuralUCB"])
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--NpS", type=int, default=16)
    parser.add_argument("--z-coef", type=float, default=0.01)
    parser.add_argument("--action-noise", type=str, default="sp")
    parser.add_argument("--update-noise", type=str, default="pm")
    parser.add_argument("--buffer-noise", type=str, default="sp")
    parser.add_argument("--prior-scale", type=float, default=0.1)
    parser.add_argument("--posterior-scale", type=float, default=0.1)
    parser.add_argument("--based-prior", type=int, default=0, choices=[0, 1])
    parser.add_argument("--feature-sg", type=int, default=1, choices=[0, 1])
    parser.add_argument("--lmcts-beta", type=float, default=0.01)
    parser.add_argument("--NUCB-nu", type=float, default=1.0)
    # model config
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-layer", type=int, default=2)
    parser.add_argument("--ensemble-size", type=int, default=64)
    parser.add_argument("--ensemble-layer", type=int, default=0)
    # optimizer config
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    # buffer config
    parser.add_argument("--buffer-size", type=int, default=10000)
    # update config
    parser.add_argument("--update-start", type=int, default=128)
    parser.add_argument("--update-num", type=int, default=1)
    parser.add_argument("--update-freq", type=int, default=1)
    # other config
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n-expe", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--log-dir", type=str, default="~/results/bandit")
    args = parser.parse_known_args()[0]
    args.class_num = 1
    if args.game == "Synthetic-v1":
        args.prior_scale = 5.0
        args.posterior_scale = 1.0
        args.update_num = 4
    elif args.game == "Synthetic-v2":
        args.prior_scale = 2.0
        args.posterior_scale = 1.0
        args.update_num = 20
    elif args.game == "Synthetic-v3":
        args.prior_scale = 0.1
        args.posterior_scale = 0.1
        args.update_num = 1
        args.update_freq = 10
        args.class_num = 2
    elif args.game == "Synthetic-v4":
        args.prior_scale = 0.2
        args.posterior_scale = 0.1
        args.update_num = 10
    elif args.game == "Synthetic-v5":
        args.prior_scale = 5.0
        args.posterior_scale = 1.0
        args.update_num = 20
    elif args.game == "Synthetic-v6":
        args.prior_scale = 1.0
        args.posterior_scale = 1.0
        args.update_num = 20
    elif args.game == "RealData-v1":
        args.prior_scale = 0.2
        args.posterior_scale = 0.1
        args.update_num = 50
    elif args.game == "RealData-v3":
        args.prior_scale = 1.0
        args.posterior_scale = 0.1
        args.update_num = 50
    elif args.game == "RealData-v4":
        args.prior_scale = 2.0
        args.posterior_scale = 1.0
        args.update_num = 20
    elif "Russo" in args.game:
        args.n_context = 20
        args.prior_scale = 5.0
        args.posterior_scale = 1.0
        args.update_num = 10
    if args.update_noise == "pm":
        args.NpS = args.noise_dim * 2
    elif args.update_noise == "oh":
        args.NpS = args.noise_dim
    elif args.update_noise == "sps":
        args.NpS = len(list(it.combinations(list(range(args.noise_dim)), 2))) * len(list(it.product([1, -1], repeat=2)))
    elif args.update_noise == "spc":
        args.NpS = len(list(it.combinations(list(range(args.noise_dim)), 2))) * 2
    elif args.update_noise == "pn":
        args.NpS = len(list((it.product(range(2), repeat=args.noise_dim))))
    args.hidden_sizes = [args.hidden_size] * args.hidden_layer
    args.ensemble_sizes = [args.ensemble_size] * args.ensemble_layer
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.method == "Hyper":
        args.update_noise = "pm"
        args.action_noise = "sp"
    elif args.method == "EpiNet":
        args.update_noise = "gs"
        args.action_noise = "gs"
    elif args.method == "Ensemble":
        args.update_noise = "oh"
        args.action_noise = "oh"
    return args


args = get_args()

model_param = {
    "in_features": args.n_features,
    "hidden_sizes": args.hidden_sizes,
    "action_num": args.class_num,
    "noise_dim": args.noise_dim,
    "prior_scale": args.prior_scale,
    "posterior_scale": args.posterior_scale,
    "device": args.device,
}
if args.method == "Hyper":
    Net = HyperNet
    model_param.update({
        "feature_sg": args.feature_sg,
        "based_prior": args.based_prior,
    })
elif args.method == "EpiNet":
    Net = EpiNet
elif args.method == "Ensemble":
    model_param.update({"ensemble_sizes": args.ensemble_sizes})
    Net = EnsembleNet
else:
    raise NotImplementedError
model = Net(**model_param).to(args.device)
param_dict = {"Trainable": [], "Frozen": []}
trainable_param_size, frozen_param_size = 0, 0
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_param_size += param.numel()
        param_dict["Trainable"].append(name)
    else:
        frozen_param_size += param.numel()
        param_dict["Frozen"].append(name)
total_params = trainable_param_size + frozen_param_size
# print(f"Trainable parameters:\n", "\n".join(param_dict['Trainable']))
# print(f"Frozen parameters:\n", "\n".join(param_dict['Frozen']))
# print(f"Network structure:\n{str(model)}")
# print(f"Network parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")

noise_params = {"M": args.noise_dim, "dim": args.NpS, "batch_size": args.batch_size}
gen_update_noise = set_update_noise(args.update_noise, noise_params)
noise_params = {"M": args.noise_dim}
gen_action_noise = set_action_noise(args.action_noise, noise_params)

f_batch = torch.randn(args.batch_size, args.n_features).to(args.device)
update_noise = torch.from_numpy(gen_update_noise(batch_size=args.batch_size)).to(args.device)
inputs = [update_noise, f_batch]

# f_batch = torch.randn(args.n_arms, args.n_features).to(args.device)
# action_noise = torch.from_numpy(gen_action_noise(dim=1)).to(args.device)
# inputs = [action_noise, f_batch]

flops, macs, params = calculate_flops(model=model, args=inputs, print_results=False, print_detailed=False, include_backPropagation=False)
print(f"Method: {args.method}\tM: {args.noise_dim}\t\tNpS: {args.NpS}\tFLOPs: {flops}\tMACs: {macs}\tParams: {total_params}\tTrainable: {trainable_param_size}\tFrozen: {frozen_param_size}")

# inputs = (update_noise, f_batch)
# flops = FlopCountAnalysis(model, inputs)
# print(f"Method: {args.method}, M: {args.noise_dim}, NpS: {args.NpS}, FLOPs: {flops.total()}, Parameters: {trainable_param_size + frozen_param_size}")

# print(f"Total FLOPs: {flops.total()}")
# print(f"FLOPs by operator:\n{flops.by_operator()}")
# print(f"FLOPs by module:\n{flops.by_module()}")

