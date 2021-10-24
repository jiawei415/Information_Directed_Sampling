# %%
""" Packages import """
import expe as exp
import numpy as np

# import jax.numpy as np
import pickle as pkl
import os
import utils
import time


# %%
# random number generation setup
np.random.seed(46)

# configurations
path = "/Users/szrlee/Code/Information_Directed_Sampling"

param = {
    "UCB1": {"rho": np.sqrt(2)},
    "LinUCB": {"lbda": 10e-4, "alpha": 10e-1},
    "BayesUCB": {"p1": 1, "p2": 1, "c": 0},
    "MOSS": {"rho": 0.2},
    "ExploreCommit": {"m": 10},
    "Tuned_GPUCB": {"c": 0.9},
    "IDS": {"M": 1000},
    "IDS_approx": {"N": 1000, "display_results": False},
    "IDS_sample": {"M": 10000, "VIDS": False},
    "VIDS_approx": {"rg": 10.0, "N": 1000},
    "VIDS_sample": {"M": 10000, "VIDS": True},
    "FGTS": {"fg_lambda": 1}
}

"""methods available : UCB1, TS, UCB_Tuned, BayesUCB, KG, KG_star, Approx_KG_star, MOSS, IDS, IDS_approx"""

# linear_methods = ["TS", "LinUCB", "BayesUCB", "GPUCB", "Tuned_GPUCB", "VIDS_sample"]
linear_methods = ["FGTS", "TS_SGMCMC", "TS", "LinUCB", "BayesUCB", "GPUCB", "Tuned_GPUCB", "VIDS_sample"]


"""Kind of Bandit problem"""
check_Linear = True
store = True # if you want to store the results
FGTSLinMAB = True
check_time = False


# %%
# Regret
labels, colors = utils.labelColor(linear_methods)
lin = exp.LinMAB_expe(
    n_expe=1,
    n_features=100,
    n_arms=2,
    T=250,
    methods=linear_methods,
    param_dic=param,
    labels=labels,
    colors=colors,
    movieLens=False,
    FGTSLinMAB=FGTSLinMAB,
)
# if store:
#     pkl.dump(lin, open(os.path.join(path, "lin10features.pkl"), "wb"))

# %%
# Computation
if check_time:
    import LinMAB as LM

    for i, j in zip([15, 30, 50, 100], [3, 5, 20, 30]):
        model = LM.PaperLinModel(u=np.sqrt(1 / 5), n_features=j, n_actions=i)
        model.threshold = 0.999
        alg = LM.LinMAB(model)
        t = time.time()
        for _ in range(100):
            model.flag = False
            alg.VIDS_sample(T=1000, M=10000)
        print((time.time() - t) / 1000 / 100)