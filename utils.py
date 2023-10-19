""" Packages import """
import numpy as np
import os
import csv

import random as rd
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm
import inspect

cmap = {
    0: "black",
    1: "blue",
    2: "yellow",
    3: "green",
    4: "red",
    5: "grey",
    6: "purple",
    7: "brown",
    8: "pink",
    9: "cyan",
}


mapping_methods_labels = {
    "Hyper": "Hyper",
    "EpiNet": "EpiNet",
    "Ensemble": "Ensemble",
    "LinUCB": "LinUCB",
    "LinUCB:test": "red-test",
    "BayesUCB": "BayesUCB",
    "GPUCB": "GPUCB",
    "Tuned_GPUCB": "Tuned_GPUCB",
    "TS": "TS - Conjugacy",
    "TS:G": "TS - Gaussian Conjugacy",
    "TS:B": "TS - Beta Conjugacy",
    "ES": "Ensemble Sampling",
    "ES:G": "Ensemble Sampling - Gaussian",
    "ES:E": "Ensemble Sampling - Exponential",
    "IS:Sphere": "Index Sampling: Sphere",
    "IS:Haar": "Index Sampling: Haar",
    "TS_hyper": "TS - HyperModel",
    "TS_hyper:Reset": "TS - HyperModel Reset",
    "TS_hyper:FG": "TS - HyperModel FG",
    "TS_hyper:FG Decay": "TS - HyperModel FG Decay",
    "VIDS_action": "VIDS-action - Conjugacy",
    "VIDS_action_hyper": "VIDS-action - HyperModel",
    "VIDS_action_hyper:Reset": "VIDS-action - HyperModel Reset",
    "VIDS_action_hyper:FG": "VIDS-action - HyperModel FG",
    "VIDS_action_hyper:FG Decay": "VIDS-action - HyperModel FG Decay",
    "VIDS_action:theta": "VIDS-action - Conjugacy theta",
    "VIDS_action_hyper:theta": "VIDS-action - HyperModel theta",
    "VIDS_policy": "VIDS-policy - Conjugacy",
    "VIDS_policy_hyper": "VIDS-policy - HyperModel",
    "VIDS_policy_hyper:Reset": "VIDS-policy - HyperModel Reset",
    "VIDS_policy_hyper:FG": "VIDS-policy - HyperModel FG",
    "VIDS_policy_hyper:FG Decay": "VIDS-policy - HyperModel FG Decay",
    "VIDS_policy:theta": "VIDS-policy - Conjugacy theta",
    "VIDS_policy_hyper:theta": "VIDS-policy - HyperModel theta",
}


mapping_methods_colors = {
    "Hyper": "blue",
    "EpiNet": "green",
    "Ensemble": "red",
    "LinUCB": "green",
    "LinUCB:test": "red",
    "BayesUCB": "purple",
    "GPUCB": "violet",
    "Tuned_GPUCB": "blue",
    "TS": "black",
    "TS:G": "black",
    "TS:B": "black",
    "ES": "green",
    "ES:G": "green",
    "ES:E": "green",
    "IS:Sphere": "red",
    "IS:Haar": "blue",
    "TS_hyper": "green",
    "TS_hyper:Reset": "purple",
    "TS_hyper:FG": "violet",
    "TS_hyper:FG Decay": "darkpurple",
    "VIDS_action": "blue",
    "VIDS_action_hyper": "red",
    "VIDS_action_hyper:Reset": "brown",
    "VIDS_action_hyper:FG": "salmon",
    "VIDS_action_hyper:FG Decay": "darksalmon",
    "VIDS_action:theta": "brown",
    "VIDS_action_hyper:theta": "salmon",
    "VIDS_policy": "yellow",
    "VIDS_policy_hyper": "grey",
    "VIDS_policy_hyper:Reset": "pink",
    "VIDS_policy_hyper:FG": "cyan",
    "VIDS_policy_hyper:FG Decay": "darkcyan",
    "VIDS_policy:theta": "pink",
    "VIDS_policy_hyper:theta": "cyan",
}


def labelColor(methods):
    """
    Map methods to labels and colors for regret curves
    :param methods: list, list of methods
    :return: lists, labels and vectors
    """
    labels = [
        mapping_methods_labels[m] if m in mapping_methods_labels.keys() else m
        for m in methods
    ]
    colors = [
        mapping_methods_colors[m] if m in mapping_methods_colors.keys() else None
        for m in methods
    ]
    return labels, colors


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rd.choice(indices)

def rd_max(vector):
    """
    Compute random among eligible maximum value
    :param vector: np.array
    :return: int, random index among eligible maximum value
    """
    index = rd_argmax(vector)
    return vector[index]


def haar_matrix(M):
    """
    Haar random matrix generation
    """
    z = np.random.randn(M, M).astype(np.float32)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return np.multiply(q, ph)


def sphere_matrix(N, M):
    v = np.random.randn(N, M).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def multi_haar_matrix(N, M):
    v = np.zeros(((N // M + 1) * M, M))
    for _ in range(N // M + 1):
        v[np.arange(M), :] = haar_matrix(M)
    return v[np.arange(N), :]


def display_results(delta, g, ratio, p_star):
    """
    Display quantities of interest in IDS algorithm
    """
    print("delta {}".format(delta))
    print("g {}".format(g))
    print("ratio : {}".format(ratio))
    print("p_star {}".format(p_star))


def plotRegret(labels, regret, colors, title, path, log=False):
    """
    Plot Bayesian regret
    :param labels: list, list of labels for the different curves
    :param mean_regret: np.array, averaged regrets from t=1 to T for all methods
    :param colors: list, list of colors for the different curves
    :param title: string, plot's title
    """

    all_regrets = regret["all_regrets"]
    mean_regret = regret["mean_regret"]
    plt.figure(figsize=(10, 8), dpi=80)

    T = mean_regret.shape[1]
    for i, l in enumerate(labels):
        c = cmap[i] if not colors else colors[i]
        x = np.arange(T)
        low_CI_bound, high_CI_bound = st.t.interval(
            0.95, T - 1, loc=mean_regret[i], scale=st.sem(all_regrets[i])
        )
        # low_CI_bound = np.quantile(all_regrets[i], 0.05, axis=0)
        # high_CI_bound = np.quantile(all_regrets[i], 0.95, axis=0)
        plt.plot(x, mean_regret[i], c=c, label=l)
        plt.fill_between(x, low_CI_bound, high_CI_bound, color=c, alpha=0.2)
        if log:
            plt.yscale("log")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.ylabel("Cumulative regret")
    plt.xlabel("Time period")
    plt.legend(loc="best")
    plt.savefig(path + "/regret.pdf")


def storeRegret(models, methods, param_dic, n_expe, T, path, use_torch=False):
    """
    Compute the experiment for all specified models and methods
    :param models: list of MAB
    :param methods: list of algorithms
    :param param_dic: parameters for all the methods
    :param n_expe: number of trials
    :param T: Time horizon
    :return: Dictionnary with results from the experiments
    """
    all_regrets = np.zeros((len(methods), n_expe, T), dtype=np.float32)
    os.makedirs(os.path.join(path, "csv_data"), exist_ok=True)
    for i, m in enumerate(methods):
        set_seed(2022, use_torch=use_torch)
        alg_name = m.split(":")[0]
        file_name = m.replace(":", "_").replace(" ", "_").lower()
        file = open(os.path.join(path, "csv_data", f"{file_name}.csv"), "w+t")
        writer = csv.writer(file, delimiter=",")
        for j in tqdm(range(n_expe)):
            model = models[j]
            alg = model.__getattribute__(alg_name)
            args = inspect.getfullargspec(alg)[0][2:]
            args = [T] + [param_dic[m][i] for i in args]
            reward, regret = alg(*args)
            writer.writerow(regret)
            file.flush()
            all_regrets[i, j, :] = np.cumsum(regret)
        print(f"{m}: {np.mean(all_regrets[i], axis=0)[-1]}")

    mean_regret = all_regrets.mean(axis=1)
    results = {
        "mean_regret": mean_regret,
        "all_regrets": all_regrets,
    }
    if models[0].store_IDS:
        IDS_res = [m.IDS_results for m in models]
        results["IDS_results"] = IDS_res
    return results


def plot_IDS_results(T, n_expe, results):
    """
    Plot the evolution of delta, g and IR with 90% quantiles for the experiments
    :param T: Time horizon
    :param n_expe: Number of experiments
    :param results: Results of the experiments as stored in StoreRegret(..)['IDS_results']
    :return:
    """
    delta = np.empty((T, n_expe))
    g = np.empty((T, n_expe))
    IR = np.empty((T, n_expe))
    for i, r in enumerate(results):
        r["policy"] = np.asarray(r["policy"])
        delta[: r["policy"].shape[0], i] = (np.asarray(r["delta"]) * r["policy"]).sum(
            axis=1
        )
        g[: r["policy"].shape[0], i] = (np.asarray(r["g"]) * r["policy"]).sum(axis=1)
        IR[: r["policy"].shape[0], i] = np.asarray(r["IR"])
    x = np.arange(1, T + 1)
    f, (ax1, ax2, ax3) = plt.subplots(3)
    plt.xlabel("Time horizon")
    ax1.semilogy(x, delta.mean(axis=1))
    ax1.fill_between(
        x,
        np.quantile(delta, 0.05, axis=1),
        np.quantile(delta, 0.95, axis=1),
        color="lightblue",
    )
    ax2.semilogy(x, g.mean(axis=1))
    ax2.fill_between(
        x, np.quantile(g, 0.05, axis=1), np.quantile(g, 0.95, axis=1), color="lightblue"
    )
    ax3.plot(x, IR.mean(axis=1), color="r")
    ax3.fill_between(
        x, np.quantile(IR, 0.05, axis=1), np.quantile(IR, 0.95, axis=1), color="#FFA07A"
    )
    ax1.set_title("Average Delta")
    ax2.set_title("Average Information Gain")
    ax3.set_title("Average Information Ratio")
    plt.show()


def build_finite(L, K, N):
    """
    Build automatically a finite bandit environment
    :param L: int, number of possible values for theta
    :param K: int, number of arms
    :param N: int, number of possible rewards
    :return: np.arrays, parameters required for launching an experiment with a finite bandit
    (prior, q values and R function)
    """
    R = np.linspace(0.0, 1.0, N)
    q = np.random.uniform(size=(L, K, N))
    for i in range(q.shape[0]):
        q[i] = np.apply_along_axis(lambda x: x / x.sum(), 1, q[i])
    p = np.random.uniform(0, 1, L)
    p = p / p.sum()
    return p, q, R


def build_bernoulli_finite_set(L, K):
    r = np.array([0, 1])
    p = np.ones(L) / L
    q = np.empty((L, K, 2))
    q[:, :, 0] = np.random.uniform(size=L * K).reshape((L, K))
    q[:, :, 1] = 1 - q[:, :, 0]
    return p, q, r


def set_seed(seed, use_torch=False):
    np.random.seed(seed)
    rd.seed(seed)
    if use_torch:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)
