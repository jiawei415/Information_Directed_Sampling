""" Packages import """
from agent.BernoulliMAB import BetaBernoulliMAB
from agent.GaussianMAB import GaussianMAB
from agent.FiniteSetsMAB import FiniteSets
from agent.closedform import LinMAB
from agent.MAB import GenericMAB
from env.linear import (
    FGTSLinModel,
    PaperLinModel,
    FreqPaperLinModel,
    ColdStartMovieLensModel,
)
from env.finitecontext import (
    FiniteContextFGTSLinModel,
    FiniteContextPaperLinModel,
    FiniteContextFreqPaperLinModel,
    SyntheticNonlinModel,
)
from env.infinitecontext import (
    InfiniteContextPaperLinModel,
    InfiniteContextFreqPaperLinModel,
)
from utils import (
    plotRegret,
    storeRegret,
    cmap,
    build_bernoulli_finite_set,
    plot_IDS_results,
)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def bernoulli_expe(
    n_expe,
    n_arms,
    T,
    methods,
    param_dic,
    labels,
    colors,
    path,
    problem="Bernoulli",
    doplot=True,
    track_ids=False,
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Bernoulli Bandits
    :param n_expe: int, number of experiments
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """

    models = [BetaBernoulliMAB(problem, n_arms) for _ in range(n_expe)]
    if track_ids:
        for m in models:
            m.store_IDS = True
    results = storeRegret(models, methods, param_dic, n_expe, T, path)
    if doplot:
        plotRegret(labels, results, colors, "Binary rewards", path)
    if track_ids:
        plot_IDS_results(T, n_expe, results["IDS_results"])
    return results


def gaussian_expe(
    n_expe,
    n_arms,
    T,
    methods,
    param_dic,
    labels,
    colors,
    path,
    problem="Gaussian",
    doplot=True,
    track_ids=False,
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Gaussian Bandits with Gaussian prior
    :param n_expe: int, number of experiments
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """

    models = [GaussianMAB(problem, n_arms) for _ in range(n_expe)]
    if track_ids:
        for m in models:
            m.store_IDS = True
    results = storeRegret(models, methods, param_dic, n_expe, T, path)
    if doplot:
        plotRegret(labels, results, colors, "Gaussian rewards", path)
    if track_ids:
        plot_IDS_results(T, n_expe, results["IDS_results"])
    return results


def LinMAB_expe(
    n_expe,
    n_features,
    n_arms,
    T,
    methods,
    param_dic,
    labels,
    colors,
    path,
    problem="FreqRusso",
    doplot=True,
    # track_ids=False,
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Linear Gaussian Bandits with multivariate Gaussian prior
    :param n_expe: int, number of experiments
    :param n_features: int, dimension of feature vectors
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :param problem: str, choose from {'FreqRusso', 'Zhang', 'Russo', 'movieLens'}
    :param path: str
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    if problem == "movieLens":
        models = [LinMAB(ColdStartMovieLensModel()) for _ in range(n_expe)]
        log = True
        title = "Movie recommendation system - n_arm: 207 - n_features: 30"
    elif problem == "Zhang":
        models = [LinMAB(FGTSLinModel(n_features, n_arms)) for _ in range(n_expe)]
        log = False
        title = (
            "Linear Gaussian Model (Zhang, 2021) - n_arms: {} - n_features: {}".format(
                n_arms, n_features
            )
        )
    elif problem == "FreqRusso":
        u = 1 / np.sqrt(5)
        models = [
            LinMAB(FreqPaperLinModel(u, n_features, n_arms, sigma=10))
            for _ in range(n_expe)
        ]
        log = False
        title = "Linear Gaussian Model (Freq MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    elif problem == "Russo":
        u = 1 / np.sqrt(5)
        models = [
            LinMAB(PaperLinModel(u, n_features, n_arms, sigma=10))
            for _ in range(n_expe)
        ]
        log = False
        title = "Linear Gaussian Model (Bayes MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    else:
        raise NotImplementedError
    # if track_ids:
    #     for m in models:
    #         m.store_IDS = True
    print("Begin experiments on '{}'".format(title))
    results = storeRegret(models, methods, param_dic, n_expe, T, path)
    if doplot:
        plotRegret(labels, results, colors, title, path, log=log)
    # if track_ids:
    #     plot_IDS_results(T, n_expe, results["IDS_results"])
    return results


def HyperMAB_expe(
    n_expe,
    n_features,
    n_arms,
    T,
    methods,
    param_dic,
    labels,
    colors,
    path,
    problem="FreqRusso",
    doplot=True,
    # track_ids=False,
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Linear Gaussian Bandits with multivariate Gaussian prior
    :param n_expe: int, number of experiments
    :param n_features: int, dimension of feature vectors
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :param problem: str, choose from {'FreqRusso', 'Zhang', 'Russo', 'movieLens'}
    :param path: str
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    from agent.hyper import HyperMAB

    if problem == "movieLens":
        models = [HyperMAB(ColdStartMovieLensModel()) for _ in range(n_expe)]
        log = True
        title = "Movie recommendation system - n_arm: 207 - n_features: 30"
    elif problem == "Zhang":
        models = [HyperMAB(FGTSLinModel(n_features, n_arms)) for _ in range(n_expe)]
        log = False
        title = (
            "Linear Gaussian Model (Zhang, 2021) - n_arms: {} - n_features: {}".format(
                n_arms, n_features
            )
        )
    elif problem == "FreqRusso":
        u = 1 / np.sqrt(5)
        models = [
            HyperMAB(FreqPaperLinModel(u, n_features, n_arms, sigma=10))
            for _ in range(n_expe)
        ]
        log = False
        title = "Linear Gaussian Model (Freq MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    elif problem == "Russo":
        u = 1 / np.sqrt(5)
        models = [
            HyperMAB(PaperLinModel(u, n_features, n_arms, sigma=10))
            for _ in range(n_expe)
        ]
        log = False
        title = "Linear Gaussian Model (Bayes MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    else:
        raise NotImplementedError
    # if track_ids:
    #     for m in models:
    #         m.store_IDS = True
    print("Begin experiments on '{}'".format(title))
    results = storeRegret(models, methods, param_dic, n_expe, T, path, use_torch=True)
    if doplot:
        plotRegret(labels, results, colors, title, path, log=log)
    # if track_ids:
    #     plot_IDS_results(T, n_expe, results["IDS_results"])
    return results


def FiniteContextHyperMAB_expe(
    n_expe,
    n_context,
    n_features,
    n_arms,
    T,
    methods,
    param_dic,
    labels,
    colors,
    path,
    problem="FreqRusso",
    doplot=True,
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Linear Gaussian Bandits with multivariate Gaussian prior
    :param n_expe: int, number of experiments
    :param n_features: int, dimension of feature vectors
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :param problem: str, choose from {'FreqRusso', 'Zhang', 'Russo', 'movieLens'}
    :param path: str
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    from agent.hyper import HyperMAB

    if problem == "Zhang":
        models = [
            HyperMAB(FiniteContextFGTSLinModel(n_context, n_features, n_arms))
            for _ in range(n_expe)
        ]
        title = (
            "Linear Gaussian Model (Zhang, 2021) - n_arms: {} - n_features: {}".format(
                n_arms, n_features
            )
        )
    elif problem == "FreqRusso":
        u = 1 / np.sqrt(5)
        models = [
            HyperMAB(
                FiniteContextFreqPaperLinModel(
                    u, n_context, n_features, n_arms, sigma=10
                )
            )
            for _ in range(n_expe)
        ]
        title = "Linear Gaussian Model (Freq MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    elif problem == "Russo":
        u = 1 / np.sqrt(5)
        models = [
            HyperMAB(
                FiniteContextPaperLinModel(u, n_context, n_features, n_arms, sigma=10)
            )
            for _ in range(n_expe)
        ]
        title = "Linear Gaussian Model (Bayes MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    elif problem == "Synthetic-v1":
        models = [
            HyperMAB(SyntheticNonlinModel(n_features, n_arms, reward_version="v1"))
            for _ in range(n_expe)
        ]
        title = f"Synthetic Bandit Model  - n_arms: {n_arms} - n_features: {n_features} - reward: v1"
    elif problem == "Synthetic-v2":
        models = [
            HyperMAB(SyntheticNonlinModel(n_features, n_arms, reward_version="v2"))
            for _ in range(n_expe)
        ]
        title = f"Synthetic Bandit Model  - n_arms: {n_arms} - n_features: {n_features} - reward: v2"
    elif problem == "Synthetic-v3":
        models = [
            HyperMAB(SyntheticNonlinModel(n_features, n_arms, reward_version="v3"))
            for _ in range(n_expe)
        ]
        title = f"Synthetic Bandit Model  - n_arms: {n_arms} - n_features: {n_features} - reward: v3"
    else:
        raise NotImplementedError

    print("Begin experiments on '{}'".format(title))
    results = storeRegret(models, methods, param_dic, n_expe, T, path, use_torch=True)
    if doplot:
        plotRegret(labels, results, colors, title, path, log=False)
    return results


def InfiniteContextHyperMAB_expe(
    n_expe,
    n_features,
    n_arms,
    T,
    methods,
    param_dic,
    labels,
    colors,
    path,
    problem="FreqRusso",
    doplot=True,
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with n_arms Linear Gaussian Bandits with multivariate Gaussian prior
    :param n_expe: int, number of experiments
    :param n_features: int, dimension of feature vectors
    :param n_arms: int, number of arms
    :param T: int, time horizon
    :param methods: list, algorithms to use
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param doplot: boolean, plot the curves or not
    :param problem: str, choose from {'FreqRusso', 'Zhang', 'Russo', 'movieLens'}
    :param path: str
    :return: dict, regrets, quantiles, means, stds of final regrets for each methods
    """
    from agent.hyper import HyperMAB

    if problem == "FreqRusso":
        u = 1 / np.sqrt(5)
        models = [
            HyperMAB(InfiniteContextFreqPaperLinModel(u, n_features, n_arms, sigma=10))
            for _ in range(n_expe)
        ]
        title = "Linear Gaussian Model (Freq MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    elif problem == "Russo":
        u = 1 / np.sqrt(5)
        models = [
            HyperMAB(InfiniteContextPaperLinModel(u, n_features, n_arms, sigma=10))
            for _ in range(n_expe)
        ]
        title = "Linear Gaussian Model (Bayes MOD, Russo and Van Roy, 2018) - n_arms: {} - n_features: {}".format(
            n_arms, n_features
        )
    else:
        raise NotImplementedError

    print("Begin experiments on '{}'".format(title))
    results = storeRegret(models, methods, param_dic, n_expe, T, path, use_torch=True)
    if doplot:
        plotRegret(labels, results, colors, title, path, log=False)
    return results


def finite_expe(
    methods, labels, colors, param_dic, prior, q, R, theta, N, T, problem="Finite"
):
    """
    Compute regrets for a given set of algorithms (methods) over t=1,...,T and for n_expe number of independent
    experiments. Here we deal with Finite Set Problems
    :param methods: list, algorithms to use
    :param labels: list, labels for the curves
    :param colors: list, colors for the curves
    :param param_dic: dict, parameters associated to each algorithm (see main for formatting)
    :param prior: np.array, prior distribution on theta
    :param q: np.array, L*K*N array with the probability of each outcome knowing theta
    :param R: np.array, mapping between outcomes and rewards
    :param theta: float, true theta
    :param N: int, number of possible rewards
    :param T: int, time horizon
    """
    nb_arms, nb_rewards = q.shape[1:3]
    p2 = [[R, q[theta, i, :]] for i in range(q.shape[1])]
    check_MAB = GenericMAB([problem] * nb_arms, p2)
    plt.figure(1)
    for i, m in enumerate(methods):
        c = cmap[i] if not colors else colors[i]
        r = check_MAB.MC_regret(m, N, T, param_dic)
        plt.plot(r, label=labels[i], c=c)
    p = [[np.arange(nb_rewards), q[theta, i, :]] for i in range(nb_arms)]
    my_MAB = FiniteSets(["F"] * nb_arms, p, q, prior, R)
    regret_IDS = my_MAB.MC_regret("IDS", N, T, param_dic)
    plt.plot(regret_IDS, label="IDS", c="cyan")
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Time horizon")
    plt.grid(color="grey", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


def Finite_Bernoulli(n_expe, nb_arms, T, M, colors, problem="Finite", doplot=False):
    """
    Run Finite Sets on IDS on Bernoulli Bandits using M samples of nb_arms dimensional
    uniformly sampled parameters.
    :param n_expe: number of trials
    :param nb_arms: number of arms in the Multi Armed Bandit
    :param T: Time horizon
    :param M: Number of Samples for the parameter space
    :param colors: colors for plotting the mean regret
    :param doplot: Boolean
    :return: Dict with results for tested algorithms
    """
    theta = np.random.uniform(0, 1, size=nb_arms * n_expe).reshape(n_expe, nb_arms)
    true_param = [
        [
            [np.array([0, 1]), np.array([theta[i, j], 1 - theta[i, j]])]
            for j in range(nb_arms)
        ]
        for i in range(n_expe)
    ]
    prior, q, R = build_bernoulli_finite_set(M, nb_arms)
    all_regrets = np.empty((n_expe, T))
    for i in tqdm(range(n_expe)):
        my_MAB = FiniteSets([problem] * nb_arms, true_param[i], q, prior, R)
        all_regrets[i] = my_MAB.regret(my_MAB.IDS(T)[0], T)
    mean_regret = all_regrets.mean(axis=0).reshape((1, T))
    quantiles = {"Finite IDS": np.quantile(mean_regret, np.arange(0, 1, 21))}
    means = {"Finite IDS": mean_regret[-1]}
    std = {"Finite IDS": all_regrets.std(axis=0).reshape((1, T))}
    if doplot:
        plotRegret(
            ["IDS with fixed parameter sample"], mean_regret, colors, "Binary rewards"
        )
    return {
        "all_regrets": all_regrets,
        "quantiles": quantiles,
        "means": means,
        "std": std,
    }
