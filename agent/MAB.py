""" Packages import """
import numpy as np
from env import arms
from utils import rd_argmax
from tqdm import tqdm
import inspect

mapping = {
    "Bernoulli": arms.ArmBernoulli,
    "FreqBernoulli": arms.ArmBernoulli,
    "Beta": arms.ArmBeta,
    "Finite": arms.ArmFinite,
    "Gaussian": arms.ArmGaussian,
    "FreqGaussian": arms.ArmGaussian,
}


class GenericMAB:
    """
    Generic class for arms that defines general methods
    """

    def __init__(self, envs, frequentist=False, p=None):
        """
        Initialization of the arms
        :param envs: string, probability distribution of each arm
        :param p: np.array or list, parameters of the probability distribution of each arm
        :param frequentist: bool, frequentist env
        """
        self.MAB = self.generate_arms(envs, frequentist, p)
        self.nb_arms = len(self.MAB)
        self.means = [el.mean for el in self.MAB]
        self.mu_max = np.max(self.means)
        self.IDS_results = {"arms": [], "policy": [], "delta": [], "g": [], "IR": []}
        self.store_IDS = False

    @staticmethod
    def generate_arms(envs, frequentist, p):
        """
        Method for generating different arms
        :param envs: string, probability distribution of each arm
        :param frequentist: bool, frequentist env
        :return: list of class objects, list of arms
        """
        arms_list = list()
        for i, e in enumerate(envs):
            if p is None:
                args = [frequentist, np.random.randint(1, 312414)]
            else:
                args = [p[i]] + [[np.random.randint(1, 312414)]]
                args = sum(args, []) if type(p[i]) == list else args
            try:
                alg = mapping[e]
                arms_list.append(alg(*args))
            except Exception:
                raise NotImplementedError
        return arms_list

    def regret(self, reward, T):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        return self.mu_max * np.arange(1, T + 1) - np.cumsum(reward)

    def expect_regret(self, reward):
        """
        Compute the regret of a single step
        """
        return self.mu_max - reward

    def MC_regret(self, method, N, T, param_dic):
        """
        Implementation of Monte Carlo method to approximate the expectation of the regret
        :param method: string, method used (UCB, Thomson Sampling, etc..)
        :param N: int, number of independent Monte Carlo simulation
        :param T: int, time horizon
        :param param_dic: dict, parameters for the different methods, can be the value of rho for UCB model or an int
        corresponding to the number of rounds of exploration for the ExploreCommit method
        """
        mc_regret = np.zeros(T)
        try:
            alg = self.__getattribute__(method)
            args = inspect.getfullargspec(alg)[0][2:]
            args = [T] + [param_dic[method][i] for i in args]
            for _ in tqdm(range(N), desc="Computing " + str(N) + " simulations"):
                mc_regret += self.regret(alg(*args)[0], T)
        except Exception:
            raise NotImplementedError
        return mc_regret / N

    def init_lists(self, T):
        """
        Initialization of quantities of interest used for all methods
        :param T: int, time horizon
        :return: - Sa: np.array, cumulative reward of arm a
                 - Na: np.array, number of times a has been pulled
                 - reward: np.array, rewards
                 - arm_sequence: np.array, arm chose at each step
        """
        Sa, Na, reward, arm_sequence, expected_regret = (
            np.zeros(self.nb_arms),
            np.zeros(self.nb_arms),
            np.zeros(T),
            np.zeros(T),
            np.zeros(T),
        )
        return Sa, Na, reward, arm_sequence, expected_regret

    def update_lists(self, t, arm, Sa, Na, reward, arm_sequence, expected_regret):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Sa:  np.array, cumulative reward array up to time t-1
        :param Na:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        Na[arm], arm_sequence[t], new_reward = Na[arm] + 1, arm, self.MAB[arm].sample()
        reward[t], Sa[arm] = new_reward, Sa[arm] + new_reward
        expected_regret[t] = self.mu_max - new_reward
        return new_reward

    def IDSAction(self, delta, g):
        """
        Implementation of IDSAction algorithm as defined in Russo & Van Roy, p. 242
        :param delta: np.array, instantaneous regrets
        :param g: np.array, information gains
        :return: int, arm to pull
        """
        Q = np.zeros((self.nb_arms, self.nb_arms))
        IR = np.ones((self.nb_arms, self.nb_arms)) * np.inf
        q = np.linspace(0, 1, 1000)
        for a in range(self.nb_arms - 1):
            for ap in range(a + 1, self.nb_arms):
                if g[a] < 1e-6 or g[ap] < 1e-6:
                    return rd_argmax(-g)
                da, dap, ga, gap = delta[a], delta[ap], g[a], g[ap]
                qaap = q[
                    rd_argmax(
                        -((q * da + (1 - q) * dap) ** 2) / (q * ga + (1 - q) * gap)
                    )
                ]
                IR[a, ap] = (qaap * (da - dap) + dap) ** 2 / (qaap * (ga - gap) + gap)
                Q[a, ap] = qaap
        amin = rd_argmax(-IR.reshape(self.nb_arms * self.nb_arms))
        a, ap = amin // self.nb_arms, amin % self.nb_arms
        b = np.random.binomial(1, Q[a, ap])
        arm = int(b * a + (1 - b) * ap)
        if self.store_IDS:
            self.IDS_results["arms"].append(arm)
            policy = np.zeros(self.nb_arms)
            policy[a], policy[ap] = Q[a, ap], (1 - Q[a, ap])
            self.IDS_results["policy"].append(policy)
            self.IDS_results["delta"].append(delta)
            self.IDS_results["g"].append(g)
            self.IDS_results["IR"].append(
                np.inner(delta**2, policy) / np.inner(g, policy)
            )
        return arm
