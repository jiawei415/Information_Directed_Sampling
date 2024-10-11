import numpy as np


class ArmGaussianLinear(object):
    def __init__(self, prior_random_state=2022, reward_random_state=2023):
        self.prior_random = np.random.RandomState(prior_random_state)
        self.reward_random = np.random.RandomState(reward_random_state)

    def reward(self, arm):
        """
        Pull 'arm' and get the reward drawn from a^T . theta + epsilon with epsilon following N(0, eta)
        :param arm: int
        :return: float
        """
        return np.dot(self.features[arm], self.real_theta) + self.reward_random.normal(
            0, self.eta, 1
        )

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]

    def set_context(self):
        pass

    def regret(self, reward, T):
        """
        Compute the regret of a single experiment
        :param reward: np.array, the array of reward obtained from the policy up to time T
        :param T: int, time horizon
        :return: np.array, cumulative regret for a single experiment
        """
        best_arm_reward = np.max(np.dot(self.features, self.real_theta))
        return best_arm_reward * np.arange(1, T + 1) - np.cumsum(reward)

    def expect_regret(self, arm, action_set):
        """
        Compute the regret of a single step
        """
        arm_reward = np.dot(action_set, self.real_theta)
        expect_reward = arm_reward[arm]
        best_arm_reward = arm_reward.max()
        return best_arm_reward - expect_reward


class PaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_features, n_actions, eta=1, sigma=10):
        """
        Initialization of the arms, features and theta in
        Russo, Daniel, and Benjamin Van Roy. "Learning to optimize via information-directed sampling." Operations Research 66.1 (2018): 230-252.
        :param u: float, features are drawn from a uniform Unif(-u, u)
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward N(a^T.theta, eta)
        :param sigma: float, multiplicative factor for the covariance matrix of theta which is drawn from a
        multivariate distribution N(0, sigma*I)
        """
        super(PaperLinModel, self).__init__(
            prior_random_state=np.random.randint(1, 312414),
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = self.prior_random.uniform(-u, u, (n_actions, n_features))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma


class FreqPaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_features, n_actions, eta=10, sigma=10):
        """
        (Frequentist modification: use fixed random seed to sample arms, features and theta.)
        Initialization of the arms, features and theta in
        Russo, Daniel, and Benjamin Van Roy. "Learning to optimize via information-directed sampling." Operations Research 66.1 (2018): 230-252.
        :param u: float, features are drawn from a uniform Unif(-u, u)
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward N(a^T.theta, eta)
        :param sigma: float, multiplicative factor for the covariance matrix of theta which is drawn from a
        multivariate distribution N(0, sigma*I)
        """
        super(FreqPaperLinModel, self).__init__(
            prior_random_state=2022,
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = self.prior_random.uniform(-u, u, (n_actions, n_features))
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma


class FGTSLinModel(ArmGaussianLinear):
    def __init__(self, n_features=100, n_actions=2, eta=np.sqrt(0.5)):
        """
        Initialization of the arms, features and theta in
        Zhang, Tong. "Feel-Good Thompson Sampling for Contextual Bandits and Reinforcement Learning." arXiv preprint arXiv:2110.00871 (2021).
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward likelihood model N(a^T.theta, eta)
        """
        super(FGTSLinModel, self).__init__(
            prior_random_state=2022,
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = np.zeros((n_actions, n_features))
        self.features[0, 0] = 1.0
        self.features[1, 1] = 0.2
        if n_actions > 2:
            self.features[2:] = self.prior_random.uniform(-1, 1, (n_actions-2, n_features)) 
            self.features[2:] = 0.2 * self.features[2:] / np.expand_dims(np.linalg.norm(self.features[2:], axis=1), axis=1)
            # print(self.features)
            # print(np.linalg.norm(self.features, axis=1))
        self.features[np.where(self.features == 0)] = 1e-6
        # print(self.features)
        self.real_theta = np.zeros(n_features)
        self.real_theta[0:2] = 1.0
        self.alg_prior_sigma = 0.01

    def reward(self, arm):
        """
        Pull 'arm' and get the reward drawn from a^T . theta + epsilon with epsilon following Unif(-0.5, 0.5)
        :param arm: int
        :return: float
        """
        return np.dot(self.features[arm], self.real_theta) + self.reward_random.uniform(
            -0.5, 0.5, 1
        )


class ColdStartMovieLensModel(ArmGaussianLinear):
    def __init__(self, n_features=30, n_actions=207, eta=1, sigma=10):
        super(ColdStartMovieLensModel, self).__init__(
            prior_random_state=np.random.randint(1, 312414),
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.features = np.loadtxt("Data/Vt.csv", delimiter=",").T
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma
