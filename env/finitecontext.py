""" Packages import """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import rd_max, sigmoid

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=2,
        hidden_sizes=[50, 50],
        temperature=1.0,
        device="cpu",
    ):
        super().__init__()
        model = []
        if len(hidden_sizes) > 0 :
            hidden_sizes = [input_dim] + list(hidden_sizes)
            for i in range(1, len(hidden_sizes)):
                model += [nn.Linear(hidden_sizes[i-1], hidden_sizes[i])]
                model += [nn.ReLU(inplace=True)]
            model += [nn.Linear(hidden_sizes[-1], output_dim)]
        self.model = nn.Sequential(*model)
        self.reset_parameters()

        self.temperature = temperature
        self.device = device

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits / self.temperature, -1)
        out = rd_max(probs.detach().cpu().numpy())
        return out

class ArmGaussianLinear(object):
    def __init__(self, n_context, prior_random_state=2022, reward_random_state=2023):
        self.prior_random = np.random.RandomState(prior_random_state)
        self.reward_random = np.random.RandomState(reward_random_state)
        self.n_context = n_context

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
        context = np.random.randint(self.n_context)
        self.features = self.all_features[context]

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


class FiniteContextPaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_context, n_features, n_actions, eta=1, sigma=10):
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
        super(FiniteContextPaperLinModel, self).__init__(
            n_context,
            prior_random_state=np.random.randint(1, 312414),
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.all_features = self.prior_random.uniform(
            -u, u, (n_context, n_actions, n_features)
        )
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma
        self.set_context()


class FiniteContextFreqPaperLinModel(ArmGaussianLinear):
    def __init__(self, u, n_context, n_features, n_actions, eta=1, sigma=10):
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
        super(FiniteContextFreqPaperLinModel, self).__init__(
            n_context,
            prior_random_state=2022,
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.all_features = self.prior_random.uniform(
            -u, u, (n_context, n_actions, n_features)
        )
        self.real_theta = self.prior_random.multivariate_normal(
            np.zeros(n_features), sigma * np.eye(n_features)
        )
        self.alg_prior_sigma = sigma
        self.set_context()


class FiniteContextFGTSLinModel(ArmGaussianLinear):
    def __init__(self, n_context, n_features=100, n_actions=2, eta=np.sqrt(0.5)):
        """
        Initialization of the arms, features and theta in
        Zhang, Tong. "Feel-Good Thompson Sampling for Contextual Bandits and Reinforcement Learning." arXiv preprint arXiv:2110.00871 (2021).
        :param n_features: int, dimension of the feature vectors
        :param n_actions: int, number of actions
        :param eta: float, std from the reward likelihood model N(a^T.theta, eta)
        """
        super(FiniteContextFGTSLinModel, self).__init__(
            n_context,
            prior_random_state=2022,
            reward_random_state=np.random.randint(1, 312414),
        )
        self.eta = eta
        self.all_features = np.zeros((n_context, n_actions, n_features))
        self.all_features[:, 0, 0] = 1.0
        self.all_features[:, 1, 1] = 0.2
        if n_actions > 2:
            self.all_features[:, 2:] = self.prior_random.uniform(
                -1, 1, (n_actions - 2, n_features)
            )
            self.all_features[:, 2:] = (
                0.2
                * self.all_features[:, 2:]
                / np.expand_dims(
                    np.linalg.norm(self.all_features[:, 2:], axis=1), axis=1
                )
            )
            # print(self.features)
            # print(np.linalg.norm(self.features, axis=1))
        self.all_features[np.where(self.all_features == 0)] = 1e-6
        # print(self.features)
        self.real_theta = np.zeros(n_features)
        self.real_theta[0:2] = 1.0
        self.alg_prior_sigma = 0.01
        self.set_context()

    def reward(self, arm):
        """
        Pull 'arm' and get the reward drawn from a^T . theta + epsilon with epsilon following Unif(-0.5, 0.5)
        :param arm: int
        :return: float
        """
        return np.dot(self.features[arm], self.real_theta) + self.reward_random.uniform(
            -0.5, 0.5, 1
        )


class SyntheticNonlinModel:
    def __init__(
        self,
        n_features=50,
        n_actions=20,
        all_actions=100,
        eta=0.1,
        sigma=1,
        reward_version="v1",
    ):
        prior_random_state = 2022
        reward_random_state = np.random.randint(1, 312414)
        self.prior_random = np.random.RandomState(prior_random_state)
        self.reward_random = np.random.RandomState(reward_random_state)

        # feture
        x = self.prior_random.randn(all_actions, n_features).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        self.all_features = x

        # reward
        if reward_version == "v1":
            self.reward_fn = getattr(self, "reward_fn1")
            theta = self.prior_random.normal(0, sigma, size=(n_features, n_features))
            self.real_theta = theta @ theta.T
        elif reward_version == "v2":
            self.reward_fn = getattr(self, "reward_fn2")
            theta = self.prior_random.normal(0, sigma, size=n_features)
            self.real_theta = theta / np.linalg.norm(theta)
        elif reward_version == "v3":
            self.set_reward_model(n_features, prior_random_state)
            self.reward_fn = getattr(self, "reward_fn3")
        else:
            raise NotImplementedError
        self.all_rewards = np.array(
            [self.reward_fn(self.all_features[arm]) for arm in range(all_actions)], dtype=np.float32
        )

        self.sub_actions = n_actions
        self.all_actions = all_actions
        self.eta = eta
        self.alg_prior_sigma = sigma
        self.set_context()

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]

    def set_context(self):
        action_set = np.arange(self.all_actions, dtype=np.int32)
        sub_action_set = self.prior_random.choice(
            action_set, size=self.sub_actions, replace=False
        )
        self.features = self.all_features[sub_action_set]
        self.sub_rewards = self.all_rewards[sub_action_set]

    def set_reward_model(self, input_dim, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reward_model = MLP(input_dim, device=device).to(device)

    def reward_fn1(self, feature):
        logist = 0.01 * feature.T @ self.real_theta @ feature
        prob = sigmoid(logist)
        reward = self.prior_random.binomial(1, prob)
        return reward

    def reward_fn2(self, feature):
        logist = np.exp(-10 * np.dot(feature, self.real_theta) ** 2)
        prob = sigmoid(logist)
        reward = self.prior_random.binomial(1, prob)
        return reward

    def reward_fn3(self, feature):
        prob = self.reward_model(feature)
        reward = self.prior_random.binomial(1, prob)
        return reward

    def reward(self, arm):
        reward = self.sub_rewards[arm]
        noise = self.reward_random.normal(0, self.eta, 1)
        return reward + noise

    def regret(self, arm):
        expect_reward = self.sub_rewards[arm]
        best_arm_reward = self.sub_rewards.max()
        return best_arm_reward - expect_reward

    def expect_regret(self, arm, features):
        """
        Compute the regret of a single step
        """
        expect_reward = self.sub_rewards[arm]
        best_arm_reward = self.sub_rewards.max()
        return best_arm_reward - expect_reward
