import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _random_argmax(vals: np.ndarray, scale: float = 1e-7):
    """Select max with additional random noise."""
    noise = np.random.uniform(vals.shape)
    index = np.argmax(vals + scale * noise, axis=-1)
    return vals[np.arange(vals.shape[0]), index]


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
        if len(hidden_sizes) > 0:
            hidden_sizes = [input_dim] + list(hidden_sizes)
            for i in range(1, len(hidden_sizes)):
                model += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])]
                model += [nn.ReLU(inplace=True)]
            model += [nn.Linear(hidden_sizes[-1], output_dim)]
        self.model = nn.Sequential(*model)
        self.reset_parameters()

        self.temperature = temperature
        self.output_dim = output_dim
        self.device = device

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        logits = self.model(x)
        if self.output_dim == 1:
            return logits.detach().cpu().numpy()
        probs = F.softmax(logits / self.temperature, -1)
        out = _random_argmax(probs.detach().cpu().numpy())
        return out


class SyntheticNonlinModel:
    def __init__(
        self,
        n_features=50,
        n_actions=20,
        all_actions=1000,
        eta=0.1,
        sigma=1,
        reward_version="v1",
        freq_task=True,
        resample_feature=False,
    ):
        prior_random_state = 2022 if freq_task else np.random.randint(1, 312414)
        reward_random_state = np.random.randint(1, 312414)
        self.prior_random = np.random.RandomState(prior_random_state)
        self.reward_random = np.random.RandomState(reward_random_state)

        self.n_actions = n_actions
        self.n_features = n_features
        self.sub_actions = n_actions
        self.all_actions = all_actions

        # feture
        self.set_feature()

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
            self.set_reward_model(n_features, 2, prior_random_state)
            self.reward_fn = getattr(self, "reward_fn3")
        elif reward_version == "v4":
            self.set_reward_model(n_features, 1, prior_random_state)
            self.reward_fn = getattr(self, "reward_fn4")
        else:
            raise NotImplementedError
        self.set_reward()

        self.eta = eta
        self.alg_prior_sigma = sigma
        self.resample_feature = resample_feature
        self.set_context()

    # @property
    # def n_features(self):
    #     return self.features.shape[1]

    # @property
    # def n_actions(self):
    #     return self.features.shape[0]

    def set_feature(self):
        x = self.prior_random.randn(self.all_actions, self.n_features).astype(
            np.float32
        )
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        self.all_features = x

    def set_reward(self):
        self.all_rewards = self.reward_fn(self.all_features).astype(np.float32)

    def set_context(self):
        if self.resample_feature:
            self.set_feature()
            self.set_reward()
        if self.sub_actions == self.all_actions:
            sub_action_set = np.arange(self.sub_actions)
        else:
            action_set = np.arange(self.all_actions, dtype=np.int32)
            sub_action_set = self.prior_random.choice(
                action_set, size=self.sub_actions, replace=False
            )
        self.features = self.all_features[sub_action_set]
        self.sub_rewards = self.all_rewards[sub_action_set]

    def set_reward_model(self, input_dim, output_dim, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward_model = MLP(input_dim, output_dim, device=device).to(device)
        print(str(self.reward_model))

    def reward_fn1(self, feature):
        reward = np.diagonal(0.01 * feature @ self.real_theta @ feature.T)
        return reward

    def reward_fn2(self, feature):
        reward = np.exp(-10 * (feature @ self.real_theta) ** 2)
        return reward

    def reward_fn3(self, feature):
        prob = self.reward_model(feature)
        reward = self.prior_random.binomial(1, prob)
        return reward

    def reward_fn4(self, feature):
        reward = self.reward_model(feature)
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
