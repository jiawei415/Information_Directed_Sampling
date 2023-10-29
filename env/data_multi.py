from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np

DATA_NAMES = {"v1": "mnist", "v2": "mushroom", "v3": "adult", "v4": "covertype"}


class Bandit_multi:
    def __init__(self, name_id, is_shuffle=True, eta=0.1, sigma=1):
        name = DATA_NAMES[name_id]
        prior_random_state = 2022
        reward_random_state = np.random.randint(1, 312414)
        self.prior_random = np.random.RandomState(prior_random_state)
        self.reward_random = np.random.RandomState(reward_random_state)
        # Fetch data
        if name == "mnist":
            X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = -1
            X = normalize(X)
        elif name == "mushroom":
            X, y = fetch_openml("mushroom", version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = -1
            X = normalize(X)
        elif name == "adult":
            X, y = fetch_openml("adult", version=2, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = -1
            X = normalize(X)
        elif name == "covertype":
            X, y = fetch_openml("covertype", version=3, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = -1
            X = normalize(X)
        elif name == "isolet":
            X, y = fetch_openml("isolet", version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = -1
            X = normalize(X)
        elif name == "letter":
            X, y = fetch_openml("letter", version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = -1
            X = normalize(X)
        elif name == "MagicTelescope":
            X, y = fetch_openml("MagicTelescope", version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = -1
            X = normalize(X)
        elif name == "shuttle":
            X, y = fetch_openml("shuttle", version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = -1
            X = normalize(X)
        else:
            raise RuntimeError("Dataset does not exist")
        # Shuffle data
        if is_shuffle:
            self.X, self.y = shuffle(X, y, random_state=prior_random_state)
        else:
            self.X, self.y = X, y
        # generate one_hot coding:
        self.y_arm = OrdinalEncoder(dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        self.n_arm = np.max(self.y_arm) + 1
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]
        self.eta = eta
        self.alg_prior_sigma = sigma
        self.features = None

    @property
    def n_features(self):
        return self.dim

    @property
    def n_actions(self):
        return self.n_arm

    def set_context(self):
        # assert self.cursor < self.size
        if self.cursor >= self.size:
            self.cursor = 0
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim : a * self.act_dim + self.act_dim] = self.X[
                self.cursor
            ]
        arm = self.y_arm[self.cursor][0]
        rwd = np.zeros((self.n_arm,))
        rwd[arm] = 1
        self.cursor += 1
        self.features = X
        self.sub_rewards = rwd

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


if __name__ == "__main__":
    b = Bandit_multi("mushroom")
    x_y, a = b.step()
    # print(x_y[0])
    # print(x_y[1])
    # print(np.linalg.norm(x_y[0]))
