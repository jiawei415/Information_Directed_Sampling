import numpy as np

from utils import rd_argmax
from agent.hypermodel import HyperModel


class HyperMAB:
    def __init__(self, env):
        self.env = env
        self.expect_regret, self.n_a, self.d, self.features = (
            env.expect_regret,
            env.n_actions,
            env.n_features,
            env.features,
        )
        self.reward, self.eta = env.reward, env.eta
        self.prior_sigma = env.alg_prior_sigma
        self.threshold = 0.999
        self.store_IDS = False

    def set_context(self):
        self.env.set_context()
        self.features = self.env.features

    def initPrior(self):
        a0 = 0
        s0 = self.prior_sigma
        mu_0 = a0 * np.ones(self.d)
        sigma_0 = s0 * np.eye(
            self.d
        )  # to adapt according to the true distribution of theta
        return mu_0, sigma_0

    def updatePosterior(self, a, mu, sigma):
        """
        Update posterior mean and covariance matrix
        :param arm: int, arm chose
        :param mu: np.array, posterior mean vector
        :param sigma: np.array, posterior covariance matrix
        :return: float and np.arrays, reward obtained with arm a, updated means and covariance matrix
        """
        f, r = self.features[a], self.reward(a)[0]
        s_inv = np.linalg.inv(sigma)
        ffT = np.outer(f, f)
        mu_ = np.dot(
            np.linalg.inv(s_inv + ffT / self.eta**2),
            np.dot(s_inv, mu) + r * f / self.eta**2,
        )
        sigma_ = np.linalg.inv(s_inv + ffT / self.eta**2)
        return r, mu_, sigma_

    def TS(self, T):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward and regret obtained by the policy
        """
        reward, expected_regret = np.zeros(T), np.zeros(T)
        mu_t, sigma_t = self.initPrior()
        for t in range(T):
            self.set_context()
            theta_t = np.random.multivariate_normal(mu_t, sigma_t, 1).T
            a_t = rd_argmax(np.dot(self.features, theta_t))
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def TS_hyper(
        self,
        T,
        file,
        noise_dim=2,
        fg_lambda=1.0,
        fg_decay=True,
        lr=0.01,
        batch_size=32,
        hidden_sizes=(),
        optim="Adam",
        update_num=2,
        reset=False,
    ):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward and regret obtained by the policy
        """
        norm_coef = (self.eta / self.prior_sigma) ** 2
        model = HyperModel(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_std=self.prior_sigma,
            fg_lambda=fg_lambda,
            fg_decay=fg_decay,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            target_noise_coef=self.eta,
            norm_coef=norm_coef,
            buffer_size=T,
            reset=reset,
        )

        reward, expected_regret = np.zeros(T), np.zeros(T)
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)[0]
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, expected_regret

    def computeVIDS_v1(self, thetas):
        """
        Implementation of linearSampleVIR (algorithm 6 in Russo & Van Roy, p. 244) applied for Linear  Bandits with
        multivariate normal prior. Here integrals are approximated in sampling thetas according to their respective
        posterior distributions.
        :param thetas: np.array, posterior samples
        :return: int, np.array, delta, v and p*
        """
        # print(thetas.shape)
        M = thetas.shape[0]
        mu = np.mean(thetas, axis=0)
        theta_hat = np.argmax(np.dot(self.features, thetas.T), axis=0)
        # print("theta_hat shape: {}".format(theta_hat.shape))
        theta_hat_ = [thetas[np.where(theta_hat == a)] for a in range(self.n_a)]
        p_a = np.array([len(theta_hat_[a]) for a in range(self.n_a)]) / M
        mu_a = np.nan_to_num(
            np.array(
                [np.mean([theta_hat_[a]], axis=1).squeeze() for a in range(self.n_a)]
            )
        )
        L_hat = np.sum(
            np.array(
                [p_a[a] * np.outer(mu_a[a] - mu, mu_a[a] - mu) for a in range(self.n_a)]
            ),
            axis=0,
        )
        rho_star = np.sum(
            np.array(
                [p_a[a] * np.dot(self.features[a], mu_a[a]) for a in range(self.n_a)]
            ),
            axis=0,
        )
        v = np.array(
            [
                np.dot(np.dot(self.features[a], L_hat), self.features[a].T)
                for a in range(self.n_a)
            ]
        )
        delta = np.array(
            [rho_star - np.dot(self.features[a], mu) for a in range(self.n_a)]
        )
        return delta, v, p_a

    def computeVIDS_v2(self, value):
        value_gap = np.max(value, axis=-1, keepdims=True) - value
        delta = value_gap.mean(axis=0)

        value_max_index = np.argmax(value, axis=-1)
        z_a = [np.where(value_max_index == a)[0] for a in range(self.n_a)]
        p_a = np.array([len(z_a[a]) for a in range(self.n_a)]) / value.shape[0]

        E = value.mean(0)
        E_a = np.nan_to_num(
            np.array([value[z_a[a]].mean(axis=0) for a in range(self.n_a)])
        )

        v = np.dot(p_a, (E_a - E) ** 2)
        return delta, v, p_a

    def computeVIDS_v3(self, value):
        value_gap = np.max(value, axis=-1, keepdims=True) - value
        delta = value_gap.mean(axis=0)
        v = np.var(value, axis=0)
        return delta, v

    def vids_sample_by_action(self, delta, v):
        arm = rd_argmax(-(delta**2) / (v + 1e-20))
        return arm

    def vids_sample_by_policy(self, delta, v):
        prob = np.zeros(shape=(self.n_a, self.n_a))
        psi = np.ones(shape=(self.n_a, self.n_a)) * np.inf
        for i in range(self.n_a - 1):
            for j in range(i + 1, self.n_a):
                if delta[j] < delta[i]:
                    D1, D2, I1, I2, flip = delta[j], delta[i], v[j], v[i], True
                else:
                    D1, D2, I1, I2, flip = delta[i], delta[j], v[i], v[j], False
                p = (
                    np.clip((D1 / (D2 - D1)) - (2 * I1 / (I2 - I1)), 0.0, 1.0)
                    if I1 < I2
                    else 0.0
                )
                psi[i][j] = ((1 - p) * D1 + p * D2) ** 2 / (
                    (1 - p) * I1 + p * I2 + 1e-20
                )
                prob[i][j] = 1 - p if flip else p
        psi = psi.flatten()
        optim_indexes = np.nonzero(psi == psi.min())[0].tolist()
        optim_index = np.random.choice(optim_indexes)
        optim_index = [optim_index // self.n_a, optim_index % self.n_a]
        optim_prob = prob[optim_index[0], optim_index[1]]
        arm = np.random.choice(optim_index, p=[1 - optim_prob, optim_prob])
        return arm

    def VIDS_action(self, T, M=10000, optim_action=True):
        """
        Implementation of V-IDS with approximation of integrals using MC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward and regret obtained by the policy
        """

        mu_t, sigma_t = self.initPrior()
        reward, expected_regret = np.zeros(T), np.zeros(T)
        for t in range(T):
            self.set_context()
            thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
            value = np.dot(thetas, self.features.T)
            if optim_action:
                delta, v, p_a = self.computeVIDS_v2(value)
            else:
                delta, v = self.computeVIDS_v3(value)
            a_t = self.vids_sample_by_action(delta, v)
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def VIDS_action_hyper(
        self,
        T,
        file,
        M=10000,
        optim_action=True,
        noise_dim=2,
        fg_lambda=1.0,
        fg_decay=True,
        lr=0.01,
        batch_size=32,
        hidden_sizes=(),
        optim="Adam",
        update_num=2,
        reset=False,
    ):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward and regret obtained by the policy
        """
        norm_coef = (self.eta / self.prior_sigma) ** 2
        model = HyperModel(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_std=self.prior_sigma,
            fg_lambda=fg_lambda,
            fg_decay=fg_decay,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            target_noise_coef=self.eta,
            norm_coef=norm_coef,
            buffer_size=T,
            reset=reset,
        )

        reward, expected_regret = np.zeros(T), np.zeros(T)
        for t in range(T):
            self.set_context()
            value = model.predict(self.features, M)
            if optim_action:
                delta, v, p_a = self.computeVIDS_v2(value)
            else:
                delta, v = self.computeVIDS_v3(value)
            a_t = self.vids_sample_by_action(delta, v)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, expected_regret

    def VIDS_policy(self, T, M=10000, optim_action=True):
        """
        Implementation of V-IDS with approximation of integrals using MC sampling for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward and regret obtained by the policy
        """

        mu_t, sigma_t = self.initPrior()
        reward, expected_regret = np.zeros(T), np.zeros(T)
        for t in range(T):
            self.set_context()
            thetas = np.random.multivariate_normal(mu_t, sigma_t, M)
            value = np.dot(thetas, self.features.T)
            if optim_action:
                delta, v, p_a = self.computeVIDS_v2(value)
            else:
                delta, v = self.computeVIDS_v3(value)
            a_t = self.vids_sample_by_policy(delta, v)
            r_t, mu_t, sigma_t = self.updatePosterior(a_t, mu_t, sigma_t)
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

        return reward, expected_regret

    def VIDS_policy_hyper(
        self,
        T,
        file,
        M=10000,
        optim_action=True,
        noise_dim=2,
        fg_lambda=1.0,
        fg_decay=True,
        lr=0.01,
        batch_size=32,
        hidden_sizes=(),
        optim="Adam",
        update_num=2,
        reset=False,
    ):
        """
        Implementation of V-IDS with hypermodel for Linear Bandits with multivariate
        normal prior
        :param T: int, time horizon
        :param M: int, number of samples. Default: 10 000
        :return: np.arrays, reward and regret obtained by the policy
        """
        norm_coef = (self.eta / self.prior_sigma) ** 2
        model = HyperModel(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_std=self.prior_sigma,
            fg_lambda=fg_lambda,
            fg_decay=fg_decay,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            target_noise_coef=self.eta,
            norm_coef=norm_coef,
            buffer_size=T,
            reset=reset,
        )

        reward, expected_regret = np.zeros(T), np.zeros(T)
        for t in range(T):
            self.set_context()
            value = model.predict(self.features, M)
            if optim_action:
                delta, v, p_a = self.computeVIDS_v2(value)
            else:
                delta, v = self.computeVIDS_v3(value)
            a_t = self.vids_sample_by_policy(delta, v)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            for _ in range(update_num):
                model.update()
        return reward, expected_regret
