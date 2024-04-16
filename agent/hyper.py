import numpy as np

from tqdm import tqdm
from utils import rd_argmax
from agent.hypersolution import HyperSolution
from agent.hyperllmsolution import HyperLLMSolution
from agent.lmcts import LMCTS


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
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
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
        noise_dim=2,
        fg_lambda=1.0,
        fg_decay=True,
        lr=0.01,
        batch_size=32,
        hidden_sizes=(),
        optim="Adam",
        update_num=2,
        NpS=20,
        action_noise="sgs",
        update_noise="pn",
        reset=False,
    ):
        """
        Implementation of Thomson Sampling (TS) algorithm for Linear Bandits with multivariate normal prior
        :param T: int, time horizon
        :return: np.arrays, reward and regret obtained by the policy
        """
        norm_coef = (self.eta / self.prior_sigma) ** 2
        model = HyperSolution(
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
            noise_coef=self.eta,
            norm_coef=norm_coef,
            buffer_size=T,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            reset=reset,
        )

        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
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

    def LLM(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="pn",
        update_noise="gs",
        buffer_noise="sp",
        buffer_size=None,
        model_type="hyper",
        llm_name="gpt2",
        use_lora=False,
        fine_tune=False,
        out_bias=True,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = HyperLLMSolution(
            noise_dim,
            self.n_a,
            self.d,
            prior_scale=prior_scale,
            lr=lr,
            NpS=NpS,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            buffer_size=buffer_size,
            model_type=model_type,
            llm_name=llm_name,
            use_lora=use_lora,
            fine_tune=fine_tune,
            out_bias=out_bias,
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in tqdm(range(T)):
            self.set_context()
            input_ids, attention_mask = self.env.get_feature()
            a_t = model.predict(input_ids, attention_mask, num=self.n_a)
            r_t = self.reward(a_t)
            regret_t = self.expect_regret(a_t, self.features)
            reward[t], expected_regret[t] = r_t.mean(), regret_t.mean()

            transitions = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "a": a_t,
                "r": r_t,
            }
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
        return reward, expected_regret

    def Hyper(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        hidden_sizes=(),
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="pn",
        update_noise="gs",
        buffer_noise="sp",
        buffer_size=None,
        out_bias=True,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = HyperSolution(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_scale=prior_scale,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            buffer_size=buffer_size,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            model_type="hyper",
            out_bias=out_bias,
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
        return reward, expected_regret

    def EpiNet(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        hidden_sizes=(),
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="gs",
        update_noise="gs",
        buffer_noise="sp",
        buffer_size=None,
        class_num=1,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = HyperSolution(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            class_num=class_num,
            prior_scale=prior_scale,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            buffer_size=buffer_size,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            model_type="epinet",
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)
            if class_num > 1:
                value = value[:, 1]
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
        return reward, expected_regret

    def Ensemble(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        hidden_sizes=(),
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="oh",
        update_noise="oh",
        buffer_noise="gs",
        buffer_size=None,
        out_bias=True,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = HyperSolution(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_scale=prior_scale,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            buffer_size=buffer_size,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            model_type="ensemble",
            out_bias=out_bias,
        )

        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                for _ in range(update_num):
                    model.update()
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
        return reward, expected_regret

    def LMCTS(
        self,
        T,
        logger,
        noise_dim=2,
        lr=0.01,
        based_weight_decay=0.0,
        hyper_weight_decay=0.0,
        z_coef=None,
        batch_size=32,
        hidden_sizes=(),
        prior_scale=1.0,
        optim="Adam",
        update_num=2,
        update_start=32,
        update_freq=1,
        NpS=20,
        action_noise="oh",
        update_noise="oh",
        buffer_noise="gs",
        buffer_size=None,
    ):
        z_coef = z_coef if z_coef is not None else self.eta
        buffer_size = buffer_size or T
        model = LMCTS(
            noise_dim,
            self.n_a,
            self.d,
            hidden_sizes=hidden_sizes,
            prior_scale=prior_scale,
            lr=lr,
            batch_size=batch_size,
            optim=optim,
            noise_coef=z_coef,
            based_weight_decay=based_weight_decay,
            hyper_weight_decay=hyper_weight_decay,
            buffer_size=buffer_size,
            NpS=NpS,
            action_noise=action_noise,
            update_noise=update_noise,
            buffer_noise=buffer_noise,
            model_type="linear",
        )

        update_step = 0
        log_interval = T // 1000
        reward, expected_regret = np.zeros(T, dtype=np.float32), np.zeros(
            T, dtype=np.float32
        )
        for t in range(T):
            self.set_context()
            value = model.predict(self.features)
            a_t = rd_argmax(value)
            f_t, r_t = self.features[a_t], self.reward(a_t)[0]
            reward[t], expected_regret[t] = r_t, self.expect_regret(a_t, self.features)

            transitions = {"s": self.features, "r": r_t, "a": a_t}
            model.put(transitions)
            # update hypermodel
            if t >= update_start and (t + 1) % update_freq == 0:
                if update_num == 0:
                    num_iter = min(t + 1, 100)
                else:
                    num_iter = update_num
                if update_num == 0 and update_step > 0 and update_step % 20 == 0:
                    model.optimizer.lr = model.lr / update_step
                for _ in range(num_iter):
                    model.update()
                update_step += 1
            if t == 0 or (t + 1) % log_interval == 0:
                logger.record("step", t + 1)
                logger.record("acc_regret", np.cumsum(expected_regret[: t + 1])[-1])
                logger.dump(t)
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
        model = HyperSolution(
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
            noise_coef=self.eta,
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
        model = HyperSolution(
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
            noise_coef=self.eta,
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
