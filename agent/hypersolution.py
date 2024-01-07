from typing import Sequence
from functools import partial
import sys

sys.path.append("..")

import numpy as np
import torch

from utils import sample_action_noise, sample_update_noise, sample_buffer_noise
from network import HyperNet, EnsembleNet, EpiNet


class ReplayBuffer:
    def __init__(
        self, buffer_size, buffer_shape, noise_type="sp", save_full_feature=False
    ):
        self.buffers = {
            key: np.empty([buffer_size, *shape], dtype=np.float32)
            for key, shape in buffer_shape.items()
        }
        self.buffer_size = buffer_size
        self.noise_dim = buffer_shape["z"][-1]
        self.save_full_feature = save_full_feature
        self.sample_num = 0
        self.set_buffer_noise(noise_type)

    def set_buffer_noise(self, noise_type):
        args = {"M": self.noise_dim}
        if noise_type == "gs":
            self.gen_noise = partial(sample_buffer_noise, "Gaussian", **args)
        elif noise_type == "sp":
            self.gen_noise = partial(sample_buffer_noise, "Sphere", **args)
        elif noise_type == "pn":
            self.gen_noise = partial(sample_buffer_noise, "UnifCube", **args)
        elif noise_type == "oh":
            self.gen_noise = partial(sample_buffer_noise, "OH", **args)
        elif noise_type == "sps":
            self.gen_noise = partial(sample_buffer_noise, "Sparse", **args)
        elif noise_type == "spc":
            self.gen_noise = partial(sample_buffer_noise, "SparseConsistent", **args)

    def __len__(self):
        return min(self.sample_num, self.buffer_size)

    def _sample(self, index):
        if self.save_full_feature:
            a_data = self.buffers["a"]
            f_data = s_data[np.arange(self.buffer_size), a_data.astype(np.int32)][index]
            s_data = self.buffers["s"][index]
        else:
            f_data = self.buffers["f"][index]
            s_data = None
        r_data = self.buffers["r"][index]
        z_data = self.buffers["z"][index]
        return s_data, f_data, r_data, z_data

    def reset(self):
        self.sample_num = 0

    def put(self, transition):
        inset_index = self.sample_num
        if self.sample_num >= self.buffer_size:
            inset_index = self.sample_num % self.buffer_size
        if self.save_full_feature:
            for k, v in transition.items():
                self.buffers[k][inset_index] = v
        else:
            self.buffers["r"][inset_index] = transition["r"]
            self.buffers["f"][inset_index] = transition["s"][transition["a"]]
        z = self.gen_noise()
        self.buffers["z"][inset_index] = z
        self.sample_num += 1

    def get(self, shuffle=True):
        # get all data in buffer
        sample_num = min(self.sample_num, self.buffer_size)
        index = list(range(sample_num))
        if shuffle:
            np.random.shuffle(index)
        return self._sample(index)

    def sample(self, n):
        # get n data in buffer
        sample_num = min(self.sample_num, self.buffer_size)
        index = np.random.randint(low=0, high=sample_num, size=n)
        return self._sample(index)


class HyperSolution:
    def __init__(
        self,
        noise_dim: int,
        n_action: int,
        n_feature: int,
        hidden_sizes: Sequence[int] = (),
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        batch_size: int = 32,
        lr: float = 0.01,
        optim: str = "Adam",
        fg_lambda: float = 0.0,
        fg_decay: bool = True,
        weight_decay: float = 0.01,
        noise_coef: float = 0.01,
        buffer_size: int = 10000,
        buffer_noise: str = "sp",
        NpS: int = 20,
        action_noise: str = "sgs",
        update_noise: str = "pn",
        model_type: str = "hyper",
        reset: bool = False,
    ):
        self.noise_dim = noise_dim
        self.action_dim = n_action
        self.feature_dim = n_feature
        self.hidden_sizes = hidden_sizes
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.lr = lr
        self.fg_lambda = fg_lambda
        self.fg_decay = fg_decay
        self.batch_size = batch_size
        self.NpS = NpS
        self.optim = optim
        self.weight_decay = weight_decay
        self.noise_coef = noise_coef
        self.buffer_size = buffer_size
        self.action_noise = action_noise
        self.update_noise = update_noise
        self.buffer_noise = buffer_noise
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type

        self.__init_model_optimizer()
        self.__init_buffer()
        self.set_update_noise()
        self.set_action_noise()
        self.update = (
            getattr(self, "_update_reset") if reset else getattr(self, "_update")
        )

    def __init_model_optimizer(self):
        # init hypermodel
        model_param = {
            "in_features": self.feature_dim,
            "hidden_sizes": self.hidden_sizes,
            "noise_dim": self.noise_dim,
            "prior_scale": self.prior_scale,
            "posterior_scale": self.posterior_scale,
            "device": self.device,
        }
        if self.model_type == "hyper":
            Net = HyperNet
            model_param.update({"hyper_bias": True})
        elif self.model_type == "epinet":
            Net = EpiNet
        elif self.model_type == "ensemble":
            Net = EnsembleNet
        else:
            raise NotImplementedError
        self.model = Net(**model_param).to(self.device)
        print(f"\nNetwork structure:\n{str(self.model)}")
        # init optimizer
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise NotImplementedError

    def __init_buffer(self):
        # init replay buffer
        # buffer_shape = {
        #     "s": (self.action_dim, self.feature_dim),
        #     "a": (),
        #     "r": (),
        #     "z": (self.noise_dim,),
        # }
        buffer_shape = {"f": (self.feature_dim,), "r": (), "z": (self.noise_dim,)}
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape, self.buffer_noise)

    def _update(self):
        s_batch, f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
        self.learn(s_batch, f_batch, r_batch, z_batch)

    def _update_reset(self):
        sample_num = len(self.buffer)
        if sample_num > self.batch_size:
            s_data, f_data, r_data, z_data = self.buffer.get()
            for i in range(0, self.batch_size, sample_num):
                s_batch, f_batch, r_batch, z_batch = (
                    s_data[i : i + self.batch_size],
                    f_data[i : i + self.batch_size],
                    r_data[i : i + self.batch_size],
                    z_data[i : i + self.batch_size],
                )
                self.learn(s_batch, f_batch, r_batch, z_batch)
            if sample_num % self.batch_size != 0:
                last_sample = sample_num % self.batch_size
                index1 = -np.arange(1, last_sample + 1).astype(np.int32)
                index2 = np.random.randint(
                    low=0, high=sample_num, size=self.batch_size - last_sample
                )
                index = np.hstack([index1, index2])
                s_batch, f_batch, r_batch, z_batch = (
                    s_data[index],
                    f_data[index],
                    r_data[index],
                    z_data[index],
                )
                self.learn(s_batch, f_batch, r_batch, z_batch)
        else:
            s_batch, f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
            self.learn(s_batch, f_batch, r_batch, z_batch)

    def put(self, transition):
        self.buffer.put(transition)

    def learn(self, s_batch, f_batch, r_batch, z_batch):
        z_batch = torch.FloatTensor(z_batch).to(self.device)
        f_batch = torch.FloatTensor(f_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)
        if s_batch is not None:
            s_batch = torch.FloatTensor(s_batch).to(self.device)

        # noise for update
        update_noise = torch.from_numpy(self.gen_update_noise()).to(self.device)
        # noise for target
        target_noise = torch.bmm(update_noise, z_batch.unsqueeze(-1)) * self.noise_coef

        predict = self.model(update_noise, f_batch)
        diff = target_noise.squeeze(-1) + r_batch.unsqueeze(-1) - predict
        diff = diff.pow(2).mean(-1)
        if self.fg_lambda:
            fg_lambda = (
                self.fg_lambda / np.sqrt(len(self.buffer))
                if self.fg_decay
                else self.fg_lambda
            )
            fg_term = self.model(update_noise, s_batch)
            fg_term = fg_term.max(dim=-1)[0]
            loss = (diff - fg_lambda * fg_term).mean()
        else:
            loss = diff.mean()
        # norm_coef = self.norm_coef / len(self.buffer)
        # reg_loss = self.model.regularization(update_noise) * norm_coef
        # loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_thetas(self, num=1):
        assert len(self.hidden_sizes) == 0, f"hidden size > 0"
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            thetas = self.model.out.get_thetas(action_noise).cpu().numpy()
        return thetas

    def predict(self, features, num=1):
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            p_a = self.model(action_noise, features).cpu().numpy()
        return p_a

    def set_action_noise(self):
        args = {"M": self.noise_dim}
        if self.action_noise == "gs":
            self.gen_action_noise = partial(sample_action_noise, "Gaussian", **args)
        elif self.action_noise == "sp":
            self.gen_action_noise = partial(sample_action_noise, "Sphere", **args)
        elif self.action_noise == "pn":
            self.gen_action_noise = partial(sample_action_noise, "UnifCube", **args)
        elif self.action_noise == "oh":
            self.gen_action_noise = partial(sample_action_noise, "OH", **args)
        elif self.action_noise == "sps":
            self.gen_action_noise = partial(sample_action_noise, "Sparse", **args)
        elif self.action_noise == "spc":
            self.gen_action_noise = partial(
                sample_action_noise, "SparseConsistent", **args
            )

    def set_update_noise(self):
        args = {"M": self.noise_dim, "dim": self.NpS, "batch_size": self.batch_size}
        if self.update_noise == "gs":
            self.gen_update_noise = partial(sample_update_noise, "Gaussian", **args)
        elif self.update_noise == "sp":
            self.gen_update_noise = partial(sample_update_noise, "Sphere", **args)
        elif self.update_noise == "pn":
            self.gen_update_noise = partial(sample_update_noise, "UnifCube", **args)
        elif self.update_noise == "oh":
            self.gen_update_noise = partial(sample_update_noise, "OH", **args)
        elif self.update_noise == "sps":
            self.gen_update_noise = partial(sample_update_noise, "Sparse", **args)
        elif self.update_noise == "spc":
            self.gen_update_noise = partial(
                sample_update_noise, "SparseConsistent", **args
            )

    def reset(self):
        self.__init_model_optimizer()
