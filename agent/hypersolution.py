from typing import Sequence
import sys

sys.path.append("..")

import itertools as it
import numpy as np
import torch
import torch.nn.functional as F

from network.epinet import EpiNet
from network.hypernet import HyperNet
from network.ensemble import EnsembleNet


class ReplayBuffer:
    def __init__(self, buffer_size, buffer_shape, noise_type="sp"):
        self.buffers = {
            key: np.empty([buffer_size, *shape]) for key, shape in buffer_shape.items()
        }
        self.noise_dim = buffer_shape["z"][-1]
        self.sample_num = 0
        if noise_type == "sp":
            self._gen_noise = self._gen_sphere_noise
        elif noise_type == "gs":
            self._gen_noise = self._gen_gs_noise
        else:
            NotImplementedError

    def __len__(self):
        return self.sample_num

    def _gen_sphere_noise(self):
        noise = np.random.randn(self.noise_dim).astype(np.float32)
        noise /= np.linalg.norm(noise)
        return noise

    def _gen_gs_noise(self):
        noise = np.random.randn(self.noise_dim).astype(np.float32)
        return noise

    def _sample(self, index):
        a_data = self.buffers["a"][: self.sample_num]
        s_data = self.buffers["s"][: self.sample_num]
        f_data = s_data[np.arange(self.sample_num), a_data.astype(np.int32)]
        r_data = self.buffers["r"][: self.sample_num]
        z_data = self.buffers["z"][: self.sample_num]
        s_data, f_data, r_data, z_data = (
            s_data[index],
            f_data[index],
            r_data[index],
            z_data[index],
        )
        return s_data, f_data, r_data, z_data

    def reset(self):
        self.sample_num = 0

    def put(self, transition):
        for k, v in transition.items():
            self.buffers[k][self.sample_num] = v
        z = self._gen_noise()
        self.buffers["z"][self.sample_num] = z
        self.sample_num += 1

    def get(self, shuffle=True):
        # get all data in buffer
        index = list(range(self.sample_num))
        if shuffle:
            np.random.shuffle(index)
        return self._sample(index)

    def sample(self, n):
        # get n data in buffer
        index = np.random.randint(low=0, high=self.sample_num, size=n)
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
        self.update = (
            getattr(self, "_update_reset") if reset else getattr(self, "_update")
        )

    def __init_model_optimizer(self):
        # init hypermodel
        if self.model_type == "hyper":
            Net = HyperNet
        elif self.model_type == "epinet":
            Net = EpiNet
        elif self.model_type == "ensemble":
            Net = EnsembleNet
        else:
            raise NotImplementedError
        self.model = Net(
            in_features=self.feature_dim,
            hidden_sizes=self.hidden_sizes,
            noise_dim=self.noise_dim,
            prior_scale=self.prior_scale,
            posterior_scale=self.posterior_scale,
            device=self.device,
        ).to(self.device)
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
        buffer_shape = {
            "s": (self.action_dim, self.feature_dim),
            "a": (),
            "r": (),
            "z": (self.noise_dim,),
        }
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
        s_batch = torch.FloatTensor(s_batch).to(self.device)

        update_noise = self.generate_update_noise(
            (self.batch_size, self.NpS), self.update_noise
        )  # noise for update
        target_noise = (
            torch.mul(z_batch.unsqueeze(1), update_noise).sum(-1) * self.noise_coef
        )  # noise for target
        predict = self.model(update_noise, f_batch)
        diff = target_noise + r_batch.unsqueeze(-1) - predict
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

    def get_thetas(self, M=1):
        assert len(self.hidden_sizes) == 0, f"hidden size > 0"
        action_noise = self.generate_action_noise((M,), self.action_noise)
        with torch.no_grad():
            thetas = self.model.out.get_thetas(action_noise).cpu().numpy()
        return thetas

    def predict(self, features, M=1):
        action_noise = self.generate_action_noise((M,), self.action_noise)
        with torch.no_grad():
            p_a = self.model(action_noise, features).cpu().numpy()
        return p_a

    def generate_action_noise(self, noise_num, noise_type):
        if noise_type == "gs":
            noise = self.gen_gs_noise(noise_num)
        elif noise_type == "sp":
            noise = self.gen_sp_noise(noise_num)
        elif noise_type == "sgs":
            noise = self.gen_sgs_noise(noise_num)
        elif noise_type == "oh":
            noise = self.gen_one_hot_noise(noise_num)
        elif noise_type == "pn":
            noise = self.gen_pn_one_noise(noise_num)
        return noise

    def generate_update_noise(self, noise_num, noise_type):
        if noise_type == "gs":
            noise = self.gen_gs_noise(noise_num)
        elif noise_type == "sp":
            noise = self.gen_sp_noise(noise_num)
        elif noise_type == "sgs":
            noise = self.gen_sgs_noise(noise_num)
        elif noise_type == "oh":
            batch_size = noise_num[0]
            noise = torch.arange(self.noise_dim).unsqueeze(0).repeat(batch_size, 1)
            noise = F.one_hot(noise, self.noise_dim).to(torch.float32).to(self.device)
        elif noise_type == "pn":
            batch_size = noise_num[0]
            noise = np.array(list((it.product(range(2), repeat=self.noise_dim))))
            noise = noise * 2 - 1
            noise = torch.from_numpy(noise).to(torch.float32).to(self.device)
            noise = noise.unsqueeze(0).repeat(batch_size, 1, 1)
        return noise

    def gen_gs_noise(self, noise_num):
        noise_shape = noise_num + (self.noise_dim,)
        noise = torch.randn(noise_shape).type(torch.float32)
        return noise.to(self.device)

    def gen_sp_noise(self, noise_num):
        noise = self.gen_gs_noise(noise_num)
        noise /= torch.norm(noise, dim=-1, keepdim=True)
        return noise

    def gen_sgs_noise(self, noise_num):
        noise = self.gen_gs_noise(noise_num)
        noise /= np.sqrt(self.noise_dim)
        return noise

    def gen_one_hot_noise(self, noise_num: tuple):
        noise = torch.randint(0, self.noise_dim, noise_num)
        noise = F.one_hot(noise, self.noise_dim).to(torch.float32)
        return noise.to(self.device)

    def gen_pn_one_noise(self, noise_num: tuple):
        noise_shape = noise_num + (self.noise_dim,)
        noise = torch.ones(noise_shape) + torch.randint(-1, 1, noise_shape) * 2
        return noise.to(self.device)

    def reset(self):
        self.__init_model_optimizer()
