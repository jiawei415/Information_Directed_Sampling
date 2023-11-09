import sys

sys.path.append("..")

import itertools as it
import copy
import numpy as np
import torch
import torch.nn.functional as F

from functools import partial

from utils import (
    rd_argmax,
    sample_action_noise,
    sample_update_noise,
    sample_buffer_noise,
)
from network import HyperNet, EnsembleNet, EpiNet


class ReplayBuffer:
    def __init__(self, buffer_size, buffer_shape, noise_type="sp"):
        print(buffer_shape)
        self.buffers = {
            key: np.empty([buffer_size, *shape]) for key, shape in buffer_shape.items()
        }
        self.noise_dim = buffer_shape["z"][-1]
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
        return self.sample_num

    def _sample(self, index):
        x_data = self.buffers["x"][: self.sample_num]
        y_data = self.buffers["y"][: self.sample_num]
        z_data = self.buffers["z"][: self.sample_num]
        x_data, y_data, z_data = x_data[index], y_data[index], z_data[index]
        return x_data, y_data, z_data

    def reset(self):
        self.sample_num = 0

    def put(self, transition):
        for k, v in transition.items():
            self.buffers[k][self.sample_num] = v
        z = self.gen_noise()
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
        input_dim: int,
        hidden_sizes: list = [],
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        batch_size: int = 32,
        lr: float = 0.01,
        optim: str = "Adam",
        weight_decay: float = 0.01,
        noise_coef: float = 0.01,
        buffer_size: int = 10000,
        buffer_noise: str = "sp",
        NpS: int = 20,
        action_noise: str = "gs",
        update_noise: str = "gs",
        model_type: str = "hyper",
        action_num: int = 10,
        action_lr: float = 0.01,
        action_max_update: int = 1000,
    ):
        self.noise_dim = noise_dim
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.lr = lr
        self.batch_size = batch_size
        self.optim = optim
        self.weight_decay = weight_decay
        self.noise_coef = noise_coef
        self.noise_coef = noise_coef
        self.buffer_size = buffer_size
        self.NpS = NpS
        self.action_noise = action_noise
        self.update_noise = update_noise
        self.buffer_noise = buffer_noise
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type

        self.action_num = action_num
        self.action_lr = action_lr
        self.action_max_update = action_max_update

        self.__init_model_optimizer()
        self.__init_buffer()
        self.set_update_noise()
        self.set_action_noise()

    def __init_model_optimizer(self):
        # init hypermodel
        model_param = {
            "in_features": self.input_dim,
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
        buffer_shape = {"x": (self.input_dim,), "y": (), "z": (self.noise_dim,)}
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape, self.buffer_noise)

    def update(self):
        x_batch, y_batch, z_batch = self.buffer.sample(self.batch_size)
        results = self.learn(x_batch, y_batch, z_batch)
        return results

    def put(self, transition):
        self.buffer.put(transition)

    def learn(self, x_batch, y_batch, z_batch):
        y_batch = torch.FloatTensor(y_batch).to(self.device)
        x_batch = torch.FloatTensor(x_batch).to(self.device)
        z_batch = torch.FloatTensor(z_batch).to(self.device)

        # noise for update
        update_noise = torch.from_numpy(self.gen_update_noise()).to(self.device)
        # noise for target
        target_noise = torch.bmm(update_noise, z_batch.unsqueeze(-1)) * self.noise_coef

        predict = self.model(update_noise, x_batch)
        diff = target_noise.squeeze(-1) + y_batch.unsqueeze(-1) - predict
        diff = diff.pow(2).mean(-1)
        loss = diff.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def get_thetas(self, num=1):
        assert len(self.hidden_sizes) == 0, f"hidden size > 0"
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            thetas = self.model.out.get_thetas(action_noise).cpu().numpy()
        return thetas

    def predict(self, features, num=1):
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            out = self.model(action_noise, features).cpu().numpy()
        return out

    def reset(self):
        self.__init_model_optimizer()

    def select_action(self, actions, bound_func, single_noise=True, maximize=True):
        # init actions
        actions = torch.from_numpy(actions).to(self.device)
        actions.requires_grad = True
        # sample noise
        if single_noise:
            noise = self.gen_action_noise(dim=1)
            noise = noise.repeat(actions.shape[0], 0)
        else:
            noise = self.gen_action_noise(dim=actions.shape[0])
        model = copy.deepcopy(self.model)
        model.requires_grad = False
        optim = torch.optim.Adam([actions], lr=self.action_lr)
        # optimze actions
        for _ in range(self.action_max_update):
            out = model(noise, actions).mean()
            if maximize:
                out = -out
            optim.zero_grad()
            out.backward()
            optim.step()
            with torch.no_grad():
                actions = bound_func(actions)
        out = model(noise, actions)
        out = out.detach().cpu().numpy()
        if maximize:
            act_index = rd_argmax(out)
        else:
            act_index = rd_argmax(-out)
        actions = actions.detach().cpu().numpy()
        return np.array([actions[act_index]])

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

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
