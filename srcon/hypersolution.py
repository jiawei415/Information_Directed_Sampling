import sys

sys.path.append("..")

import itertools as it
import copy
import numpy as np
import torch
import torch.nn.functional as F

from utils import sample_action_noise, sample_update_noise, rd_argmax
from network.epinet import EpiNet
from network.hypernet import HyperNet
from network.ensemble import EnsembleNet


class ReplayBuffer:
    def __init__(self, buffer_size, buffer_shape, noise_type="sp"):
        print(buffer_shape)
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

        update_noise = self.generate_update_noise(
            (self.batch_size, self.NpS), self.update_noise
        )  # noise for update
        target_noise = (
            torch.mul(z_batch.unsqueeze(1), update_noise).sum(-1) * self.noise_coef
        )  # noise for target
        predict = self.model(update_noise, x_batch)
        diff = target_noise + y_batch.unsqueeze(-1) - predict
        diff = diff.pow(2).mean(-1)
        loss = diff.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def get_thetas(self, M=1):
        assert len(self.hidden_sizes) == 0, f"hidden size > 0"
        action_noise = self.generate_action_noise((M,), self.action_noise)
        with torch.no_grad():
            thetas = self.model.out.get_thetas(action_noise).cpu().numpy()
        return thetas

    def predict(self, features, M=1):
        action_noise = self.generate_action_noise((M,), self.action_noise)
        with torch.no_grad():
            out = self.model(action_noise, features).cpu().numpy()
        return out

    def reset(self):
        self.__init_model_optimizer()

    def select_action(self, actions, bound_func, single_noise=True, maximize=True):
        # init actions
        # actions = np.random.rand(self.action_num, self.input_dim).astype(np.float32)
        actions = torch.from_numpy(actions).to(self.device)
        actions.requires_grad = True
        # sample noise
        if single_noise:
            noise = self.generate_action_noise((1,), self.action_noise)
            noise = noise.repeat(actions.shape[0], 1)
        else:
            noise = self.generate_action_noise((actions.shape[0],), self.action_noise)
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
        elif noise_type == "sps":
            noise = sample_action_noise("Sparse", M=self.noise_dim, dim=noise_num[0])
            noise = torch.from_numpy(noise).to(torch.float32).to(self.device)
        elif noise_type == "spc":
            noise = sample_action_noise(
                "SparseConsistent", M=self.noise_dim, dim=noise_num[0]
            )
            noise = torch.from_numpy(noise).to(torch.float32).to(self.device)
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
        elif noise_type == "sps":
            batch_size = noise_num[0]
            noise = sample_update_noise(
                "Sparse", M=self.noise_dim, batch_size=batch_size
            )
            noise = torch.from_numpy(noise).to(torch.float32).to(self.device)
        elif noise_type == "spc":
            batch_size = noise_num[0]
            noise = sample_update_noise(
                "SparseConsistent", M=self.noise_dim, batch_size=batch_size
            )
            noise = torch.from_numpy(noise).to(torch.float32).to(self.device)
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

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
