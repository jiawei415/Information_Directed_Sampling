from typing import Sequence
from functools import partial
import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F

from logger import Logger
from network import HyperNet, LinearNet


class ReplayBuffer:
    def __init__(
        self, buffer_size, buffer_shape
    ):
        self.buffers = {
            key: np.empty([buffer_size, *shape], dtype=np.float32)
            for key, shape in buffer_shape.items()
        }
        self.buffer_size = buffer_size
        self.current_size = 0
        self.point = 0

    def __len__(self):
        return self.current_size

    def _sample(self, index):
        f_data = self.buffers["f"][index]
        r_data = self.buffers["r"][index]
        return f_data, r_data

    def reset(self):
        self.current_size = 0

    def put(self, transition):
        transition.pop("a")
        batch_size = 1
        idx = self._get_ordered_storage_idx(batch_size)
        for k, v in transition.items():
            self.buffers[k][idx] = v

    def get(self, shuffle=True):
        # get all data in buffer
        index = list(range(self.current_size))
        if shuffle:
            np.random.shuffle(index)
        return self._sample(index)

    def sample(self, n):
        # get n data in buffer
        index = np.random.randint(low=0, high=self.current_size, size=n)
        return self._sample(index)

    def sample_all(self):
        return self._sample(range(self.current_size))

    # if full, insert in order
    def _get_ordered_storage_idx(self, inc=None):
        inc = inc or 1  # size increment
        assert inc <= self.buffer_size, "Batch committed to replay is too large!"

        if self.point + inc <= self.buffer_size - 1:
            idx = np.arange(self.point, self.point + inc)
        else:
            overflow = inc - (self.buffer_size - self.point)
            idx_a = np.arange(self.point, self.buffer_size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])

        self.point = (self.point + inc) % self.buffer_size

        # update replay size, don't add when it already surpass self.size
        if self.current_size < self.buffer_size:
            self.current_size = min(self.buffer_size, self.current_size + inc)

        if inc == 1:
            idx = idx[0]
        return idx


class GreedySolution:
    def __init__(
        self,
        n_action: int,
        n_feature: int,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        hidden_sizes: Sequence[int] = (),
        class_num: int = 1,
        optim: str = "Adam",
        lr: float = 0.01,
        batch_size: int = 32,
        weight_decay: float = 0.01,
        buffer_size: int = 10000,
        model_type: str = "linear",
        logger: Logger = None,
    ):
        self.action_dim = n_action
        self.feature_dim = n_feature

        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

        self.hidden_sizes = hidden_sizes
        self.class_num = class_num

        self.optim = optim
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.buffer_size = buffer_size
        self.model_type = model_type
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.init_model_optimizer()
        self.init_buffer()

    def init_model_optimizer(self):
        # init hypermodel
        model_param = {
            "in_features": self.feature_dim,
            "hidden_sizes": self.hidden_sizes,
            "action_num": self.class_num,
            "prior_scale": self.prior_scale,
            "posterior_scale": self.posterior_scale,
            "device": self.device,
        }
        self.model = LinearNet(**model_param).to(self.device)
        param_dict = {"Trainable": [], "Frozen": []}
        trainable_param_size, frozen_param_size = 0, 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_param_size += param.numel()
                param_dict["Trainable"].append(name)
            else:
                frozen_param_size += param.numel()
                param_dict["Frozen"].append(name)
        self.logger.info(f"Trainable parameters:\n", "\n".join(param_dict['Trainable']))
        self.logger.info(f"Frozen parameters:\n", "\n".join(param_dict['Frozen']))
        self.logger.info(f"Network structure:\n{str(self.model)}")
        self.logger.info(
            f"Network parameters: {sum(param.numel() for param in self.model.parameters() if param.requires_grad)}"
        )
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim == "SGD":
            self.optimizer = torch.optim.SGD(trainable_params, lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            raise NotImplementedError

    def init_buffer(self):
        buffer_shape = {"f": (self.feature_dim,), "r": ()}
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape)

    def update(self):
        if self.batch_size is None:
            f_batch, r_batch = self.buffer.sample_all()
        else:
            f_batch, r_batch = self.buffer.sample(self.batch_size)
        self.learn(f_batch, r_batch)

    def put(self, transition):
        self.buffer.put(transition)

    def learn(self, f_batch, r_batch):
        f_batch = torch.FloatTensor(f_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)

        predict = self.model(f_batch)
        if self.class_num > 1:
            r_batch = r_batch.to(torch.int64)
            loss = F.cross_entropy(predict, r_batch)
        else:
            diff = r_batch - predict
            loss = diff.pow(2).mean()

        for param_group in self.optimizer.param_groups:
            param_group["weight_decay"] = self.weight_decay / len(self.buffer)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_thetas(self, num=1):
        assert len(self.hidden_sizes) == 0, f"hidden size > 0"
        action_noise = self.gen_action_noise(dim=num)
        with torch.no_grad():
            thetas = self.model.out.get_thetas(action_noise).cpu().numpy()
        return thetas

    def predict(self, features):
        with torch.no_grad():
            p_a = self.model(features).cpu().numpy()
        return p_a
