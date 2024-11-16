from typing import List

import numpy as np
import math
import torch
import torch.optim as optim

from network import LinearNet
from .hypersolution import HyperSolution, ReplayBuffer


class NeuralUCB(HyperSolution):
    def __init__(self, nu=1.0, **kwargs):
        self.nu = nu
        super(NeuralUCB, self).__init__(**kwargs)

    def init_buffer(self):
        buffer_shape = {"f": (self.feature_dim,), "r": (), }
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape, self.buffer_noise)

    def init_model_optimizer(self):
        # init hypermodel
        model_param = {
            "in_features": self.feature_dim,
            "hidden_sizes": self.hidden_sizes,
            "prior_scale": self.prior_scale,
            "posterior_scale": self.posterior_scale,
            "device": self.device,
        }

        self.model = LinearNet(**model_param).to(self.device)
        self.logger.info(f"Network structure:\n{str(self.model)}")
        self.logger.info(
            f"Network parameters: {sum(param.numel() for param in self.model.parameters() if param.requires_grad)}"
        )
        self.total_param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.U = self.weight_decay * torch.ones((self.total_param,)).cuda()

    def put(self, transition):
        self.buffer.put(transition)

    def update(self):
        if self.batch_size == 0:
            f_batch, r_batch, z_batch = self.buffer.sample_all()
        else:
            f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
        return self.learn(f_batch, r_batch)

    def learn(self, f_batch, r_batch):
        optimizer = optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=self.weight_decay)
        f_batch = torch.FloatTensor(f_batch).to(self.device)
        r_batch = torch.FloatTensor(r_batch).to(self.device)
        length = len(r_batch)
        index = np.arange(length)
        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0
        while True:
            batch_loss = 0
            for idx in index:
                c = f_batch[idx]
                r = r_batch[idx]
                optimizer.zero_grad()
                delta = self.model(c.cuda()) - r
                loss = delta * delta
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                tot_loss += loss.item()
                cnt += 1
                if cnt >= 1000:
                    return tot_loss / 1000
            if batch_loss / length <= 1e-3:
                return batch_loss / length

    def predict(self, features, num=1):
        mu = self.model(features)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in mu:
            self.model.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.model.parameters()])
            g_list.append(g)
            sigma2 = self.weight_decay * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))

            sample_r = fx.item() + sigma.item()

            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r
        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm
