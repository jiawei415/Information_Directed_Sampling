import numpy as np
import torch

from SRCON_simulator.utils import BlackBoxSRCONObjective


class SRCON:
    def __init__(self, dataset) -> None:
        self.objective = BlackBoxSRCONObjective(dataset)
        self.input_dim = self.objective.num_para

    def call(self, x):
        y = self.objective.call(x.squeeze(0))
        return y

    def sample_action(self, num):
        xs = np.random.rand(num, self.input_dim).astype(np.float32)
        return xs

    def bound_point(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, 0, 1)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, 0, 1)
        return x
