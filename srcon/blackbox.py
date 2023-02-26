import numpy as np
import torch


class BlackBox:
    def __init__(self, func_name: str = "Branin", input_dim: int = 20) -> None:
        assert func_name in ["Branin", "Schwefel", "Ackley"]
        self.function = getattr(self, f"_{func_name}_func")
        self.bound_point = getattr(self, f"_{func_name}_bound")
        self.sample_action = getattr(self, f"_{func_name}_sample")
        self.input_dim = input_dim

    def _Branin_func(self, x):
        """
        x1 in [-5, 10] x2 in [0, 15]
        global minimum 0.397887 at (-pi, 12.275), (pi, 2.275) and (9.42478, 2.475)
        """
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        x1, x2 = x[:, 0], x[:, 1]
        y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        return y

    def _Branin_bound(self, x):
        if isinstance(x, np.ndarray):
            x[:, 0] = np.clip(x[:, 0], -5, 10)
            x[:, 1] = np.clip(x[:, 1], 0, 15)
        elif isinstance(x, torch.Tensor):
            x[:, 0] = torch.clamp(x[:, 0], -5, 10)
            x[:, 1] = torch.clamp(x[:, 1], 0, 15)
        return x

    def _Branin_sample(self, num):
        x1 = np.random.uniform(-5, 10, (num, 1))
        x2 = np.random.uniform(0, 15, (num, 1))
        x = np.concatenate([x1, x2], -1)
        return x

    def _Schwefel_func(self, x):
        """
        x in [-500, 500]
        global minimum 0 at (420.9687,...,420.9687)
        """
        d = len(x)
        y = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return y

    def _Schwefel_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, -500, 500)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -500, 500)
        return x

    def _Schwefel_sample(self, num):
        x = np.random.uniform(-500, 500, (num, self.input_dim))
        return x

    def _Ackley_func(self, x):
        """
        x in [-32.768, 32.768]
        global minimum 0 at (0,...,0)
        """
        a = 20
        b = 0.2
        c = 2 * np.pi
        y = (
            -a * np.exp(-b * np.sqrt(np.mean(x**2)))
            - np.exp(np.mean(np.cos(c * x)))
            + a
            + np.exp(1)
        )
        return y

    def _Ackley_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, -32.768, 32.768)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -32.768, 32.768)
        return x

    def _Ackley_sample(self, num):
        x = np.random.uniform(-32.768, 32.768, (num, self.input_dim))
        return x

    def call(self, x):
        return self.function(x)
