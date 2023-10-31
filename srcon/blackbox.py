import numpy as np
import torch

D_THETA = {
    "Branin": 2,
    "Schwefel": 3,
    "Ackley": 20,
    "Hartmann": 6,
}


class BlackBox:
    def __init__(self, func_name: str = "Branin", input_dim: int = 20) -> None:
        assert func_name in ["Branin", "Schwefel", "Ackley", "Hartmann"]
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
        return y[0]

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
        x *= 100
        d = x.shape[-1]
        y = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)
        return y[0]

    def _Schwefel_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, -5, 5)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -5, 5)
        return x

    def _Schwefel_sample(self, num):
        x = np.random.uniform(-5, 5, (num, self.input_dim))
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
            -a * np.exp(-b * np.sqrt(np.mean(x**2, axis=-1)))
            - np.exp(np.mean(np.cos(c * x), axis=-1))
            + a
            + np.exp(1)
        )
        return y[0]

    def _Ackley_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, -32.768, 32.768)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -32.768, 32.768)
        return x

    def _Ackley_sample(self, num):
        x = np.random.uniform(-32.768, 32.768, (num, self.input_dim))
        return x

    def _Hartmann_func(self, x):
        """
        x in [0, 1]
        global minimum -3.32237 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
        """
        alpha = [1.0, 1.2, 3.0, 3.2]
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = (
            np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )
            * 1e-4
        )
        temp_y = 0
        for i in range(len(alpha)):
            temp_y += alpha[i] * np.exp(
                -np.sum(A[i, :] * (x[0] - P[i, :]) ** 2, axis=-1)
            )

        y = -(2.58 + temp_y) / 1.94
        return y

    def _Hartmann_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, 0, 1)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, 0, 1)
        return x

    def _Hartmann_sample(self, num):
        x = np.random.uniform(0, 1, (num, self.input_dim))
        return x

    def call(self, x):
        return self.function(x)


if __name__ == "__main__":
    input_dim = 10
    blackbox = BlackBox("Schwefel", 3)
    x = np.array([[4.209687] * 3])
    print(blackbox.call(x))
    blackbox = BlackBox("Branin", 2)
    x = np.array([[9.42478, 2.475]])
    print(blackbox.call(x))
    blackbox = BlackBox("Ackley", 20)
    x = np.array([[0] * 20])
    print(blackbox.call(x))
    blackbox = BlackBox("Hartmann", 6)
    x = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
    print(blackbox.call(x))
