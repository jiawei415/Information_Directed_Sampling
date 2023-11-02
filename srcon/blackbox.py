# from https://www.sfu.ca/~ssurjano/optimization.html
import numpy as np
import torch

D_THETA = {
    "Bukin": 2,
    "Branin": 2,
    "Schwefel": 6,
    "Hartmann": 6,
}

FUNC_NAMES = [
    "Bukin",
    "Branin",
    "Hartmann",
    "Schwefel",
    "Ackley",
    "Michalewicz",
    "Levy",
    "Levy",
]


class BlackBox:
    def __init__(self, func_name: str = "Branin", input_dim: int = 20) -> None:
        assert func_name in FUNC_NAMES
        self.function = getattr(self, f"_{func_name}_func")
        self.bound_point = getattr(self, f"_{func_name}_bound")
        self.sample_action = getattr(self, f"_{func_name}_sample")
        self.optimal_action = getattr(self, f"_{func_name}_optimal")
        self.input_dim = input_dim

    def _Rosenbrock_func(self, x):
        term1 = 100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2
        term2 = (x[:, :-1] - 1) ** 2
        y = (term1 + term2).sum(-1)
        return y

    def _Rosenbrock_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, -5, 10)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -5, 10)
        return x

    def _Rosenbrock_sample(self, num):
        x = np.random.uniform(-5, 10, (num, self.input_dim))
        return x

    def _Rosenbrock_optimal(self):
        x = np.array([[1] * self.input_dim])
        y = 0
        return x, y

    def _Michalewicz_func(self, x):
        m = 10
        i = np.arange(x.shape[-1]) + 1
        term = np.sin(x) * (np.sin((i * x**2) / np.pi) ** (2 * m))
        y = -term.sum(-1)
        return y

    def _Michalewicz_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, 0, np.pi)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, 0, np.pi)
        return x

    def _Michalewicz_sample(self, num):
        x = np.random.uniform(0, np.pi, (num, self.input_dim))
        return x

    def _Michalewicz_optimal(self):
        x = np.array([[2.20, 1.57]])
        y = -1.8013
        return x, y

    def _Levy_func(self, x):
        w = 1 + (x - 1) / 4
        term1 = np.square(np.sin(np.pi * w[:, 0]))
        term3 = np.square((w[:, -1] - 1)) * (
            1 + np.square(np.sin(2 * np.pi * w[:, -1]))
        )
        term2 = np.square((w[:, :-1] - 1)) * (
            1 + 10 * np.square(np.sin(np.pi * w[:, :-1] + 1))
        )
        y = term1 + term2.sum(-1) + term3
        return y

    def _Levy_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, -10, 10)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -10, 10)
        return x

    def _Levy_sample(self, num):
        x = np.random.uniform(-10, 10, (num, self.input_dim))
        return x

    def _Levy_optimal(self):
        x = np.array([[1] * self.input_dim])
        y = 0
        return x, y

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

    def _Branin_optimal(self):
        x = np.array([[9.42478, 2.475]])
        y = 0.397887
        return x, y

    def _Bukin_func(self, x):
        term1 = 100 * np.sqrt(np.abs(x[:, 1] - 0.01 * x[:, 0] ** 2))
        term2 = 0.01 * np.abs(x[:, 0] + 10)
        y = term1 + term2
        return y

    def _Bukin_bound(self, x):
        if isinstance(x, np.ndarray):
            x[:, 0] = np.clip(x[:, 0], -15, 5)
            x[:, 1] = np.clip(x[:, 1], -3, 1)
        elif isinstance(x, torch.Tensor):
            x[:, 0] = torch.clamp(x[:, 0], -15, 5)
            x[:, 1] = torch.clamp(x[:, 1], -3, 1)
        return x

    def _Bukin_sample(self, num):
        x1 = np.random.uniform(-15, 5, (num, 1))
        x2 = np.random.uniform(-3, 1, (num, 1))
        x = np.concatenate([x1, x2], -1)
        return x

    def _Bukin_optimal(self):
        x = np.array([[-10, 1]])
        y = 0
        return x, y

    def _Schwefel_func(self, x):
        """
        x in [-500, 500]
        global minimum 0 at (420.9687,...,420.9687)
        """
        x *= 100
        d = x.shape[-1]
        y = 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)
        return y

    def _Schwefel_bound(self, x):
        if isinstance(x, np.ndarray):
            x = np.clip(x, -5, 5)
        elif isinstance(x, torch.Tensor):
            x = torch.clamp(x, -5, 5)
        return x

    def _Schwefel_sample(self, num):
        x = np.random.uniform(-5, 5, (num, self.input_dim))
        return x

    def _Schwefel_optimal(self):
        x = np.array([[4.209687] * self.input_dim])
        y = 0
        return x, y

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

    def _Ackley_optimal(self):
        x = np.array([[0] * self.input_dim])
        y = 0
        return x, y

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
            temp_y += alpha[i] * np.exp(-np.sum(A[i, :] * (x - P[i, :]) ** 2, axis=-1))

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

    def _Hartmann_optimal(self):
        x = np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]])
        y = -3.32237
        return x, y

    def call(self, x):
        return self.function(x)

    def local(self):
        x, y_true = self.optimal_action()
        y_call = self.call(x)[0]
        print(f"x: {x}")
        print(f"y_true: {y_true} y_call: {y_call} error: {abs(y_true - y_call)}")
        return x, y_call


def Plot(func_name):
    from matplotlib import pyplot as plt

    blackbox = BlackBox(func_name, 2)

    if func_name == "Ackley":
        xx = np.arange(-32, 32, 0.05)
        yy = np.arange(-32, 32, 0.05)
    elif func_name == "Michalewicz":
        xx = np.arange(0, np.pi, 0.05)
        yy = np.arange(0, np.pi, 0.05)
    elif func_name == "Levy":
        xx = np.arange(-10, 10, 0.05)
        yy = np.arange(-10, 10, 0.05)
    elif func_name == "Rosenbrock":
        xx = np.arange(-10, 10, 0.05)
        yy = np.arange(-6, 6, 0.05)
    else:
        return
    X, Y = np.meshgrid(xx, yy)
    XY = np.stack([X.flatten(), Y.flatten()], 1)
    Z = blackbox.call(XY)
    Z = Z.reshape(X.shape)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="rainbow")
    if func_name == "Ackley":
        ax.set_xlim(32, -32)
    elif func_name == "Michalewicz":
        ax.set_xlim(np.pi, 0)
    elif func_name == "Levy":
        ax.set_xlim(10, -10)
    elif func_name == "Rosenbrock":
        ax.set_xlim(10, -10)
    else:
        return
    plt.savefig(f"test_{func_name}")


if __name__ == "__main__":
    for name in FUNC_NAMES:
        print(f"{name}:")
        blackbox = BlackBox(name, 10)
        blackbox.local()
        print(f"==" * 10)
        Plot(name)
