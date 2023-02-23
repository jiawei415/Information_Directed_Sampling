import os
import copy
import random as rd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    np.random.seed(seed)
    rd.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)


def bound_point(now_query):
    now_query[now_query > 1] = 1
    now_query[now_query < 0] = 0
    return now_query


def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return rd.choice(indices)


def mlp(input_dim, hidden_sizes, linear_layer=nn.Linear):
    model = []
    if len(hidden_sizes) > 0:
        hidden_sizes = [input_dim] + list(hidden_sizes)
        for i in range(1, len(hidden_sizes)):
            model += [linear_layer(hidden_sizes[i - 1], hidden_sizes[i])]
            model += [nn.ReLU(inplace=True)]
    model = nn.Sequential(*model)
    return model


class HyperLayer(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        feature_dim: int,
        prior_std: float = 1.0,
        use_bias: bool = True,
        trainable: bool = True,
        out_type: str = "weight",
        weight_init: str = "xavier_normal",
        bias_init: str = "xavier-uniform",
        device: str = "cpu",
    ):
        super().__init__()
        assert out_type in ["weight", "bias"], f"No out type {out_type} in HyperLayer"
        self.noise_dim = noise_dim
        self.feature_dim = feature_dim
        self.prior_std = prior_std
        self.use_bias = use_bias
        self.trainable = trainable
        self.out_type = out_type
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.device = device

        self.in_features = noise_dim
        if out_type == "weight":
            self.out_features = feature_dim
        elif out_type == "bias":
            self.out_features = 1

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        # init weight
        if self.weight_init == "DB":
            weight = np.random.randn(self.out_features, self.in_features).astype(
                np.float32
            )
            weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
            self.weight = nn.Parameter(
                torch.from_numpy(self.prior_std * weight).float()
            )
        elif self.weight_init == "trunc_normal":
            bound = 1.0 / np.sqrt(self.in_features)
            nn.init.trunc_normal_(self.weight, std=bound, a=-2 * bound, b=2 * bound)
        elif self.weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=1.0)
        elif self.weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=1.0)
        else:
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # init bias
        if self.use_bias:
            if self.bias_init == "default":
                bound = 1.0 / np.sqrt(self.in_features)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                weight_bias_init, bias_bias_init = self.bias_init.split("-")
                if self.out_type == "weight":
                    if weight_bias_init == "zeros":
                        nn.init.zeros_(self.bias)
                    elif weight_bias_init == "sphere":
                        bias = np.random.randn(self.out_features).astype(np.float32)
                        bias = bias / np.linalg.norm(bias)
                        self.bias = nn.Parameter(
                            torch.from_numpy(self.prior_std * bias).float()
                        )
                    elif weight_bias_init == "xavier":
                        bias = nn.init.xavier_normal_(
                            torch.zeros((1, self.feature_dim))
                        )
                        self.bias = nn.Parameter(bias.flatten())
                elif self.out_type == "bias":
                    if bias_bias_init == "zeros":
                        nn.init.zeros_(self.bias)
                    elif bias_bias_init == "sphere":
                        bias = np.random.randn(self.out_features).astype(np.float32)
                        bias = bias / np.linalg.norm(bias)
                        self.bias = nn.Parameter(
                            torch.from_numpy(self.prior_std * bias).float()
                        )
                    elif bias_bias_init == "uniform":
                        bound = 1 / np.sqrt(self.feature_dim)
                        nn.init.uniform_(self.bias, -bound, bound)
                    elif bias_bias_init == "pos":
                        bias = 1 * np.ones(self.out_features)
                        self.bias = nn.Parameter(torch.from_numpy(bias).float())
                    elif bias_bias_init == "neg":
                        bias = -1 * np.ones(self.out_features)
                        self.bias = nn.Parameter(torch.from_numpy(bias).float())

    def forward(self, z: torch.Tensor):
        z = z.to(self.device)
        return F.linear(z, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class HyperLinear(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        feature_dim: int,
        prior_std: float or np.ndarray = 1.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        hyperlayer_params = dict(
            noise_dim=noise_dim,
            feature_dim=feature_dim,
            prior_std=prior_std,
            bias_init="xavier-uniform",
            device=device,
        )
        self.hyper_weight = HyperLayer(
            **hyperlayer_params,
            use_bias=True,
            trainable=True,
            weight_init="xavier_normal",
            out_type="weight",
        )
        self.prior_weight = HyperLayer(
            **hyperlayer_params,
            use_bias=True,
            trainable=False,
            weight_init="DB",
            out_type="weight",
        )
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, z, x, prior_x):
        theta = self.hyper_weight(z)
        prior_theta = self.prior_weight(z)

        out = torch.mul(x, theta).sum(-1, keepdim=True)
        prior_out = torch.mul(prior_x, prior_theta).sum(-1, keepdim=True)

        out = self.posterior_scale * out + self.prior_scale * prior_out
        return out

    def regularization(self, z):
        theta = self.hyper_weight(z)
        reg_loss = theta.pow(2).mean()
        return reg_loss

    def get_thetas(self, z):
        theta = self.hyper_weight(z)
        prior_theta = self.prior_weight(z)
        theta = self.posterior_scale * theta + self.prior_scale * prior_theta
        return theta


class Net(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        in_features: int,
        hidden_sizes: list = [],
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.basedmodel = mlp(in_features, hidden_sizes)
        self.priormodel = mlp(in_features, hidden_sizes)
        for param in self.priormodel.parameters():
            param.requires_grad = False

        feature_dim = in_features if len(hidden_sizes) == 0 else hidden_sizes[-1]
        self.out = HyperLinear(
            noise_dim=noise_dim,
            feature_dim=feature_dim,
            prior_scale=prior_scale,
            posterior_scale=posterior_scale,
            device=device,
        )
        self.device = device

    def forward(self, z, x):
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        logits = self.basedmodel(x)
        prior_logits = self.priormodel(x)
        out = self.out(z, logits, prior_logits)
        return out

    def regularization(self, z):
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        return self.out.regularization(z)


class ReplayBuffer:
    def __init__(self, buffer_size, buffer_shape):
        print(buffer_shape)
        self.buffers = {
            key: np.empty([buffer_size, *shape]) for key, shape in buffer_shape.items()
        }
        self.noise_dim = buffer_shape["z"][-1]
        self.sample_num = 0

    def __len__(self):
        return self.sample_num

    def _unit_sphere_noise(self):
        noise = np.random.randn(self.noise_dim).astype(np.float32)
        noise /= np.linalg.norm(noise)
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
        z = self._unit_sphere_noise()
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
        norm_coef: float = 0.01,
        target_noise_coef: float = 0.01,
        buffer_size: int = 10000,
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
        self.norm_coef = norm_coef
        self.target_noise_coef = target_noise_coef
        self.buffer_size = buffer_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.action_num = action_num
        self.action_lr = action_lr
        self.action_max_update = action_max_update

        self.__init_model_optimizer()
        self.__init_buffer()

    def __init_model_optimizer(self):
        # init hypermodel
        self.model = Net(
            self.noise_dim,
            self.input_dim,
            self.hidden_sizes,
            self.prior_scale,
            self.posterior_scale,
            self.device,
        ).to(self.device)
        print(f"Network structure:\n{str(self.model)}")
        # init optimizer
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9
            )
        else:
            raise NotImplementedError

    def __init_buffer(self):
        # init replay buffer
        buffer_shape = {"x": (self.input_dim,), "y": (), "z": (self.noise_dim,)}
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape)

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

        update_noise = self.generate_noise(self.batch_size)  # sample noise for update
        target_noise = (
            torch.mul(z_batch, update_noise).sum(-1) * self.target_noise_coef
        )  # noise for target
        predict = self.model(update_noise, x_batch).squeeze(-1)
        diff = target_noise + y_batch - predict
        loss = diff.pow(2).mean()
        norm_coef = self.norm_coef / len(self.buffer)
        reg_loss = self.model.regularization(update_noise) * norm_coef
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def get_thetas(self, M=1):
        assert len(self.hidden_sizes) == 0, f"hidden size > 0"
        action_noise = self.generate_noise(M)
        with torch.no_grad():
            thetas = self.model.out.get_thetas(action_noise).cpu().numpy()
        return thetas

    def predict(self, features, M=1):
        action_noise = self.generate_noise(M)
        with torch.no_grad():
            out = self.model(action_noise, features).cpu().numpy()
        return out

    def generate_noise(self, batch_size):
        noise = (
            torch.randn(batch_size, self.noise_dim).type(torch.float32).to(self.device)
        )
        return noise

    def reset(self):
        self.__init_model_optimizer()

    def select_action(self):
        # init actions
        actions = np.random.rand(self.action_num, self.input_dim).astype(np.float32)
        actions = torch.from_numpy(actions).to(self.device)
        actions.requires_grad = True
        noise = self.generate_noise(actions.shape[0])  # sample noise
        model = copy.deepcopy(self.model)
        model.requires_grad = False
        optim = torch.optim.Adam([actions], lr=self.action_lr)
        # optimze actions
        for _ in range(self.action_max_update):
            out = model(noise, actions)
            out = -out.mean()
            optim.zero_grad()
            out.backward()
            optim.step()
            with torch.no_grad():
                actions = bound_point(actions)
        out = model(noise, actions)
        out = out.detach().cpu().numpy()
        act_index = rd_argmax(out.squeeze(-1))
        actions = actions.detach().cpu().numpy()
        return actions[act_index]

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
