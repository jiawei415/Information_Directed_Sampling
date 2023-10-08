from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import itertools as it
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim, hidden_sizes, linear_layer=nn.Linear):
    model = []
    if len(hidden_sizes) > 0 :
        hidden_sizes = [input_dim] + list(hidden_sizes)
        for i in range(1, len(hidden_sizes)):
            model += [linear_layer(hidden_sizes[i-1], hidden_sizes[i])]
            model += [nn.ReLU(inplace=True)]
    model = nn.Sequential(*model)
    return model


class HyperLayer(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        action_dim: int = 1,
        prior_std: float = 1.0,
        use_bias: bool = True,
        trainable: bool = True,
        out_type: str = "weight",
        weight_init: str = "xavier_normal",
        bias_init: str = "sphere-sphere",
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        assert out_type in ["weight", "bias"], f"No out type {out_type} in HyperLayer"
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.prior_std = prior_std
        self.use_bias = use_bias
        self.trainable = trainable
        self.out_type = out_type
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.device = device

        self.in_features = noise_dim
        if out_type == "weight":
            self.out_features = action_dim * hidden_dim
        elif out_type == "bias":
            self.out_features = action_dim

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        # init weight
        if self.weight_init == "sDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(np.float32)
            weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
            self.weight = nn.Parameter(torch.from_numpy(self.prior_std * weight).float())
        elif self.weight_init == "gDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(np.float32)
            self.weight = nn.Parameter(torch.from_numpy(self.prior_std * weight).float())
        elif self.weight_init == "trunc_normal":
            bound = 1.0 / np.sqrt(self.in_features)
            nn.init.trunc_normal_(self.weight, std=bound, a=-2*bound, b=2*bound)
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
                        self.bias = nn.Parameter(torch.from_numpy(self.prior_std * bias).float())
                    elif weight_bias_init == "xavier":
                        bias = nn.init.xavier_normal_(torch.zeros((self.action_dim, self.hidden_dim)))
                        self.bias = nn.Parameter(bias.flatten())
                elif self.out_type == "bias":
                    if bias_bias_init == "zeros":
                        nn.init.zeros_(self.bias)
                    elif bias_bias_init == "sphere":
                        bias = np.random.randn(self.out_features).astype(np.float32)
                        bias = bias / np.linalg.norm(bias)
                        self.bias = nn.Parameter(torch.from_numpy(self.prior_std * bias).float())
                    elif bias_bias_init == "uniform":
                        bound = 1 / np.sqrt(self.hidden_dim)
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
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class HyperLinear(nn.Module):
    def __init__(
        self,
        noise_dim,
        out_features,
        prior_std: float = 1.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()
        hyperlayer_params = dict(noise_dim=noise_dim, hidden_dim=out_features, prior_std=prior_std, out_type="weight", device=device)
        self.hyper_weight = HyperLayer(**hyperlayer_params, trainable=True, weight_init="xavier_normal")
        self.prior_weight = HyperLayer(**hyperlayer_params, trainable=False, weight_init="sDB")

        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, z, x, prior_x):
        theta = self.hyper_weight(z)
        prior_theta = self.prior_weight(z)

        if len(x.shape) > 2:
            # compute feel-good term
            out = torch.einsum('bd,bad -> ba', theta, x)
            prior_out = torch.einsum('bd,bad -> ba', prior_theta, prior_x)
        elif x.shape[0] != z.shape[0]:
            # compute action value for one action set
            out = torch.mm(theta, x.T)
            prior_out = torch.mm(prior_theta, prior_x.T)
        else:
            # compute predict reward in batch
            out = torch.einsum('bnw,bw -> bn', theta, x)
            prior_out = torch.einsum('bnw,bw -> bn', theta, x)

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
        in_features: int,
        hidden_sizes: Sequence[int] = (),
        noise_dim: int = 2,
        prior_std: float = 1.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.basedmodel = mlp(in_features, hidden_sizes)
        self.priormodel = mlp(in_features, hidden_sizes)
        for param in self.priormodel.parameters():
            param.requires_grad = False

        hyper_out_features = in_features if len(hidden_sizes) == 0 else hidden_sizes[-1]
        self.out = HyperLinear(
            noise_dim, hyper_out_features, prior_std, prior_scale, posterior_scale, device
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
        self.buffers = {
            key: np.empty([buffer_size, *shape]) for key, shape in buffer_shape.items()
        }
        self.noise_dim = buffer_shape['z'][-1]
        self.sample_num = 0

    def __len__(self):
        return self.sample_num

    def _unit_sphere_noise(self):
        noise = np.random.randn(self.noise_dim).astype(np.float32)
        noise /= np.linalg.norm(noise)
        return noise

    def _sample(self, index):
        a_data = self.buffers['a'][:self.sample_num]
        s_data = self.buffers['s'][:self.sample_num]
        f_data = s_data[np.arange(self.sample_num), a_data.astype(np.int32)]
        r_data = self.buffers['r'][:self.sample_num]
        z_data = self.buffers['z'][:self.sample_num]
        s_data, f_data, r_data, z_data \
            = s_data[index], f_data[index], r_data[index], z_data[index]
        return s_data, f_data, r_data, z_data

    def reset(self):
        self.sample_num = 0

    def put(self, transition):
        for k, v in transition.items():
            self.buffers[k][self.sample_num] = v
        z = self._unit_sphere_noise()
        self.buffers['z'][self.sample_num] = z
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


class HyperModel:
    def __init__(
        self,
        noise_dim: int,
        n_action: int,
        n_feature: int,
        hidden_sizes: Sequence[int] = (),
        prior_std: float = 1.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        batch_size: int = 32,
        lr: float = 0.01,
        optim: str = 'Adam',
        fg_lambda: float = 0.0,
        fg_decay: bool = True,
        norm_coef: float = 0.01,
        target_noise_coef: float = 0.01,
        buffer_size: int = 10000,
        NpS: int = 20,
        action_noise: str = "sgs",
        update_noise: str = "pn",
        reset: bool = False,
    ):

        self.noise_dim = noise_dim
        self.action_dim = n_action
        self.feature_dim = n_feature
        self.hidden_sizes = hidden_sizes
        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.lr = lr
        self.fg_lambda = fg_lambda
        self.fg_decay = fg_decay
        self.batch_size = batch_size
        self.NpS = NpS
        self.optim = optim
        self.norm_coef = norm_coef
        self.target_noise_coef = target_noise_coef
        self.buffer_size = buffer_size
        self.action_noise = action_noise
        self.update_noise = update_noise
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.__init_model_optimizer()
        self.__init_buffer()
        self.update = getattr(self, '_update_reset') if reset else getattr(self, '_update')

    def __init_model_optimizer(self):
        # init hypermodel
        self.model = Net(
            self.feature_dim, self.hidden_sizes, self.noise_dim, self.prior_std,
            self.prior_scale, self.posterior_scale, device=self.device
        ).to(self.device)
        print(f"\nNetwork structure:\n{str(self.model)}")
         # init optimizer
        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optim == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError

    def __init_buffer(self):
        # init replay buffer
        buffer_shape = {
            's': (self.action_dim, self.feature_dim),
            'a': (),
            'r': (),
            'z': (self.noise_dim, )
        }
        self.buffer = ReplayBuffer(self.buffer_size, buffer_shape)

    def _update(self):
        s_batch, f_batch, r_batch, z_batch = self.buffer.sample(self.batch_size)
        self.learn(s_batch, f_batch, r_batch, z_batch)

    def _update_reset(self):
        sample_num = len(self.buffer)
        if sample_num > self.batch_size:
            s_data, f_data, r_data, z_data = self.buffer.get()
            for i in range(0, self.batch_size, sample_num):
                s_batch, f_batch, r_batch, z_batch \
                    = s_data[i:i+self.batch_size], f_data[i:i+self.batch_size], r_data[i: i+self.batch_size], z_data[i:i+self.batch_size]
                self.learn(s_batch, f_batch, r_batch, z_batch)
            if sample_num % self.batch_size != 0:
                last_sample = sample_num % self.batch_size
                index1 = -np.arange(1, last_sample + 1).astype(np.int32)
                index2 = np.random.randint(low=0, high=sample_num, size=self.batch_size-last_sample)
                index = np.hstack([index1, index2])
                s_batch, f_batch, r_batch, z_batch = s_data[index], f_data[index], r_data[index], z_data[index]
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

        update_noise = self.generate_update_noise((self.batch_size, self.NpS), self.update_noise) # sample noise for update
        target_noise = torch.mul(z_batch.unsqueeze(1), update_noise).sum(-1) * self.target_noise_coef # noise for target
        predict = self.model(update_noise, f_batch)
        diff = target_noise + r_batch.unsqueeze(-1) - predict
        diff = diff.pow(2).mean(-1)
        if self.fg_lambda:
            fg_lambda = self.fg_lambda / np.sqrt(len(self.buffer)) if self.fg_decay else self.fg_lambda
            fg_term = self.model(update_noise, s_batch)
            fg_term = fg_term.max(dim=-1)[0]
            loss = (diff - fg_lambda * fg_term).mean()
        else:
            loss = diff.mean()
        norm_coef = self.norm_coef / len(self.buffer)
        reg_loss = self.model.regularization(update_noise) * norm_coef
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_thetas(self, M=1):
        assert len(self.hidden_sizes) == 0, f'hidden size > 0'
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
            noise =  self.gen_gs_noise(noise_num)
        elif noise_type == "sp":
            noise =  self.gen_sp_noise(noise_num)
        elif noise_type == "sgs":
            noise =  self.gen_sgs_noise(noise_num)
        elif noise_type == "oh":
            noise =  self.gen_one_hot_noise(noise_num)
        elif noise_type == "pn":
            noise =  self.gen_pn_one_noise(noise_num)
        return noise

    def generate_update_noise(self, noise_num, noise_type):
        if noise_type == "gs":
            noise =  self.gen_gs_noise(noise_num)
        elif noise_type == "sp":
            noise =  self.gen_sp_noise(noise_num)
        elif noise_type == "sgs":
            noise =  self.gen_sgs_noise(noise_num)
        elif noise_type == "oh":
            batch_size = noise_num[0]
            noise = torch.arange(self.noise_dim).unsqueeze(0).repeat(batch_size, 1)
            noise = F.one_hot(noise, self.noise_dim).to(torch.float32).to(self.device)
        elif noise_type == "pn":
            batch_size = noise_num[0]
            noise = np.array(list((it.product(range(2), repeat=self.noise_dim))))
            noise[np.where(noise==0)] = -1
            noise = torch.from_numpy(noise).to(torch.float32).to(self.device)
            noise = noise.unsqueeze(0).repeat(batch_size, 1, 1)
        return noise

    def gen_gs_noise(self, noise_num):
        noise_shape = noise_num + (self.noise_dim, )
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
        noise_shape = noise_num + (self.noise_dim, )
        noise = torch.ones(noise_shape) + torch.randint(-1, 1, noise_shape) * 2
        return noise.to(self.device)

    def reset(self):
        self.__init_model_optimizer()
