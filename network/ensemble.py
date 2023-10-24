from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(inp_dim, out_dim, hidden_sizes, bias=True):
    if len(hidden_sizes) == 0:
        return nn.Linear(inp_dim, out_dim, bias=bias)
    model = [nn.Linear(inp_dim, hidden_sizes[0], bias=bias)]
    model += [nn.ReLU(inplace=True)]
    for i in range(1, len(hidden_sizes)):
        model += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i], bias=bias)]
        model += [nn.ReLU(inplace=True)]
    if out_dim != 0:
        model += [nn.Linear(hidden_sizes[-1], out_dim, bias=bias)]
    return nn.Sequential(*model)


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
            self.register_parameter("bias", None)
        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def reset_parameters(self) -> None:
        # init weight
        if self.weight_init == "sDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(
                np.float32
            )
            weight = weight / np.linalg.norm(weight, axis=1, keepdims=True)
            self.weight = nn.Parameter(
                torch.from_numpy(self.prior_std * weight).float()
            )
        elif self.weight_init == "gDB":
            weight = np.random.randn(self.out_features, self.in_features).astype(
                np.float32
            )
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
                            torch.zeros((self.action_dim, self.hidden_dim))
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
        return "in_features={}, out_features={}, bias={}".format(
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
        hyperlayer_params = dict(
            noise_dim=noise_dim,
            hidden_dim=out_features,
            prior_std=prior_std,
            out_type="weight",
            device=device,
        )
        self.hyper_weight = HyperLayer(
            **hyperlayer_params, trainable=True, weight_init="xavier_normal"
        )
        self.prior_weight = HyperLayer(
            **hyperlayer_params, trainable=False, weight_init="sDB"
        )

        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, z, x, prior_x):
        theta = self.hyper_weight(z)
        prior_theta = self.prior_weight(z)

        if len(x.shape) > 2:
            # compute feel-good term
            out = torch.einsum("bd,bad -> ba", theta, x)
            prior_out = torch.einsum("bd,bad -> ba", prior_theta, prior_x)
        elif x.shape[0] != z.shape[0]:
            # compute action value for one action set
            out = torch.mm(theta, x.T)
            prior_out = torch.mm(prior_theta, prior_x.T)
        else:
            # compute predict reward in batch
            out = torch.einsum("bnw,bw -> bn", theta, x)
            prior_out = torch.einsum("bnw,bw -> bn", theta, x)

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


class EnsembleNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (),
        ensemble_sizes: Sequence[int] = (),
        noise_dim: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.basedmodel = mlp(in_features, 0, hidden_sizes)
        self.out = nn.ModuleList(
            [mlp(hidden_sizes[-1], 1, ensemble_sizes) for _ in range(noise_dim)]
        )
        if prior_scale > 0:
            self.priormodel = mlp(in_features, 0, hidden_sizes)
            self.prior_out = nn.ModuleList(
                [mlp(hidden_sizes[-1], 1, ensemble_sizes) for _ in range(noise_dim)]
            )
            for param in self.priormodel.parameters():
                param.requires_grad = False
            for param in self.prior_out.parameters():
                param.requires_grad = False

        self.ensemble_num = noise_dim
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.device = device

    def forward(self, z, x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        logits = self.basedmodel(x)
        if self.prior_scale > 0:
            prior_logits = self.priormodel(x)
        if z.shape[0] == 1:
            ensemble_index = torch.where(z == 1)[1].cpu().int()
            out = self.out[ensemble_index](logits)
            if self.prior_scale > 0:
                prior_out = self.prior_out[ensemble_index](prior_logits)
                out = self.posterior_scale * out + self.prior_scale * prior_out
        else:
            out = [self.out[k](logits) for k in range(self.ensemble_num)]
            out = torch.stack(out, dim=1)
            if self.prior_scale > 0:
                prior_out = [
                    self.prior_out[k](prior_logits) for k in range(self.ensemble_num)
                ]
                prior_out = torch.stack(prior_out, dim=1)
                out = self.posterior_scale * out + self.prior_scale * prior_out
        return out.squeeze(-1)

    def regularization(self, z):
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        return self.out.regularization(z)
