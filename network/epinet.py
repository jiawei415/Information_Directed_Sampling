from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

ModuleType = Type[nn.Module]


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

class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, out_features, in_features)
        )
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=np.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.bmm(x, self.weight.transpose(1, 2)) + self.bias
        return out

    def extra_repr(self) -> str:
        return 'ensemble_size={}, in_features={}, out_features={}, bias={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )

class EnsemblePrior(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        ensemble_num: int,
        ensemble_sizes: Sequence[int] = [5, 5],
    ):
        super().__init__()
        ensemble_sizes = [in_features] + ensemble_sizes + [out_features]
        layers = []
        for i in range(len(ensemble_sizes) - 1):
            layers.append(VectorizedLinear(ensemble_sizes[i], ensemble_sizes[i + 1], ensemble_num))
            if i < len(ensemble_sizes) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.basedmodel = nn.Sequential(*layers)

        self.device = device
        self.ensemble_num = ensemble_num

    def forward(
        self, x: torch.Tensor, noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Any]:
        x = x.unsqueeze(0).repeat_interleave(self.ensemble_num, dim=0)
        out = self.basedmodel(x)
        out = out.transpose(0, 1)
        out = torch.bmm(noise, out)
        return out


class EpiLinear(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        state_dim: int,
        hidden_dim: int,
        class_num: int = 1,
        epinet_sizes: Sequence[int] = (15,),
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        epinet_init: str = "xavier_normal",
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.epinet_init = epinet_init
        self.in_features = noise_dim + hidden_dim + state_dim
        self.out_features = noise_dim * class_num
        self.epinet = mlp(self.in_features, self.out_features, epinet_sizes)
        if prior_scale > 0:
            self.priornet = EnsemblePrior(state_dim, class_num, device, noise_dim)
            for param in self.priornet.parameters():
                param.requires_grad = False
        self.reset_parameters()

        self.noise_dim = noise_dim
        self.class_num = class_num
        self.posterior_scale = posterior_scale
        self.prior_scale = prior_scale
        self.device = device

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(param)
            elif "weight" in name:
                if self.epinet_init == "trunc_normal":
                    bound = 1.0 / np.sqrt(param.shape[-1])
                    torch.nn.init.trunc_normal_(
                        param, std=bound, a=-2 * bound, b=2 * bound
                    )
                elif self.epinet_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(param, gain=1.0)
                elif self.epinet_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(param, gain=1.0)

    def forward(
        self, x: torch.Tensor, feature: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        epinet_inp = torch.cat([x, feature], dim=-1)
        if len(z.shape) == 2:
            z = z.unsqueeze(1)
            if z.shape[0] == 1:
                z = z.repeat(batch_size, 1, 1)
        epinet_inp = epinet_inp.unsqueeze(1).repeat(1, z.shape[1], 1)
        epinet_inp = torch.cat([epinet_inp, z], dim=-1)
        out = self.epinet(epinet_inp)
        out = out.view(batch_size, -1, self.noise_dim, self.class_num)
        out = torch.einsum("bsna, bsn -> bsa", out, z)
        if self.prior_scale > 0:
            prior_out = self.priornet(x, z)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out


class EpiNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        action_num: int = 1,
        hidden_sizes: Sequence[int] = (),
        noise_dim: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        feature_sg: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()

        self.basedmodel = mlp(in_features, 0, hidden_sizes)
        self.based_out = nn.Linear(hidden_sizes[-1], action_num)

        based_in_feature = self.based_out.in_features
        self.epi_out = EpiLinear(
            noise_dim,
            in_features,
            based_in_feature,
            action_num,
            [15, 15],
            prior_scale,
            posterior_scale,
            device=device,
        )
        self.class_num = action_num
        self.feature_sg = feature_sg
        self.device = device

    def forward(self, z, x):
        z = torch.as_tensor(z, device=self.device, dtype=torch.float32)
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        logits = self.basedmodel(x)
        based_out = self.based_out(logits)
        if self.feature_sg:
            logits = logits.detach()
        epi_out = self.epi_out(x, logits, z)
        out = epi_out + based_out.unsqueeze(1)
        out = out.squeeze(-1)
        if out.shape[1] == 1 and z.shape[0] == 1:
            out = out.squeeze(1)
        return out
