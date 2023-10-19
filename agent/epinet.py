from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn

ModuleType = Type[nn.Module]

def mlp(input_dim, hidden_sizes, linear_layer=nn.Linear):
    model = []
    if len(hidden_sizes) > 0 :
        hidden_sizes = [input_dim] + list(hidden_sizes)
        for i in range(1, len(hidden_sizes)):
            model += [linear_layer(hidden_sizes[i-1], hidden_sizes[i])]
            model += [nn.ReLU(inplace=True)]
    model = nn.Sequential(*model)
    return model


class EnsemblePrior(nn.Module):
    def __init__(
        self,
        in_features: int,
        device: Optional[Union[str, int, torch.device]],
        ensemble_num: int,
        ensemble_sizes: Sequence[int] = [5],
    ):
        super().__init__()
        self.basedmodel = nn.ModuleList([
            mlp(in_features, ensemble_sizes + [1]) for _ in range(ensemble_num)
        ])

        self.device = device
        self.ensemble_num = ensemble_num
        self.head_list = list(range(self.ensemble_num))

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> Tuple[torch.Tensor, Any]:
        out = [self.basedmodel[k](x) for k in self.head_list]
        out = torch.hstack(out)
        out = torch.einsum("bz, bnz -> bn", out, noise)
        return out


class EpiLinear(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        state_dim: int,
        hidden_dim: int,
        epinet_sizes: Sequence[int],
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
        epinet_init: str = "xavier_normal",
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.epinet_init = epinet_init
        self.in_features = noise_dim + hidden_dim + state_dim
        self.out_features = noise_dim
        self.epinet = mlp(self.in_features, epinet_sizes + [self.out_features])
        if prior_scale > 0:
            self.priornet = EnsemblePrior(state_dim, device, noise_dim)
            for param in self.priornet.parameters():
                param.requires_grad = False
        self.reset_parameters()

        self.noise_dim = noise_dim
        self.posterior_scale = posterior_scale
        self.prior_scale = prior_scale
        self.device = device

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(param)
            elif 'weight' in name:
                if self.epinet_init == "trunc_normal":
                    bound = 1.0 / np.sqrt(param.shape[-1])
                    torch.nn.init.trunc_normal_(param, std=bound, a=-2*bound, b=2*bound)
                elif self.epinet_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(param, gain=1.0)
                elif self.epinet_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(param, gain=1.0)

    def forward(self, x: torch.Tensor, feature: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        epinet_inp = torch.cat([x, feature], dim=-1)
        if len(z.shape) == 2:
            z = z.unsqueeze(1).repeat(batch_size, 1, 1)
        epinet_inp = epinet_inp.unsqueeze(1).repeat(1, z.shape[1], 1)
        epinet_inp = torch.cat([epinet_inp, z], dim=-1)
        out = self.epinet(epinet_inp)
        out = out.view(batch_size, -1, self.noise_dim)
        out = torch.einsum('bsn, bsn -> bs', out, z)
        if self.prior_scale > 0:
            prior_out = self.priornet(x, z)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        out = out.squeeze(1)
        return out


class EpiNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (),
        noise_dim: int = 2,
        prior_std: float = 1.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        feature_sg: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()

        self.basedmodel = mlp(in_features, hidden_sizes)
        self.based_out = nn.Linear(hidden_sizes[-1], 1)

        based_in_feature = self.based_out.in_features
        self.epi_out = EpiLinear(
            noise_dim, in_features, based_in_feature, [15], prior_scale, posterior_scale, device=device
        )
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
        if len(epi_out.shape) == 3:
            based_out = based_out.unsqueeze(1).repeat(1, z.shape[1], 1)
        out = epi_out + based_out
        return out
