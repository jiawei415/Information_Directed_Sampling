from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(inp_dim, out_dim, hidden_sizes, bias=True):
    if len(hidden_sizes) == 0 and out_dim == 0:
        return nn.Identity()
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
        # if z.shape[0] == 1:
        #     enemble_index = int(np.where(z == 1)[1])
        #     out = F.linear(x, self.weight[enemble_index], self.bias[enemble_index])
        # else:
        #     out = torch.bmm(x, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)
        #     out = out.transpose(0, 1)
        #     # out = torch.einsum("eio,bi->beo", self.weight.transpose(1, 2), x) + self.bias.unsqueeze(0)
        return out

    def extra_repr(self) -> str:
        return 'ensemble_size={}, in_features={}, out_features={}, bias={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.bias is not None
        )

class EnsembleNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_sizes: Sequence[int] = (),
        ensemble_sizes: Sequence[int] = (),
        action_num: int = 1,
        noise_dim: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        based_prior: float = False,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        feature_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else in_features
        ensemble_sizes = [feature_dim] + ensemble_sizes

        self.basedmodel = mlp(in_features, 0, hidden_sizes)
        out = []
        for i in range(len(ensemble_sizes) - 1):
            out.append(VectorizedLinear(ensemble_sizes[i], ensemble_sizes[i + 1], noise_dim))
            out.append(nn.ReLU(inplace=True))
        out.append(VectorizedLinear(ensemble_sizes[-1], action_num, noise_dim))
        self.out = nn.Sequential(*out)

        if prior_scale > 0:
            if based_prior:
                self.priormodel = mlp(in_features, 0, hidden_sizes)
                for param in self.priormodel.parameters():
                    param.requires_grad = False
            prior_out = []
            for i in range(len(ensemble_sizes) - 1):
                prior_out.append(VectorizedLinear(ensemble_sizes[i], ensemble_sizes[i + 1], noise_dim))
                prior_out.append(nn.ReLU(inplace=True))
            prior_out.append(VectorizedLinear(ensemble_sizes[-1], action_num, noise_dim))
            self.prior_out = nn.Sequential(*prior_out)
            for param in self.prior_out.parameters():
                param.requires_grad = False

        self.ensemble_num = noise_dim
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.based_prior = based_prior
        self.device = device

        # self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.out.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param, gain=1.0)
        if self.prior_scale > 0:
            for name, param in self.prior_out.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.xavier_normal_(param, gain=1.0)

    def forward(self, z, x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        logits = self.basedmodel(x)
        logits = logits.unsqueeze(0).repeat_interleave(self.ensemble_num, dim=0)
        out = self.out(logits)
        if self.prior_scale > 0:
            if self.based_prior:
                prior_logits = self.priormodel(x)
                prior_logits = prior_logits.unsqueeze(0).repeat_interleave(self.ensemble_num, dim=0)
            else:
                prior_logits = logits.detach()
            prior_out = self.prior_out(prior_logits)
            out = self.posterior_scale * out + self.prior_scale * prior_out
        if z.shape[0] == 1:
            if isinstance(z, np.ndarray):
                enemble_index = int(np.where(z == 1)[1])
            elif isinstance(z, torch.Tensor):
                enemble_index = int(torch.where(z == 1)[1])
            out = out[enemble_index]
        else:
            out = out.transpose(0, 1)
        return out.squeeze(-1)
