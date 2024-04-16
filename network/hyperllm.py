from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import loralib

from .trajectory_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2Config
from .trajectory_gpt2_LoRA import GPT2Model_LoRA, GPT2LMHeadModel_LoRA, GPT2Config_LoRA

from .hypernet import HyperLayer
from .ensemble import mlp


class LinearLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        out_bias: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.basedmodel = nn.Linear(in_features, out_features, out_bias)
        if prior_scale > 0:
            self.priormodel = nn.Linear(in_features, out_features, out_bias)
            for param in self.priormodel.parameters():
                param.requires_grad = False
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.basedmodel.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.xavier_normal_(param, gain=1.0)
        if self.prior_scale > 0:
            for name, param in self.priormodel.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.xavier_normal_(param, gain=1.0)

    def forward(self, x):
        out = self.basedmodel(x)
        if self.prior_scale > 0:
            prior_out = self.priormodel(x)
            out = self.posterior_scale * out + self.prior_scale * prior_out
        return out


class HyperLinear(nn.Module):
    def __init__(
        self,
        noise_dim,
        out_features,
        action_dim: int = 2,
        prior_std: float = 1.0,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        use_bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        hyperlayer_params = dict(
            noise_dim=noise_dim,
            hidden_dim=out_features,
            action_dim=action_dim,
            prior_std=prior_std,
            out_type="weight",
            use_bias=use_bias,
            device=device,
        )
        self.hyper_weight = HyperLayer(
            **hyperlayer_params, trainable=True, weight_init="xavier_normal"
        )
        self.prior_weight = HyperLayer(
            **hyperlayer_params, trainable=False, weight_init="sDB"
        )

        self.hidden_dim = out_features
        self.action_dim = action_dim
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, z, x, prior_x):
        theta = self.hyper_weight(z)
        prior_theta = self.prior_weight(z)
        if len(theta.shape) == 2:
            theta = theta.view(-1, self.action_dim, self.hidden_dim)
            prior_theta = prior_theta.view(-1, self.action_dim, self.hidden_dim)
        elif len(theta.shape) == 3:
            theta = theta.view(theta.shape[0], -1, self.action_dim, self.hidden_dim)
            prior_theta = prior_theta.view(
                prior_theta.shape[0], -1, self.action_dim, self.hidden_dim
            )

        if len(x.shape) > 2:
            # compute feel-good term
            out = torch.einsum("bd,bad -> ba", theta, x)
            prior_out = torch.einsum("bd,bad -> ba", prior_theta, prior_x)
        elif x.shape[0] != z.shape[0]:
            # compute action value for one action set
            theta = theta.squeeze(0)
            prior_theta = prior_theta.squeeze(0)
            out = torch.mm(theta, x.T).squeeze(0)
            prior_out = torch.mm(prior_theta, prior_x.T).squeeze(0)
        elif x.shape == theta.shape:
            out = torch.sum(x * theta, -1)
            prior_out = torch.sum(prior_x * prior_theta, -1)
        elif len(theta.shape) == 4:
            out = torch.einsum("bnad,bd -> bna", theta, x)
            prior_out = torch.einsum("bnad,bd -> bna", prior_theta, prior_x)
        else:
            # compute predict reward in batch
            out = torch.bmm(theta, x.unsqueeze(-1)).squeeze(-1)
            prior_out = torch.bmm(prior_theta, prior_x.unsqueeze(-1)).squeeze(-1)

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


class HyperLLM(nn.Module):
    def __init__(
        self,
        noise_dim: int = 2,
        action_num: int = 2,
        prior_scale: float = 1.0,
        posterior_scale: float = 1.0,
        out_bias: bool = True,
        head_name: str = "hyper",
        llm_name: str = "gpt2",
        use_lora: bool = False,
        fine_tune: bool = False,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        base_path = "/apdcephfs/share_1563664/ztjiaweixu/huggingface"
        if use_lora:
            name_or_path = f"{base_path}/pretrain_{llm_name}_lora"
            config = GPT2Config_LoRA.from_pretrained(name_or_path)
            self.transformer_model = GPT2LMHeadModel_LoRA.from_pretrained(
                name_or_path, config=config
            )
            loralib.mark_only_lora_as_trainable(
                self.transformer_model, bias="lora_only"
            )
        else:
            name_or_path = f"{base_path}/pretrain_{llm_name}"
            config = GPT2Config.from_pretrained(name_or_path)
            self.transformer_model = GPT2LMHeadModel.from_pretrained(
                name_or_path,
                config=config,
            )
        if not fine_tune:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        feature_dim = config.n_embd
        if head_name == "linear":
            self.out = LinearLayer(
                feature_dim,
                action_num,
                prior_scale=prior_scale,
                posterior_scale=posterior_scale,
                out_bias=out_bias,
                device=device,
            )
        elif head_name == "hyper":
            self.out = HyperLinear(
                noise_dim,
                feature_dim,
                action_dim=action_num,
                prior_scale=prior_scale,
                posterior_scale=posterior_scale,
                use_bias=out_bias,
                device=device,
            )
        elif head_name == "ensemble":
            self.out = nn.ModuleList(
                [
                    mlp(feature_dim, action_num, noise_dim, out_bias)
                    for _ in range(noise_dim)
                ]
            )
        else:
            raise ValueError(f"Unknown head_name: {head_name}")

        self.fine_tune = fine_tune
        self.head_name = head_name
        self.device = device

    def forward(self, noise, input_ids, attention_mask):
        if isinstance(noise, np.ndarray):
            noise = torch.as_tensor(noise, device=self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        transformer_out = self.transformer_model.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = transformer_out.last_hidden_state[:, -1, :]
        if not self.fine_tune:
            logits = logits.detach()
        if self.head_name == "linear":
            out = self.out(logits)
        else:
            out = self.out(noise, logits, logits)
        return out


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperLLM(device=device).to(device)
    print(model)
