import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .positional_encoding import SphericalHarmonics


def satclip_location_encoder(weights: str):
    weights = torch.load(weights, map_location="cpu")
    model = LocationEncoder(
        num_layers=weights["hyper_parameters"]["num_hidden_layers"],
        dim_hidden=weights["hyper_parameters"]["capacity"],
        dim_out=weights["hyper_parameters"]["embed_dim"],
        legendre_polys=weights["hyper_parameters"]["legendre_polys"],
    ).to("cpu")
    state_dict = {
        k.replace("model.location.", ""): v
        for k, v in weights["state_dict"].items()
        if "location" in k
    }
    model.load_state_dict(state_dict, strict=True)
    return model


class LocationEncoder(nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        num_layers: int,
        dim_out: int,
        legendre_polys: int = 10,
    ):
        super().__init__()
        self.posenc = SphericalHarmonics(legendre_polys=legendre_polys)
        self.nnet = SirenNet(
            dim_in=self.posenc.embedding_dim,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            dim_out=dim_out,
        )

    def forward(self, x):
        x = self.posenc(x)
        return self.nnet(x)


class SirenNet(nn.Module):
    """Sinusoidal Representation Network (SIREN)"""

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        degreeinput=False,
        dropout=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.degreeinput = degreeinput

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    dropout=dropout,
                )
            )

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
            dropout=False,
        )

    def forward(self, x, mods=None):
        # do some normalization to bring degrees in a -pi to pi range
        if self.degreeinput:
            x = torch.deg2rad(x) - torch.pi

        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        out = self.activation(out)
        return out


def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)
