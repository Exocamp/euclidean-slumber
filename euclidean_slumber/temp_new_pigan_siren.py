import math

import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn

#For now most SIREN modules here will be reimplemented from scratch, to clean up clutter.
from .utils import exists, enable

#this code's an abomination, a fusion of original repo's pi-GAN code, lucidrain's pi-GAN code and my own code

#todo: replace this later with one of the new biological activation functions?
def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

#custom layer activation
class LayerActivation(nn.Module):
    """The activation for any SIREN layer.
    This can be customized, though it defaults to torch.sin"""
    def __init__(self, w0=1., activation=torch.sin):
        super().__init__()
        self.w0 = w0
        self.activation = activation

    def forward(self, x):
        return self.activation(self.w0 * x)

#SIREN base layer.
class SIRENLayer(nn.Module):
    """The SIREN layer. Despite the name, this class actually can be a standard linear layer by setting weight_dist to 'normal'/'gaussian', w0 to 1, and layer_act = nn.Identity()
    This way I don't need two separate linear layer classes to handle both SIREN and the mapping network. Plus it's easier to modify down the line."""
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False, use_bias=True, layer_act=torch.sin, final_act=None, weight_dist='uniform', lr_mul=1.0):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.lr_mul = lr_mul
        assert weight_dist in ['normal', 'gaussian', 'uniform'], 'invalid initialization scheme'

        weight = torch.zeros(dim_out, dim_in)
        bias = enable(use_bias, torch.zeros(dim_out))
        self.init_(weight, bias, w0 = w0, c = c, dist=weight_dist)

        self.weight = nn.Parameter(weight)
        self.bias = enable(use_bias, nn.Parameter(bias))
        if final_act is None:
            self.activation = LayerActivation(w0=w0, activation=layer_act)
        else:
            self.activation = final_act

    def init_(self, weight, bias, w0, c, dist):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        if dist == 'uniform':
            weight.uniform_(-w_std, w_std)
        else:
            weight.normal_(0., 1.)

        if exists(bias) and dist == 'uniform':
            bias.uniform_(-w_std, w_std)

    def forward(self, x, freq=None, p_shift=None):
        out = F.linear(x, self.weight * self.lr_mul, self.bias * self.lr_mul)
        #apply FiLM modulation lol
        if exists(freq):
            out = out * freq

        if exists(p_shift):
            out = out + p_shift

        return self.activation(out)

#Mapping network
class CustomMappingNetwork(nn.Module):
    def __init__(self, *, dim_in, dim_out, num_cmn_layers=3, lr_mul = 0.1):
        super().__init__()
        layers = []

        for i in range(num_cmn_layers):
            layers.extend([SIRENLayer(dim_in, dim_out, w0=1., lr_mul=lr_mul, layer_act=nn.Identity(), weight_dist='normal'), leaky_relu()]) #todo: customize activation functiion

            self.net = nn.Sequential(*layers)

            self.to_freqs = nn.Linear(dim_in, dim_out)
            self.to_p_shift = nn.Linear(dim_in, dim_out)

        def forward(self, x):
            x = F.normalize(x, dim=-1)
            x = self.net(x)
            return self.to_freqs(x), self.to_p_shift(x)

#The big momma SIREN network itself

class SIRENNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1., w0_initial=30., use_bias=True, layer_act=torch.sin, final_act=nn.Identity()):
        super().__init__()
        self.layers = nn.ModuleList([])

        #append first layer
        self.layers.append(SIRENLayer(
            dim_in=dim_in,
            dim_out=dim_hidden,
            w0=w0_initial,
            is_first=True,
            use_bias=use_bias,
            layer_act=layer_act
        ))

        #intermediate layers
        for layer in range(num_layers - 1):
            self.layers.append(SIRENLayer(
                dim_in=dim_hidden,
                dim_out=dim_hidden,
                w0=w0,
                use_bias=use_bias,
                layer_act=layer_act
            ))

        #final layer
        self.final_layer = SIRENLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            final_act=final_act
        )

    def forward(self, x, freqs, p_shifts):
        for layer in self.layers:
            x = layer(x, freqs, p_shifts)
        return self.final_layer(x)

#Classic SIREN wrapper adapted for pi-GAN. Generates an image
class SIRENWrapper(nn.Module):
    def __init__(self, net, )