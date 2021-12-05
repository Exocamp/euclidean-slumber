## Implementing pi-GAN SIRENs based off the offical repo's source code here. I'm just curious. ##

import math

import numpy as np
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn

#For now most SIREN modules here will be reimplemented from scratch, to clean up clutter.
from .utils import exists, enable

#taken from pi-GAN source code
def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init

def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input) 

class LayerActivation(nn.Module):
    def __init__(self, torch_activation=torch.sin, w0 = 1., learnable=False):
        super().__init__()
        self.w0 = nn.Parameter(torch.ones(1) * w0) if learnable else w0
        self.activation = torch_activation
    def forward(self, x):
        return self.activation(self.w0 * x)

class UniformBoxWarp(nn.Module):
	def __init__(self, sidelength):
		super().__init__()
		self.scale_fac = 2 / sidelength

	def forward(self, coords):
		return coords * self.scale_fac

class FilmLayer(nn.Module):
	def __init__(self, dim_in, dim_hidden):
		super().__init__()
		self.layer = nn.Linear(dim_in, dim_hidden)

	def forward(self, x, freq, phase_shift):
		x = self.layer(x)
		freq = freq.unsqueeze(1).expand_as(x)
		phase_shift = phase_shift.unsqueeze(1).expand_as(x)
		return torch.sin(freq * x + phase_shift)

class CMNBlock(nn.Module):
	def __init__(self, dim_in, dim_out, do_activation=True, block_activation=nn.LeakyReLU(0.2, inplace=True)):
		super().__init__()
		if do_activation:
			self.block = nn.Sequential(nn.Linear(dim_in, dim_out), block_activation)
		else:
			self.block = nn.Linear(dim_in, dim_out)

	def forward(self, x):
		return self.block(x)

## CustomMappingNetwork from pi-GAN source code adapted to here ##
class CustomMappingNetwork(nn.Module):
	def __init__(self, z_dim, map_hidden_dim, map_output_dim, num_layers=4):
		super().__init__()

		assert num_layers > 2, 'Not enough layers in mapping network (3 or more needed)'
		blocks = []
		#First 'block'
		blocks.append(CMNBlock(z_dim, map_hidden_dim))
		#Rest of the blocks
		for _ in range(num_layers - 2):
			blocks.append(CMNBlock(map_hidden_dim, map_hidden_dim))
		#Last block
		blocks.append(CMNBlock(map_hidden_dim, map_output_dim, do_activation=False))

		self.network = nn.Sequential(*blocks)
		self.network.apply(kaiming_leaky_init)
		with torch.no_grad():
			self.network[-1].block.weight *= 0.25
		#lol. lmao
		del blocks

	def forward(self, z):
		freq_offsets = self.network(z)

		freqs = freq_offsets[..., :freq_offsets.shape[-1] // 2]
		phase_shifts = freq_offsets[..., freq_offsets.shape[-1] // 2:]

		return freqs, phase_shifts

class TALLSIREN(nn.Module):
	"""Description from pi-GAN repo: Primary SIREN architecture used in pi-GAN generators. Why not use it here?
	NOTE: num_layers is the TOTAL number of layers within the whole SIREN, including the color layers.
	TALLSIREN from original repo has 8 B&W layers and 2 color layers making ten in total - not including the layers in the mapping network."""

	def __init__(self, num_film_layers=10, num_map_layers=4, dim_in=2, dim_hidden=256, dim_out=1, z_dim=100, w0=25., layer_activation=torch.sin, final_activation=nn.Sigmoid()):
		super().__init__()
		self.dim_in = dim_in
		self.dim_hidden = dim_hidden
		self.dim_out = dim_out
		self.z_dim = z_dim
		self.network = nn.ModuleList([])
		
		#Layer 1
		self.network.append(FilmLayer(dim_in, dim_hidden))
		#Intermediate layers
		for _ in range(num_film_layers - 3):
			self.network.append(FilmLayer(dim_hidden, dim_hidden))
		self.final_layer = nn.Linear(dim_hidden, 1)

		self.color_layer_sine = FilmLayer(dim_hidden + 3, dim_hidden)
		self.color_layer_linear = nn.Sequential(nn.Linear(dim_hidden, 3), final_activation)

		self.mapping_network = CustomMappingNetwork(z_dim=z_dim, map_hidden_dim=dim_hidden, map_output_dim=((len(self.network) + 1) * dim_hidden * 2), num_layers=num_map_layers)

		self.network.apply(frequency_init(25))
		self.final_layer.apply(frequency_init(25))
		self.color_layer_sine.apply(frequency_init(25))
		self.color_layer_linear.apply(frequency_init(25))
		self.network[0].apply(first_layer_film_sine_init)

	def forward(self, input, z, ray_directions, **kwargs):
		freqs, phase_shifts = self.mapping_network(z)
		return self.forward_with_freqs_phase_shifts(input, freqs, phase_shifts, ray_directions, **kwargs)

	def forward_with_freqs_phase_shifts(self, input, freqs, phase_shifts, ray_directions, **kwargs):
		freqs = freqs * 15 + 30
		x = input

		for i, layer in enumerate(self.network):
			start = i * self.dim_hidden
			end = (i + 1) * self.dim_hidden
			x = layer(x, freqs[..., start:end], phase_shifts[..., start:end])

		sigma = self.final_layer(x)
		rgb = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), freqs[..., -self.dim_hidden:], phase_shifts[..., -self.dim_hidden:])
		rgb = self.color_layer_linear(rgb)

		return torch.cat([rgb, sigma], dim=-1)

class SPATIALSIRENBASELINE(nn.Module):
	"""Description from pi-GAN repo: Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1. Why not use it here?
	NOTE: num_layers is the TOTAL number of layers within the whole SIREN, including the color layers.
	SPATIALSIRENBASELINE from original repo has 8 B&W layers and 2 color layers making ten in total - not including the layers in the mapping network."""

	def __init__(self, num_film_layers=10, num_map_layers=4, dim_in=2, dim_hidden=256, dim_out=1, z_dim=100, w0=25., layer_activation=torch.sin, final_activation=nn.Sigmoid()):
		super().__init__()
		self.dim_in = dim_in
		self.dim_hidden = dim_hidden
		self.dim_out = dim_out
		self.z_dim = z_dim
		self.network = nn.ModuleList([])
		
		#Layer 1
		self.network.append(FilmLayer(3, dim_hidden))
		#Intermediate layers
		for _ in range(num_film_layers - 3):
			self.network.append(FilmLayer(dim_hidden, dim_hidden))
		self.final_layer = nn.Linear(dim_hidden, 1)

		self.color_layer_sine = FilmLayer(dim_hidden + 3, dim_hidden)
		self.color_layer_linear = nn.Sequential(nn.Linear(dim_hidden, 3))

		self.mapping_network = CustomMappingNetwork(z_dim=z_dim, map_hidden_dim=dim_hidden, map_output_dim=((len(self.network) + 1) * dim_hidden * 2), num_layers=num_map_layers)

		self.network.apply(frequency_init(25))
		self.final_layer.apply(frequency_init(25))
		self.color_layer_sine.apply(frequency_init(25))
		self.color_layer_linear.apply(frequency_init(25))
		self.network[0].apply(first_layer_film_sine_init)

		self.gridwarper = UniformBoxWarp(0.24)

	def forward(self, input, z, ray_directions, **kwargs):
		freqs, phase_shifts = self.mapping_network(z)
		#input shape: torch.Size([1, 196608, 3]), z shape: torch.Size([1, 256]), ray_directions shape: torch.Size([1, 196608, 3]), freqs shape: torch.Size([1, 2304]), phase_shifts shape: torch.Size([1, 2304])
		return self.forward_with_freqs_phase_shifts(input, freqs, phase_shifts, ray_directions, **kwargs)

	def forward_with_freqs_phase_shifts(self, input, freqs, phase_shifts, ray_directions, **kwargs):
		freqs = freqs * 15 + 30
		input = self.gridwarper(input)
		x = input

		for i, layer in enumerate(self.network):
			start = i * self.dim_hidden
			end = (i + 1) * self.dim_hidden
			x = layer(x, freqs[..., start:end], phase_shifts[..., start:end])

		sigma = self.final_layer(x)
		rgb = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), freqs[..., -self.dim_hidden:], phase_shifts[..., -self.dim_hidden:])
		rgb = torch.sigmoid(self.color_layer_linear(rgb))
		#print(f"rgb shape: {rgb.shape}, sigma shape: {sigma.shape}")
		#rgb shape: torch.Size([1, 46608, 3]), sigma shape: torch.Size([1, 46608, 1])
		#rgb shape: torch.Size([1, 50000, 3]), sigma shape: torch.Size([1, 50000, 1])
		#rgb shape: torch.Size([1, 196608, 3]), sigma shape: torch.Size([1, 196608, 1])

		return torch.cat([rgb, sigma], dim=-1)