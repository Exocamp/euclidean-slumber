## Implementing pi-GAN SIRENs based off the offical repo's source code here. I'm just curious. ##

import math
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

class LayerActivation(nn.Module):
    def __init__(self, torch_activation=torch.sin, w0 = 1., learnable=False):
        super().__init__()
        self.w0 = nn.Parameter(torch.ones(1) * w0) if learnable else w0
        self.activation = torch_activation
    def forward(self, x):
        return self.activation(self.w0 * x)

class UniformBoxWarp(nn.Module):
	def forward(self, coords, side_length):
		return coords * (2 / side_length)

class FilmLayer(nn.Module):
	def __init__(self, dim_in, dim_out, is_first = False, use_bias=True, c = 6., w0 = 1., activation=torch.sin):
		super().__init__()
		self.dim_in = dim_in
		self.dim_out = dim_out
		self.is_first = is_first
		self.use_bias = use_bias

		weight = torch.zeros(dim_out, dim_in)
		bias = enable(use_bias, torch.zeros(dim_out))
		self.init_(weight, bias, c = c, w0 = w0)

		self.weight = nn.Parameter(weight)
		self.bias = enable(use_bias, nn.Parameter(bias))
		self.activation = activation

	def init_(self, weight, bias, c, w0):
		w_std = (1 / self.dim_in) if self.is_first else (math.sqrt(c / self.dim_in) / w0)
		weight.uniform_(-w_std, w_std)

		if exists(bias):
			bias.uniform_(-w_std, w_std)

	def forward(self, x, freq, phase_shift):
		x = F.linear(x, self.weight, self.bias)
		freq = freq.unsqueeze(1).expand_as(x)
		phase_shift = phase_shift.unsqueeze(1).expand_as(x)

		return self.activation(freq * x + phase_shift)

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
			self.network[-1].weight *= 0.25
		#lol. lmao
		del blocks

	def forward(self, z):
		freq_offsets = self.network(z)
		freqs = freq_offsets[..., :freq_offsets.shape[-1] // 2]
		phase_shifts = freq_offsets[..., freq_offsets.shape[-1] // 2]

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

		assert num_layers > 5, 'Not enough layers in TALLSIREN (5 or more needed)'
		
		#Layer 1
		self.network.append(FilmLayer(dim_in, dim_hidden, is_first = True, w0=w0, activation=layer_activation))
		#Intermediate layers
		for _ in range(num_film_layers - 3):
			self.network.append(FilmLayer(dim_hidden, dim_hidden, w0=w0, activation=layer_activation))
		self.final_layer = nn.Linear(dim_hidden, dim_out)

		self.color_layer_sine = FilmLayer(dim_hidden + 3, dim_hidden, activation=layer_activation)
		self.color_layer_linear = FilmLayer(dim_hidden, 3, w0=w0, activation=final_activation)

		self.mapping_network = CustomMappingNetwork(z_dim=z_dim, map_hidden_dim=dim_hidden, map_output_dim=((len(self.network) + 1) * dim_hidden * 2), num_layers=num_map_layers)

		self.final_layer.apply(frequency_init(25))

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
		rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), freqs[..., -self.dim_hidden:], phase_shifts[..., -self.dim_hidden:])
		rbg = self.color_layer_linear(rbg)

		return torch.cat([rbg, sigma], dim=-1)

class SPATIALSIRENBASELINE(nn.Module):
	"""Description from pi-GAN repo: Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1. Why not use it here?
	NOTE: num_layers is the TOTAL number of layers within the whole SIREN, including the color layers.
	TALLSIREN from original repo has 8 B&W layers and 2 color layers making ten in total - not including the layers in the mapping network."""

	def __init__(self, num_layers, dim_in=3, dim_hidden=256, dim_out=1, z_dim=100, w0=25., layer_activation=torch.sin, final_activation=nn.Sigmoid()):
		super().__init__()
		self.dim_in = dim_in
		self.dim_hidden = dim_hidden
		self.dim_out = dim_out
		self.z_dim = z_dim
		self.network = nn.ModuleList([])

		assert num_layers > 5, 'Not enough layers in TALLSIREN (5 or more needed)'
		
		#Layer 1
		self.network.append(FilmLayer(3, dim_hidden, is_first = True, w0=w0, activation=layer_activation))
		#Intermediate layers
		for _ in range(num_layers - 3):
			self.network.append(FilmLayer(dim_hidden, dim_hidden, w0=w0, activation=layer_activation))
		self.final_layer = nn.Linear(dim_hidden, dim_out)

		self.color_layer_sine = FilmLayer(dim_hidden + 3, dim_hidden, activation=layer_activation)
		self.color_layer_linear = nn.Linear(dim_hidden, 3)

		self.mapping_network = CustomMappingNetwork(z_dim=z_dim, map_hidden_dim=dim_hidden, map_output_dim=((len(self.network) + 1) * dim_hidden * 2))

		self.final_layer.apply(frequency_init(25))
		self.color_layer_linear.apply(frequency_init(25))

		self.gridwarper = UniformBoxWarp(0.24)

	def forward(self, input, z, ray_directions, **kwargs):
		freqs, phase_shifts = self.mapping_network(z)
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
		rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), freqs[..., -self.dim_hidden:], phase_shifts[..., -self.dim_hidden:])
		rbg = self.color_layer_linear(rbg)

		return torch.cat([rbg, sigma], dim=-1)

#near-copy of SirenWrapper, but useful to keep it by itself
class SirenWrapper(nn.Module):
    def __init__(self, net, image_width, image_height, latent_dim = None):
        super().__init__()
        assert isinstance(net, SirenNetwork), 'SirenWrapper must receive a Siren network'

        self.net = net
        self.image_width = image_width
        self.image_height = image_height

        self.modulator = None
        if exists(latent_dim):
            self.modulator = Modulator(
                dim_in = latent_dim,
                dim_hidden = net.dim_hidden,
                num_layers = net.num_layers
            )

        tensors = [torch.linspace(-1, 1, steps = image_width), torch.linspace(-1, 1, steps = image_height)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')

        self.register_buffer('grid', mgrid)

    def forward(self, img = None, *, latent = None):
        modulate = exists(self.modulator)
        assert not (modulate ^ exists(latent)), 'latent vector must be only supplied if `latent_dim` was passed in on instantiation'

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords)
        out = rearrange(out, '(h w) c -> () c h w', h = self.image_height, w = self.image_width)

        if exists(img):
            return F.mse_loss(img, out)

        return out