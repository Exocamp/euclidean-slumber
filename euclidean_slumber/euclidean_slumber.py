## EUCLIDEAN SLUMBER: A pi-GAN plus CLIP experience based off of Deep Daze and pi-GAN source code.

#imports
import math
import os
import subprocess
import sys
import random

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_optimizer as opt
import torchvision
import torchvision.transforms as T

from imageio import imread, mimsave
from PIL import Image
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch_ema import ExponentialMovingAverage
from tqdm import trange, tqdm

from .clip import load, tokenize
from .pigan_generator import ImplicitGenerator3d
from .pigan_sirens import TALLSIREN, SPATIALSIRENBASELINE
from .resample import resample
from .utils import *

#utils
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def interpolate(image, size):
	return F.interpolate(image, (size, size), mode='bilinear', align_corners=False)

def create_clip_img_transform(image_width):
	transform = T.Compose([
		#T.ToPILImage(),
		T.Resize(image_width),
		T.CenterCrop((image_width, image_width)),
		T.ToTensor(),
		T.Normalize(mean=clip_mean, std=clip_std)
	])
	return transform

def open_folder(path):
	if os.path.isfile(path):
		path = os.path.dirname(path)

	if not os.path.isdir(path):
		return

	cmd_list = None
	if sys.platform == 'darwin':
		cmd_list = ['open', '--', path]
	elif sys.platform == 'linux2' or sys.platform == 'linux':
		cmd_list = ['xdg-open', path]
	elif sys.platform in ['win32', 'win64']:
		cmd_list = ['explorer', path.replace('/', '\\')]
	if cmd_list is None:
		return

	try:
		subprocess.check_call(cmd_list)
	except subprocess.CalledProcessError:
		pass
	except OSError:
		pass

def create_text_path(context_length, text=None, img=None, encoding=None, separator=None):
	if exists(text):
		if exists(separator) and separator in text:
			#Reduces filename to first epoch text
			text = text[:text.index(separator, )]
		input_name = text.replace(" ", "_")[:context_length]
	elif exists(img):
		if isinstance(img, str):
			input_name = "".join(img.replace(" ", "_").split(".")[:-1])
		else:
			input_name = "PIL_img"
	else:
		input_name = "your_encoding"
	return input_name


def sample_z(shape, device, dist_type):
	if dist_type == 'gaussian':
		z = torch.randn(shape, device=device)
	elif dist_type == 'uniform':
		z = torch.rand(shape, device=device) * 2 - 1
	return z

class EuclideanSlumber(nn.Module):
	def __init__(
		self,
		clip_perceptor,
		clip_norm,
		image_size,
		input_res,
		num_cutouts,
		hidden_size=256,
		num_film_layers=10,
		num_map_layers=4,
		num_steps=12,
		theta_initial=25.,
		theta_hidden=25.,
		latent_dim=256,
		fov=12,
		ray_start=0.8,
		ray_end=1.2,
		fade_steps=10000,
		h_mean=0.5*math.pi,
		h_stddev=0.5,
		v_mean=0.5*math.pi,
		v_stddev=0.4,
		z_dist='gaussian',
		sample_dist='uniform',
		hierarchical_sample=True,
		lock_view_dependence=False,
		siren='TALLSIREN',
		clamp_mode='softplus',
		averaging_weight=0.3
	):

		super().__init__()

		#Load CLIP
		self.clip_perceptor = clip_perceptor
		self.input_resolution = input_res
		self.normalize_image = clip_norm

		self.image_size = image_size

		self.num_cutouts = num_cutouts
		self.num_batches_processed = 0
		self.averaging_weight = averaging_weight

		w0 = default(theta_hidden, 25.)
		w0_initial = default(theta_initial, 25.)

		assert siren in ['TALLSIREN', 'SPATIALSIRENBASELINE'], 'Invalid SIREN'
		if siren == "TALLSIREN":
			siren = TALLSIREN(
				num_film_layers=num_film_layers,
				num_map_layers=num_map_layers,
				dim_in=3,
				dim_hidden=hidden_size,
				dim_out=4,
				z_dim=latent_dim,
				w0=theta_hidden,
				final_activation=nn.Sigmoid()
			)
		elif siren == "SPATIALSIRENBASELINE":
			siren = SPATIALSIRENBASELINE(
				num_film_layers=num_film_layers,
				num_map_layers=num_map_layers,
				dim_in=3,
				dim_hidden=hidden_size,
				dim_out=4,
				z_dim=latent_dim,
				w0=theta_hidden,
				final_activation=nn.Sigmoid()
			)

		self.generator = ImplicitGenerator3d(siren=siren, z_dim=latent_dim, clamp_mode=clamp_mode)

		#Generate fixed z:
		self.latent_dim = latent_dim
		self.z_dist = z_dist
		self.fixed_z = sample_z((1, latent_dim), device=device, dist_type=z_dist)

		#Generator hyperparams
		self.fov = fov
		self.ray_start = ray_start
		self.ray_end = ray_end
		self.num_steps = num_steps
		self.h_mean = h_mean
		self.h_stddev = h_stddev
		self.v_mean = v_mean
		self.v_stddev = v_stddev
		self.hierarchical_sample = hierarchical_sample
		self.sample_dist = sample_dist
		self.lock_view_dependence = lock_view_dependence
		self.counter = 0 #step counter for nerf noise

	def forward(self, text_embed, return_loss=True, dry_run=False):
		self.generator.train()
		#Generate z
		z = sample_z((1, self.latent_dim), device=device, dist_type=self.z_dist)
		self.nerf_noise = max(0, 1. - self.counter/5000.)
		#Generate image from z.
		#Yes I know this is inefficient. We can make this nicer a little later when I'm less lazy
		img, _ = self.generator(z, self.image_size, self.fov, self.ray_start, self.ray_end, self.num_steps, self.h_stddev, self.v_stddev, self.h_mean, self.v_mean, self.hierarchical_sample, self.nerf_noise, self.sample_dist, self.lock_view_dependence)
		#normalize image
		img = ((img + 1) * 0.5).clamp(0.0, 1.0)

		if not return_loss:
			return img

		#cutouts
		image_pieces = []
		height, width = img.shape[2:4]
		max_size = min(height, width)
		min_size = min(height, width, self.input_resolution)
		min_size_width = min(height, width)
		lower_bound = float(self.input_resolution / min_size_width)

		for cutout in range(self.num_cutouts):
			size = int(min_size_width*torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound, 1.))
			offsetx = torch.randint(0, width - size + 1, ())
			offsety = torch.randint(0, height - size + 1, ())
			image_piece = img[:, :, offsety:offsety + size, offsetx:offsetx + size]
			#Re-add experimental resampling later
			image_piece = interpolate(image_piece, self.input_resolution)

			image_pieces.append(image_piece)

		#normalize
		image_pieces = torch.cat([self.normalize_image(piece) for piece in image_pieces])
		#calc image embedding
		with autocast(enabled=False):
			image_embed = self.clip_perceptor.encode_image(image_pieces)

		# calc loss
		# loss over averaged features of cutouts
		avg_image_embed = image_embed.mean(dim=0).unsqueeze(0)
		averaged_loss = -100 * torch.cosine_similarity(text_embed, avg_image_embed, dim=-1).mean()
		general_loss = -100 * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
		# merge losses
		loss = averaged_loss * (self.averaging_weight) + general_loss * (1 - self.averaging_weight)

		# count batches
		#if not dry_run:
			#self.num_batches_processed += self.batch_size
		self.counter += 1

		return img, loss

class ESWrapper(nn.Module):
	def __init__(
		self,
		text,
		image_size,
		num_cutouts,
		hidden_size=256,
		num_film_layers=10,
		num_map_layers=4,
		num_steps=12,
		theta_initial=25.,
		theta_hidden=25.,
		latent_dim=256,
		fov=12,
		ray_start=0.8,
		ray_end=1.2,
		fade_steps=10000,
		h_mean=0.5*math.pi,
		h_stddev=0.5,
		v_mean=0.5*math.pi,
		v_stddev=0.4,
		z_dist='uniform',
		sample_dist='uniform',
		hierarchical_sample=True,
		lock_view_dependence=False,
		siren='TALLSIREN',
		clamp_mode='softplus',
		averaging_weight=0.3,
		epochs=20,
		iterations=1050,
		model_name="ViT-B/32",
		gradient_accum_every = 1,
		unique_lr=False,
		gen_lr=6e-5,
		weight_decay=0,
		grad_clip=0.3,
		betas=(0, 0.9),
		save_every=20,
		seed=0
	):

		super().__init__()
		tqdm.write(f'Using device {device}')

		if exists(seed):
			tqdm.write(f'Setting seed to {seed}')
			torch.manual_seed(seed)
			torch.cuda.manual_seed(seed)
			random.seed(seed)
			torch.backends.cudnn.deterministic = True

		self.image_size = image_size
		self.iterations = iterations
		self.epochs = epochs

		#Load CLIP
		clip_perceptor, norm = load(model_name, jit=False, device=device)
		self.perceptor = clip_perceptor.eval()
		for param in self.perceptor.parameters():
			param.requires_grad = False
		input_res = clip_perceptor.visual.input_resolution
		self.clip_transform = create_clip_img_transform(input_res)

		model = EuclideanSlumber(
			self.perceptor,
			norm,
			image_size,
			input_res,
			num_cutouts,
			hidden_size=hidden_size,
			num_film_layers=num_film_layers,
			num_map_layers=num_map_layers,
			num_steps=num_steps,
			theta_initial=theta_initial,
			theta_hidden=theta_hidden,
			latent_dim=latent_dim,
			fov=fov,
			ray_start=ray_start,
			ray_end=ray_end,
			fade_steps=fade_steps,
			h_mean=h_mean,
			h_stddev=h_stddev,
			v_mean=v_mean,
			v_stddev=v_stddev,
			z_dist=z_dist,
			sample_dist=sample_dist,
			hierarchical_sample=hierarchical_sample,
			lock_view_dependence=lock_view_dependence,
			siren=siren,
			clamp_mode=clamp_mode,
			averaging_weight=averaging_weight
		).to(device)

		self.model = model
		self.ema = ExponentialMovingAverage(model.generator.parameters(), decay=0.999)
		self.ema2 = ExponentialMovingAverage(model.generator.parameters(), decay=0.9999)

		self.scaler = GradScaler()
		#Modify LRs
		if not unique_lr:
			mapping_network_param_names = [name for name, _ in model.generator.siren.mapping_network.named_parameters()]
			mapping_network_params = [p for n, p in model.generator.named_parameters() if n in mapping_network_param_names]
			gen_parameters = [p for n, p in model.generator.named_parameters() if n not in mapping_network_param_names]
			self.optimizer = optim.Adam([{'params': gen_parameters, 'name': 'generator'},
				{'params': mapping_network_params, 'name': 'mapping_network', 'lr': gen_lr*5e-2}],
				lr=gen_lr, betas=betas, weight_decay=weight_decay)
		else:
			self.optimizer = optim.Adam(model.generator.parameters(), lr=gen_lr, betas=betas)

		self.model.generator.set_device(device)

		self.gradient_accum_every = gradient_accum_every
		self.grad_clip = grad_clip
		self.save_every = save_every
		self.text = text
		self.textpath = create_text_path(self.perceptor.context_length, text=text)
		# create coding to optimize for
		self.clip_encoding = self.create_clip_encoding(text)

	def create_clip_encoding(self, text=None):
		self.text = text
		#will implement rest later
		tokenized_text = tokenize(text).to(device)
		with torch.no_grad():
			text_encoding = self.perceptor.encode_text(tokenized_text).to(device)
		return text_encoding

	def image_output_path(self):
		output_path = self.textpath
		#will implement rest later
		return Path(f"{output_path}.jpg")

	def train_step(self, epoch, iteration):
		total_loss = 0

		if self.scaler.get_scale() < 1:
			self.scaler.update(1.)

		for _ in range(self.gradient_accum_every):
			with autocast(enabled=True):
				out, loss = self.model(self.clip_encoding)
				loss = loss / self.gradient_accum_every
				total_loss += loss
				self.scaler.scale(loss).backward()

		out = out.cpu().float().clamp(0., 1.)
		self.scaler.unscale_(self.optimizer)
		nn.utils.clip_grad_norm_(self.model.generator.parameters(), self.grad_clip)
		self.scaler.step(self.optimizer)
		self.scaler.update()
		self.optimizer.zero_grad()
		self.ema.update(self.model.generator.parameters())
		self.ema2.update(self.model.generator.parameters())

		if iteration % self.save_every == 0:
			self.save_image(epoch, iteration)

		return out, total_loss

	def save_image(self, epoch, iteration, img=None):
		#if not exists(img):
		#	img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
		self.filename = self.image_output_path()

		self.model.generator.eval()
		with torch.no_grad():
			with autocast(enabled=True):
				#fixed: h and v stddev = 0
				gen_image = self.model.generator.staged_forward(self.model.fixed_z, (self.image_size * 2), self.model.fov, self.model.ray_start, self.model.ray_end, self.model.num_steps, 0, 0, self.model.h_mean, self.model.v_mean, hierarchical_sample=self.model.hierarchical_sample, nerf_noise=self.model.nerf_noise, sample_dist=self.model.sample_dist, lock_view_dependence=self.model.lock_view_dependence)[0]
			torchvision.utils.save_image(gen_image, f"{self.textpath}_fixed.jpg", normalize=True)
			with autocast(enabled=True):
				#tilted: h and v stddev = 0, h_mean += 0.5
				gen_image = self.model.generator.staged_forward(self.model.fixed_z, (self.image_size * 2), self.model.fov, self.model.ray_start, self.model.ray_end, self.model.num_steps, 0, 0, (self.model.h_mean + 0.5), self.model.v_mean, hierarchical_sample=self.model.hierarchical_sample, nerf_noise=self.model.nerf_noise, sample_dist=self.model.sample_dist, lock_view_dependence=self.model.lock_view_dependence)[0]
			torchvision.utils.save_image(gen_image, f"{self.textpath}_tilted.jpg", normalize=True)

		self.ema.store(self.model.generator.parameters())
		self.ema.copy_to(self.model.generator.parameters())
		self.model.generator.eval()

		with torch.no_grad():
			with autocast(enabled=True):
				#ema fixed
				gen_image = self.model.generator.staged_forward(self.model.fixed_z, (self.image_size * 2), self.model.fov, self.model.ray_start, self.model.ray_end, self.model.num_steps, 0, 0, self.model.h_mean, self.model.v_mean, hierarchical_sample=self.model.hierarchical_sample, nerf_noise=self.model.nerf_noise, sample_dist=self.model.sample_dist, lock_view_dependence=self.model.lock_view_dependence)[0]
			torchvision.utils.save_image(gen_image, f"{self.textpath}_fixed_ema.jpg", normalize=True)
			with autocast(enabled=True):
				#ema tilted
				gen_image = self.model.generator.staged_forward(self.model.fixed_z, (self.image_size * 2), self.model.fov, self.model.ray_start, self.model.ray_end, self.model.num_steps, 0, 0, (self.model.h_mean + 0.5), self.model.v_mean, hierarchical_sample=self.model.hierarchical_sample, nerf_noise=self.model.nerf_noise, sample_dist=self.model.sample_dist, lock_view_dependence=self.model.lock_view_dependence)[0]
			torchvision.utils.save_image(gen_image, f"{self.textpath}_tilted_ema.jpg", normalize=True)
			with autocast(enabled=True):
				gen_image = self.model.generator.staged_forward(self.model.fixed_z, (self.image_size * 2), self.model.fov, self.model.ray_start, self.model.ray_end, self.model.num_steps, 0, 0, self.model.h_mean, self.model.v_mean, hierarchical_sample=self.model.hierarchical_sample, nerf_noise=self.model.nerf_noise, sample_dist=self.model.sample_dist, lock_view_dependence=self.model.lock_view_dependence, psi=0.7)[0]
			torchvision.utils.save_image(gen_image, f"{self.textpath}_random.jpg", normalize=True)

		self.ema.restore(self.model.generator.parameters())
		self.model.generator.train()

		#pil_img = T.ToPILImage()(img.squeeze())
		#pil_img.save(self.filename, quality=95, subsampling=0)
		#pil_img.save(f"{self.textpath}.jpg", quality=95, subsampling=0)

		tqdm.write(f'images updated, check file directory')

	def forward(self):
		tqdm.write(f'Prepare for magic with your prompt "{self.textpath}"')

		with torch.no_grad():
			self.model(self.clip_encoding, dry_run = True)

		#TODO: implement the rest later lmao
		return
