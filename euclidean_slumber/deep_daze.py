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
import torchvision.transforms as T

from imageio import imread, mimsave
from PIL import Image
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import trange, tqdm

from .clip import load, tokenize
from .resample import resample
from .siren import SirenNetwork, LayerActivation, SirenWrapper
from .utils import *

clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]

#adapted from VQGAN notebooks - kornia does not like playing with FP16 so I have to use torchvision instead
augs = nn.Sequential(
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAdjustSharpness(0.3,p=0.4),
    ActuallyRandomAffine(degrees=30, translate=(0.1, 0.1), p=0.8),
    T.RandomPerspective(0.7,p=0.7),
    ActuallyRandomColorJitter(saturation=0.01, contrast=0.01, p=0.7) #adjusting hue causes nan losses for some reason
)
# Helpers

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


def norm_siren_output(img, norm_type):
    assert norm_type in ["none", "clamp", "unmap"], "Invalid normalization type"
    if norm_type == "none":
        return img
    elif norm_type == "clamp":
        return ((img + 1) * 0.5).clamp(0.0, 1.0)
    else:
        return unmap_pixels(img)


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


class DeepDaze(nn.Module):
    def __init__(
            self,
            clip_perceptor,
            clip_norm,
            input_res,
            total_batches,
            batch_size,
            num_layers=8,
            image_width=512,
            image_height=512,
            loss_coef=100,
            theta_initial=None,
            theta_hidden=None,
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            saturate_bound=False,
            gauss_sampling=False,
            gauss_mean=0.6,
            gauss_std=0.2,
            do_cutout=True,
            num_cutouts=16,
            hidden_size=256,
            averaging_weight=0.3,
            experimental_resample=None,
            resample_padding="circular",
            layer_activation=None,
            final_activation=nn.Identity(),
            num_linears=1,
            multiply=None,
            norm_type="unmap",
            fourier=False,
            pooling=False,
            erf_init=False,
            loss_calc="cos_sim",
            augment=False,
            learnable_w0=False
    ):
        super().__init__()
        # load clip
        self.perceptor = clip_perceptor
        self.input_resolution = input_res
        self.normalize_image = clip_norm
        
        self.loss_coef = loss_coef
        self.image_width = image_width

        self.batch_size = batch_size
        self.total_batches = total_batches
        self.num_batches_processed = 0

        self.layer_activation = layer_activation
        self.final_activation = final_activation
        self.num_linears = num_linears
        self.norm_type = norm_type

        w0 = default(theta_hidden, 30.)
        w0_initial = default(theta_initial, 30.)

        siren = SirenNetwork(
            dim_in=2,
            dim_hidden=hidden_size,
            num_layers=num_layers,
            dim_out=3,
            use_bias=True,
            w0=w0,
            w0_initial=w0_initial,
            layer_activation=layer_activation,
            final_activation=final_activation,
            num_linears=num_linears,
            multiply=multiply,
            fourier=fourier,
            erf_init=erf_init,
            learnable=learnable_w0
        )

        self.model = SirenWrapper(
            siren,
            image_width=image_width,
            image_height=image_height
        )

        self.saturate_bound = saturate_bound
        self.saturate_limit = 0.75  # cutouts above this value lead to destabilization
        self.lower_bound_cutout = lower_bound_cutout
        self.upper_bound_cutout = upper_bound_cutout

        self.gauss_sampling = gauss_sampling
        self.gauss_mean = gauss_mean
        self.gauss_std = gauss_std

        self.do_cutout = do_cutout
        self.cut_size = clip_perceptor.visual.input_resolution
        self.num_cutouts = num_cutouts

        self.averaging_weight = averaging_weight
        self.experimental_resample = experimental_resample
        self.resample_padding = resample_padding

        self.av_pool= nn.AdaptiveAvgPool2d((input_res, input_res))
        self.max_pool = nn.AdaptiveMaxPool2d((input_res, input_res))
        self.pooling = pooling
        self.loss_calc = loss_calc
        self.augment = augment

        
    def sample_sizes(self, lower, upper, width, gauss_mean):
        if self.gauss_sampling:
            gauss_samples = torch.zeros(self.batch_size).normal_(mean=gauss_mean, std=self.gauss_std)
            outside_bounds_mask = (gauss_samples > upper) | (gauss_samples < upper)
            gauss_samples[outside_bounds_mask] = torch.zeros((len(gauss_samples[outside_bounds_mask]),)).uniform_(lower, upper)
            sizes = (gauss_samples * width).int()
        else:
            lower *= width
            upper *= width
            sizes = torch.randint(int(lower), int(upper), (self.batch_size,))
        return sizes

    def sdl(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

    def forward(self, text_embed, return_loss=True, dry_run=False):
        out = self.model()
        out = norm_siren_output(out, norm_type=self.norm_type)

        if not return_loss:
            return out
                
        # determine upper and lower sampling bound
        height, width = out.shape[2:4]
        lower_bound = self.lower_bound_cutout
        if self.saturate_bound:
            progress_fraction = self.num_batches_processed / self.total_batches
            lower_bound += (self.saturate_limit - self.lower_bound_cutout) * progress_fraction

        # sample cutout sizes between lower and upper bound
        sizes = self.sample_sizes(lower_bound, self.upper_bound_cutout, width, self.gauss_mean)

        image_pieces = []
        # create normalized random cutouts
        if self.do_cutout:
            max_size = min(height, width)
            min_size = min(height, width, self.cut_size)
            min_size_width = min(height, width)

            lower_bound = float(self.cut_size / min_size_width)
            for cutout in range(self.num_cutouts):
                size = int(min_size_width*torch.zeros(1,).normal_(mean=.8, std=.3).clip(lower_bound, 1.))
                offsetx = torch.randint(0, width - size + 1, ())
                offsety = torch.randint(0, height - size + 1, ())
                image_piece = out[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if self.pooling:
                    image_piece = self.av_pool(image_piece)

                #Implement experimental resampling.
                if exists(self.experimental_resample):
                    image_piece = resample(image_piece, (self.input_resolution, self.input_resolution), self.experimental_resample, align_corners=False, mode='bilinear', padding_mode=self.resample_padding)
                else:
                    image_piece = interpolate(image_piece, self.input_resolution)

                image_pieces.append(image_piece)
        else:
            image_pieces = [interpolate(out.clone(), self.input_resolution) for _ in sizes]

        # normalize
        image_pieces = torch.cat([self.normalize_image(piece) for piece in image_pieces])
        if self.augment:
            image_pieces = augs(image_pieces)
        
        # calc image embedding
        with autocast(enabled=False):
            image_embed = self.perceptor.encode_image(image_pieces)
            
        # calc loss
        # loss over averaged features of cutouts
        avg_image_embed = image_embed.mean(dim=0).unsqueeze(0)
        if self.loss_calc == "cos_sim":
            averaged_loss = -self.loss_coef * torch.cosine_similarity(text_embed, avg_image_embed, dim=-1).mean()
            general_loss = -self.loss_coef * torch.cosine_similarity(text_embed, image_embed, dim=-1).mean()
        elif self.loss_calc == "sdl":
            averaged_loss = self.sdl(avg_image_embed, text_embed).mean()
            general_loss = self.sdl(image_embed, text_embed).mean()
        # merge losses
        loss = averaged_loss * (self.averaging_weight) + general_loss * (1 - self.averaging_weight)

        # count batches
        if not dry_run:
            self.num_batches_processed += self.batch_size
        
        return out, loss


class Imagine(nn.Module):
    def __init__(
            self,
            *,
            #Basic parameters
            text=None,
            img=None,
            clip_encoding=None,
            image_width=512,
            image_height=512,
            gradient_accumulate_every=4,
            save_every=100,
            seed=None,
            save_progress=True,
            open_folder=True,
            save_date_time=False,
            model_name="ViT-B/32",
            num_layers=16,
            epochs=20,
            iterations=1050,

            #SIREN hyperparameters
            theta_initial=None,
            theta_hidden=None,
            hidden_size=256,
            layer_activation=None,
            final_activation="identity",
            num_linears=1,
            multiply=None,
            fourier=False,
            pooling=False,
            erf_init=False,
            loss_calc="cos_sim",
            learnable_w0=False,
            
            #Deepdaze hyperparameters
            lower_bound_cutout=0.1, # should be smaller than 0.8
            upper_bound_cutout=1.0,
            batch_size=4,
            saturate_bound=False,
            averaging_weight=0.3,
            gauss_sampling=False,
            gauss_mean=0.6,
            gauss_std=0.2,
            do_cutout=True,
            num_cutouts=16,
            experimental_resample=None,
            resample_padding="circular",
            norm_type="unmap",
            augment=False,

            #Imagine hyperparameters
            start_image_path=None,
            start_image_train_iters=10,
            start_image_lr=3e-4,
            create_story=False,
            story_start_words=5,
            story_words_per_epoch=5,
            story_separator=None,
            optimizer=opt.AdamP,
            jit=True,
            save_gif=False,
            save_video=False,
            save_best=True,
            lr=1e-5,

            clip_activation=nn.ReLU(inplace=True),
            rotary=False,
            freq_type="lang"
    ):

        super().__init__()

        if exists(seed):
            tqdm.write(f'setting seed: {seed}')
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            
        # fields for story creation:
        self.create_story = create_story
        self.words = None

        self.separator = enable(exists(story_separator), str(story_separator))
        if exists(self.separator) and exists(text):
            #exit if text is just the separator
            if str(text).replace(' ','').replace(self.separator,'') == '':
                print('Exiting because the text only consists of the separator! Needs words or phrases that are separated by the separator.')
                exit()
            #adds a space to each separator and removes double spaces that might be generated
            text = text.replace(self.separator,self.separator+' ').replace('  ',' ').strip()
        self.all_words = enable(exists(text), text.split(" "))
        self.num_start_words = story_start_words
        self.words_per_epoch = story_words_per_epoch
        if create_story:
            assert exists(text),  "We need text input to create a story..."
            # overwrite epochs to match story length
            num_words = len(self.all_words)
            self.epochs = 1 + (num_words - self.num_start_words) / self.words_per_epoch
            # add one epoch if not divisible
            self.epochs = int(self.epochs) if int(self.epochs) == self.epochs else int(self.epochs) + 1
            if exists(self.separator):
                if self.separator not in text:
                    print("Separator '"+self.separator+"' will be ignored since not in text!")
                    self.separator = None
                else:
                    self.epochs = len(list(filter(None,text.split(self.separator))))
            print("Running for", self.epochs, "epochs" + (" (split with '"+self.separator+"' as the separator)" if self.separator is not None else ""))
        else: 
            self.epochs = epochs

        # jit models only compatible with version 1.7.1
        if "1.7.1" not in torch.__version__:
            if jit == True:
                print("Setting jit to False because torch version is not 1.7.1.")
            jit = False

        # Load CLIP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_perceptor, norm = load(model_name, jit=jit, device=self.device, clip_activation=clip_activation, rotary=rotary, freq_type=freq_type)
        self.perceptor = clip_perceptor.eval()
        for param in self.perceptor.parameters():
            param.requires_grad = False
        if jit == False:
            input_res = clip_perceptor.visual.input_resolution
        else:
            input_res = clip_perceptor.input_resolution.item()
        self.clip_transform = create_clip_img_transform(input_res)
        
        self.iterations = iterations
        self.image_width = image_width
        total_batches = self.epochs * self.iterations * batch_size * gradient_accumulate_every
        model = DeepDaze(
                self.perceptor,
                norm,
                input_res,
                total_batches,
                batch_size=batch_size,
                image_width=image_width,
                image_height=image_height,
                num_layers=num_layers,
                theta_initial=theta_initial,
                theta_hidden=theta_hidden,
                lower_bound_cutout=lower_bound_cutout,
                upper_bound_cutout=upper_bound_cutout,
                saturate_bound=saturate_bound,
                gauss_sampling=gauss_sampling,
                gauss_mean=gauss_mean,
                gauss_std=gauss_std,
                do_cutout=do_cutout,
                hidden_size=hidden_size,
                averaging_weight=averaging_weight,
                experimental_resample=experimental_resample,
                resample_padding=resample_padding,
                layer_activation=layer_activation,
                final_activation=final_activation,
                num_linears=num_linears,
                multiply=multiply,
                norm_type=norm_type,
                fourier=fourier,
                num_cutouts=num_cutouts,
                pooling=pooling,
                erf_init=erf_init,
                loss_calc=loss_calc,
                augment=augment,
                learnable_w0=learnable_w0
            ).to(self.device)
        self.model = model
        self.scaler = GradScaler()
        siren_params = model.model.parameters()

        self.optimizer = optimizer(siren_params, lr)

        self.gradient_accumulate_every = gradient_accumulate_every
        self.save_every = save_every
        self.save_date_time = save_date_time
        self.open_folder = open_folder
        self.save_progress = save_progress
        self.text = text
        self.image = img
        self.textpath = create_text_path(self.perceptor.context_length, text=text, img=img, encoding=clip_encoding, separator=story_separator)
        self.filename = self.image_output_path()
        self.save_best = save_best
        self.best_loss = 0
        
        # create coding to optimize for
        self.clip_encoding = self.create_clip_encoding(text=text, img=img, encoding=clip_encoding)

        self.start_image = None
        self.start_image_train_iters = start_image_train_iters
        self.start_image_lr = start_image_lr
        if exists(start_image_path):
            file = Path(start_image_path)
            assert file.exists(), f'file does not exist at given starting image path {start_image_path}'
            image = Image.open(str(file))
            start_img_transform = T.Compose([T.Resize(image_width),
                                             T.CenterCrop((image_width, image_width)),
                                             T.ToTensor()])
            image_tensor = start_img_transform(image).unsqueeze(0).to(self.device)
            self.start_image = image_tensor

        self.save_gif = save_gif
        self.save_video = save_video
            
    def create_clip_encoding(self, text=None, img=None, encoding=None):
        self.text = text
        self.img = img
        if exists(encoding):
            encoding = encoding.to(self.device)
        elif self.create_story:
            encoding = self.update_story_encoding(epoch=0, iteration=1)
        elif exists(text) and exists(img):
            encoding = (self.create_text_encoding(text) + self.create_img_encoding(img)) / 2
        elif exists(text):
            encoding = self.create_text_encoding(text)
        elif exists(img):
            encoding = self.create_img_encoding(img)
        return encoding

    def create_text_encoding(self, text):
        tokenized_text = tokenize(text).to(self.device)
        with torch.no_grad():
            text_encoding = self.perceptor.encode_text(tokenized_text).detach()
        return text_encoding
    
    def create_img_encoding(self, img):
        if isinstance(img, str):
            img = Image.open(img)
        normed_img = self.clip_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_encoding = self.perceptor.encode_image(normed_img).detach()
        return img_encoding
    
    def set_clip_encoding(self, text=None, img=None, encoding=None):
        encoding = self.create_clip_encoding(text=text, img=img, encoding=encoding)
        self.clip_encoding = encoding.to(self.device)
    
    def index_of_first_separator(self) -> int:
        for c, word in enumerate(self.all_words):
            if self.separator in str(word):
                return c +1

    def update_story_encoding(self, epoch, iteration):
        if exists(self.separator):
            self.words = " ".join(self.all_words[:self.index_of_first_separator()])
            #removes separator from epoch-text
            self.words = self.words.replace(self.separator,'')
            self.all_words = self.all_words[self.index_of_first_separator():]
        else:
            if not exists(self.words):
                self.words = " ".join(self.all_words[:self.num_start_words])
                self.all_words = self.all_words[self.num_start_words:]
            else:
                # add words_per_epoch new words
                count = 0
                while count < self.words_per_epoch and len(self.all_words) > 0:
                    new_word = self.all_words[0]
                    self.words = " ".join(self.words.split(" ") + [new_word])
                    self.all_words = self.all_words[1:]
                    count += 1
                # remove words until it fits in context length
                while len(self.words) > self.perceptor.context_length:
                    # remove first word
                    self.words = " ".join(self.words.split(" ")[1:])
        # get new encoding
        print("Now thinking of: ", '"', self.words, '"')
        sequence_number = self.get_img_sequence_number(epoch, iteration)
        # save new words to disc
        with open("story_transitions.txt", "a") as f:
            f.write(f"{epoch}, {sequence_number}, {self.words}\n")
        
        encoding = self.create_text_encoding(self.words)
        return encoding

    def image_output_path(self, sequence_number=None):
        """
        Returns underscore separated Path.
        A current timestamp is prepended if `self.save_date_time` is set.
        Sequence number left padded with 6 zeroes is appended if `save_every` is set.
        :rtype: Path
        """
        output_path = self.textpath
        if sequence_number:
            sequence_number_left_padded = str(sequence_number).zfill(6)
            output_path = f"{output_path}.{sequence_number_left_padded}"
        if self.save_date_time:
            current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
            output_path = f"{current_time}_{output_path}"
        return Path(f"{output_path}.jpg")

    def train_step(self, epoch, iteration):
        total_loss = 0

        for _ in range(self.gradient_accumulate_every):
            with autocast(enabled=True):
                out, loss = self.model(self.clip_encoding)
            loss = loss / self.gradient_accumulate_every
            total_loss += loss
            self.scaler.scale(loss).backward() 
        out = out.cpu().float().clamp(0., 1.)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if iteration % self.save_every == 0:
          self.save_image(epoch, iteration, img=out, progress=self.save_progress)
          if self.save_best and total_loss < self.best_loss:
            self.best_loss = total_loss
            self.save_image(epoch, iteration, img=out, best=True)

        return out, total_loss
    
    def get_img_sequence_number(self, epoch, iteration):
        current_total_iterations = epoch * self.iterations + iteration
        sequence_number = current_total_iterations // self.save_every
        return sequence_number

    @torch.no_grad()
    def save_image(self, epoch, iteration, img=None, progress=False, best=False):
        sequence_number = enable(progress, self.get_img_sequence_number(epoch, iteration))

        if img is None:
            img = self.model(self.clip_encoding, return_loss=False).cpu().float().clamp(0., 1.)
        self.filename = self.image_output_path(sequence_number=sequence_number)
        
        pil_img = T.ToPILImage()(img.squeeze())
        pil_img.save(self.filename, quality=95, subsampling=0)
        pil_img.save(f"{self.textpath}.jpg", quality=95, subsampling=0)
        if best:
            pil_img.save(f"{self.textpath}_best.jpg", quality=95, subsampling=0)

        tqdm.write(f'image updated at "./{str(self.filename)}"')

    def generate_gif(self):
        images = []
        for file_name in sorted(os.listdir('./')):
            if file_name.startswith(self.textpath) and file_name != f'{self.textpath}.jpg':
                images.append(imread(os.path.join('./', file_name)))

        if self.save_video:
            mimsave(f'{self.textpath}.mp4', images)
            print(f'Generated image generation animation at ./{self.textpath}.mp4')
        if self.save_gif:
            mimsave(f'{self.textpath}.gif', images)
            print(f'Generated image generation animation at ./{self.textpath}.gif')

    def forward(self):
        if exists(self.start_image):
            tqdm.write('Preparing with initial image...')
            optim = DiffGrad(self.model.model.parameters(), lr = self.start_image_lr)
            pbar = trange(self.start_image_train_iters, desc='iteration')
            try:
                for _ in pbar:
                    loss = self.model.model(self.start_image)
                    loss.backward()
                    pbar.set_description(f'loss: {loss.item():.2f}')

                    optim.step()
                    optim.zero_grad()
            except KeyboardInterrupt:
                print('interrupted by keyboard, gracefully exiting')
                return exit()

            del self.start_image
            del optim

        tqdm.write(f'Imagining "{self.textpath}" from the depths of my weights...')

        with torch.no_grad():
            self.model(self.clip_encoding, dry_run=True) # do one warmup step due to potential issue with CLIP and CUDA

        if self.open_folder:
            open_folder('./')
            self.open_folder = False

        try:
            for epoch in trange(self.epochs, desc='epochs'):
                pbar = trange(self.iterations, desc='iteration')
                for i in pbar:
                    _, loss = self.train_step(epoch, i)
                    pbar.set_description(f'loss: {loss.item():.2f}')

                # Update clip_encoding per epoch if we are creating a story
                if self.create_story:
                    self.clip_encoding = self.update_story_encoding(epoch, i)
        except KeyboardInterrupt:
            print('interrupted by keyboard, gracefully exiting')
            return

        self.save_image(epoch, i) # one final save at end

        if (self.save_gif or self.save_video) and self.save_progress:
            self.generate_gif()