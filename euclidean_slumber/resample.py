#Code ported over from others, not mine. Credits to crowsonkb and alstroemeria313 for code

"""Good differentiable image resampling for PyTorch."""

from functools import update_wrapper

import torch
from torch.nn import functional as F

from .utils import lanczos, ramp


def odd(fn):
    return update_wrapper(lambda x: torch.sign(x) * fn(abs(x)), fn)


def _to_linear_srgb(input):
    cond = input <= 0.04045
    a = input / 12.92
    b = ((input + 0.055) / 1.055)**2.4
    return torch.where(cond, a, b)


def _to_nonlinear_srgb(input):
    cond = input <= 0.0031308
    a = 12.92 * input
    b = 1.055 * input**(1/2.4) - 0.055
    return torch.where(cond, a, b)


to_linear_srgb = odd(_to_linear_srgb)
to_nonlinear_srgb = odd(_to_nonlinear_srgb)


def resample(input, size, num, align_corners=True, is_srgb=False, mode='bicubic', padding_mode='constant'):
    #bigsleep uses a num of 3, vqgan uses a num of 2
    assert padding_mode in ['constant', 'reflect', 'replicate', 'circular'], "Invalid padding mode for resample"

    n, c, h, w = input.shape
    dh, dw = size

    if is_srgb:
        input = to_linear_srgb(input)

    viewshape = [n * c, 1, h, w]

    input = input.view(viewshape)

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, num), num).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), padding_mode)
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, num), num).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), padding_mode)
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    input = F.interpolate(input, size, mode=mode, align_corners=align_corners)

    if is_srgb:
        input = to_nonlinear_srgb(input)

    return input