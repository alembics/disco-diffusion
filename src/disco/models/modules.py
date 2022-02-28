from dataclasses import dataclass
import math

import torch
from torch import nn

def append_dims(x, n):
    """
    Append `n` `None` values to the end of the `x` array's dimensions
    
    :param x: The tensor to be reshaped
    :param n: The number of dimensions to add
    :return: a tensor with the same shape as x, but with additional dimensions of size n inserted at the
    front.
    """
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    """
    Given a tensor x, expand it to a tensor of shape (1, 1, ..., 1, *shape[2:])
    
    :param x: The input tensor
    :param shape: the shape of the tensor to be expanded
    :return: The input tensor x is being expanded to the shape of the output tensor.
    """
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    """
    Given an alpha and sigma, return the corresponding t
    
    :param alpha: the rotation angle in radians
    :param sigma: The standard deviation of the Gaussian kernel
    :return: the angle in radians.
    """
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    """
    Given a tensor of angles, return a tuple of two tensors of the same shape, one containing the cosine
    of the angles and the other containing the sine of the angles
    
    :param t: the time parameter
    :return: the cosine and sine of the input t multiplied by pi/2.
    """
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    """
    3x3 conv + ReLU
    """
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
