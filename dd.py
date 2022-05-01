import os
import random
import gc
from IPython import display
from ipywidgets import Output
import lpips
import pathlib, shutil
import json
import subprocess
import requests
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass
from functools import partial
from torch import nn
from torch.nn import functional as F
import math
import cv2
from PIL import Image, ImageOps
from datetime import datetime
import climage
from types import SimpleNamespace
from glob import glob
from pydotted import pydot

from tqdm.notebook import tqdm
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from resize_right import resize
import disco_xform_utils as dxf
import pytorch3dlite.pytorch3dlite as p3d

from clip import clip
from ipywidgets import Output
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2json(v):
    try:
        j = json.loads(v)
        return j
    except:
        raise argparse.ArgumentTypeError(f"⚠️ Could not parse CLI parameter.  Check your quotation marks and special characters. ⚠️ Value:\n{v}")


def get_param(key, fallback=None):
    if os.getenv(key, None) != None:
        try:
            return json.loads(os.getenv(key))
        except:
            print(f'⚠️ Could not parse environment parameter "{key}".  Check your quotation marks and special characters. ⚠️')
            return fallback
    return fallback


def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)


def gitclone(url):
    res = subprocess.run(["git", "clone", url], stdout=subprocess.PIPE).stdout.decode("utf-8")
    print(res)


def pipi(modulestr):
    res = subprocess.run(["pip", "install", modulestr], stdout=subprocess.PIPE).stdout.decode("utf-8")
    print(res)


def pipie(modulestr):
    res = subprocess.run(["git", "install", "-e", modulestr], stdout=subprocess.PIPE).stdout.decode("utf-8")
    print(res)


# def wget(url, outputdir):
#  res = subprocess.run(['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE).stdout.decode('utf-8')
#  print(res)

# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869


def interp(t):
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=None):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(
    octaves=[1, 1, 1, 1],
    width=2,
    height=2,
    grayscale=True,
    device=None,
    side_x=None,
    side_y=None,
):
    out = perlin_ms(octaves, width, height, grayscale, device)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert("RGB")
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out


def regen_perlin(perlin_mode=None, device=None, batch_size=None):
    if perlin_mode == "color":
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, device=device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, False, device=device)
    elif perlin_mode == "gray":
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, True, device=device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, device=device)
    else:
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, device=device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, device=device)

    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size, -1, -1, -1)


def fetch(url_or_path):
    if str(url_or_path).startswith("http://") or str(url_or_path).startswith("https://"):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def read_image_workaround(path):
    """OpenCV reads images as BGR, Pillow saves them as RGB. Work around
    this incompatibility to avoid colour inversions."""
    im_tmp = cv2.imread(path)
    return cv2.cvtColor(im_tmp, cv2.COLOR_BGR2RGB)


def parse_prompt(prompt):
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(
                    max_size
                    * torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(float(self.cut_size / max_size), 1.0)
                )
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


class MakeCutoutsDango(nn.Module):
    def __init__(
        self,
        cut_size,
        Overview=4,
        InnerCrop=0,
        IC_Size_Pow=0.5,
        IC_Grey_P=0.2,
        args=None,
    ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.cutout_debug = args.cutout_debug
        self.debug_folder = f"{args.batch_folder}/debug"
        if args.animation_mode == "None":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(
                        degrees=10,
                        translate=(0.05, 0.05),
                        interpolation=T.InterpolationMode.BILINEAR,
                    ),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.1),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )
        elif args.animation_mode == "Video Input":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomPerspective(distortion_scale=0.4, p=0.7),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.15),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )
        elif args.animation_mode == "2D" or args.animation_mode == "3D":
            self.augs = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.4),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomAffine(
                        degrees=10,
                        translate=(0.05, 0.05),
                        interpolation=T.InterpolationMode.BILINEAR,
                    ),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.RandomGrayscale(p=0.1),
                    T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
                ]
            )

    def forward(self, input, skip_augs=None):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        output_shape_2 = [1, 3, self.cut_size + 2, self.cut_size + 2]
        padargs = {}
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
            **padargs,
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if self.cutout_debug:
                TF.to_pil_image(cutouts[0].clamp(0, 1).squeeze(0)).save(f"{self.debug_folder}/cutout_overview0.jpg", quality=99)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if self.cutout_debug:
                TF.to_pil_image(cutouts[-1].clamp(0, 1).squeeze(0)).save(f"{self.debug_folder}/cutout_InnerCrop.jpg", quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def do_3d_step(
    img_filepath,
    frame_num,
    midas_model,
    midas_transform,
    translations=None,
    device=None,
    TRANSLATION_SCALE=None,
    key_frames=True,
    args=None,
):
    if key_frames:
        translation_x = translations.translation_x_series[frame_num]
        translation_y = translations.translation_y_series[frame_num]
        translation_z = translations.translation_z_series[frame_num]
        rotation_3d_x = translations.rotation_3d_x_series[frame_num]
        rotation_3d_y = translations.rotation_3d_y_series[frame_num]
        rotation_3d_z = translations.rotation_3d_z_series[frame_num]
        print(
            f"translation_x: {translation_x}",
            f"translation_y: {translation_y}",
            f"translation_z: {translation_z}",
            f"rotation_3d_x: {rotation_3d_x}",
            f"rotation_3d_y: {rotation_3d_y}",
            f"rotation_3d_z: {rotation_3d_z}",
        )

    translate_xyz = [
        -translation_x * TRANSLATION_SCALE,
        translation_y * TRANSLATION_SCALE,
        -translation_z * TRANSLATION_SCALE,
    ]
    rotate_xyz_degrees = [rotation_3d_x, rotation_3d_y, rotation_3d_z]
    print("translation:", translate_xyz)
    print("rotation:", rotate_xyz_degrees)
    rotate_xyz = [
        math.radians(rotate_xyz_degrees[0]),
        math.radians(rotate_xyz_degrees[1]),
        math.radians(rotate_xyz_degrees[2]),
    ]
    rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    print("rot_mat: " + str(rot_mat))
    next_step_pil = dxf.transform_image_3d(
        img_filepath,
        midas_model,
        midas_transform,
        device,
        rot_mat,
        translate_xyz,
        args.near_plane,
        args.far_plane,
        args.fov,
        padding_mode=args.padding_mode,
        sampling_mode=args.sampling_mode,
        midas_weight=args.midas_weight,
    )
    return next_step_pil


def save_settings(setting_list=None, batchFolder=None, batch_name=None, batchNum=None):
    with open(f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+") as f:
        json.dump(pydot(setting_list), f, ensure_ascii=False, indent=4)


def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
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
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock(
                [
                    nn.AvgPool2d(2),
                    ConvBlock(c, c * 2),
                    ConvBlock(c * 2, c * 2),
                    SkipBlock(
                        [
                            nn.AvgPool2d(2),
                            ConvBlock(c * 2, c * 4),
                            ConvBlock(c * 4, c * 4),
                            SkipBlock(
                                [
                                    nn.AvgPool2d(2),
                                    ConvBlock(c * 4, c * 8),
                                    ConvBlock(c * 8, c * 4),
                                    nn.Upsample(
                                        scale_factor=2,
                                        mode="bilinear",
                                        align_corners=False,
                                    ),
                                ]
                            ),
                            ConvBlock(c * 8, c * 4),
                            ConvBlock(c * 4, c * 2),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        ]
                    ),
                    ConvBlock(c * 4, c * 2),
                    ConvBlock(c * 2, c),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ]
            ),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock(
                [
                    self.down,
                    ConvBlock(cs[0], cs[1]),
                    ConvBlock(cs[1], cs[1]),
                    SkipBlock(
                        [
                            self.down,
                            ConvBlock(cs[1], cs[2]),
                            ConvBlock(cs[2], cs[2]),
                            SkipBlock(
                                [
                                    self.down,
                                    ConvBlock(cs[2], cs[3]),
                                    ConvBlock(cs[3], cs[3]),
                                    SkipBlock(
                                        [
                                            self.down,
                                            ConvBlock(cs[3], cs[4]),
                                            ConvBlock(cs[4], cs[4]),
                                            SkipBlock(
                                                [
                                                    self.down,
                                                    ConvBlock(cs[4], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[4]),
                                                    self.up,
                                                ]
                                            ),
                                            ConvBlock(cs[4] * 2, cs[4]),
                                            ConvBlock(cs[4], cs[3]),
                                            self.up,
                                        ]
                                    ),
                                    ConvBlock(cs[3] * 2, cs[3]),
                                    ConvBlock(cs[3], cs[2]),
                                    self.up,
                                ]
                            ),
                            ConvBlock(cs[2] * 2, cs[2]),
                            ConvBlock(cs[2], cs[1]),
                            self.up,
                        ]
                    ),
                    ConvBlock(cs[1] * 2, cs[1]),
                    ConvBlock(cs[1], cs[0]),
                    self.up,
                ]
            ),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


# Initialize MiDaS depth model.
# It remains resident in VRAM and likely takes around 2GB VRAM.
# You could instead initialize it for each frame (and free it after each frame) to save VRAM.. but initializing it is slow.
def init_midas_depth_model(midas_model_type="dpt_large", optimize=True, model_path=None, device=None):
    DEVICE = device
    default_models = {
        "midas_v21_small": f"{model_path}/midas_v21_small-70d6b9c8.pt",
        "midas_v21": f"{model_path}/midas_v21-f6b98070.pt",
        "dpt_large": f"{model_path}/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": f"{model_path}/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_nyu": f"{model_path}/dpt_hybrid_nyu-2ce69ec7.pt",
    }
    midas_model = None
    net_w = None
    net_h = None
    resize_mode = None
    normalization = None

    print(f"Initializing MiDaS '{midas_model_type}' depth model...")
    # load network
    midas_model_path = default_models[midas_model_type]

    if midas_model_type == "dpt_large":  # DPT-Large
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "dpt_hybrid":  # DPT-Hybrid
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "dpt_hybrid_nyu":  # DPT-Hybrid-NYU
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "midas_v21":
        midas_model = MidasNet(midas_model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif midas_model_type == "midas_v21_small":
        midas_model = MidasNet_small(
            midas_model_path,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={"expand": True},
        )
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        print(f"midas_model_type '{midas_model_type}' not implemented")
        assert False

    midas_transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    midas_model.eval()

    if optimize == True:
        if DEVICE == torch.device("cuda"):
            midas_model = midas_model.to(memory_format=torch.channels_last)
            midas_model = midas_model.half()

    midas_model.to(DEVICE)

    print(f"MiDaS '{midas_model_type}' depth model initialized.")
    return midas_model, midas_transform, net_w, net_h, resize_mode, normalization


def generate_eye_views(
    trans_scale,
    batchFolder,
    filename,
    frame_num,
    midas_model,
    midas_transform,
    vr_eye_angle=None,
    vr_ipd=None,
    device=None,
    args=None,
):
    for i in range(2):
        theta = vr_eye_angle * (math.pi / 180)
        ray_origin = math.cos(theta) * vr_ipd / 2 * (-1.0 if i == 0 else 1.0)
        ray_rotation = theta if i == 0 else -theta
        translate_xyz = [-(ray_origin) * trans_scale, 0, 0]
        rotate_xyz = [0, (ray_rotation), 0]
        rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
        transformed_image = dxf.transform_image_3d(
            f"{batchFolder}/{filename}",
            midas_model,
            midas_transform,
            device,
            rot_mat,
            translate_xyz,
            args.near_plane,
            args.far_plane,
            args.fov,
            padding_mode=args.padding_mode,
            sampling_mode=args.sampling_mode,
            midas_weight=args.midas_weight,
            spherical=True,
        )
        eye_file_path = batchFolder + f"/frame_{frame_num-1:04}" + ("_l" if i == 0 else "_r") + ".png"
        transformed_image.save(eye_file_path)


def parse_key_frames(string, prompt_parser=None):
    """Given a string representing frame numbers paired with parameter values at that frame,
    return a dictionary with the frame numbers as keys and the parameter values as the values.

    Parameters
    ----------
    string: string
        Frame numbers paired with parameter values at that frame number, in the format
        'framenumber1: (parametervalues1), framenumber2: (parametervalues2), ...'
    prompt_parser: function or None, optional
        If provided, prompt_parser will be applied to each string of parameter values.

    Returns
    -------
    dict
        Frame numbers as keys, parameter values at that frame number as values

    Raises
    ------
    RuntimeError
        If the input string does not match the expected format.

    Examples
    --------
    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)")
    {10: 'Apple: 1| Orange: 0', 20: 'Apple: 0| Orange: 1| Peach: 1'}

    >>> parse_key_frames("10:(Apple: 1| Orange: 0), 20: (Apple: 0| Orange: 1| Peach: 1)", prompt_parser=lambda x: x.lower()))
    {10: 'apple: 1| orange: 0', 20: 'apple: 0| orange: 1| peach: 1'}
    """
    import re

    pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
    frames = dict()
    for match_object in re.finditer(pattern, string):
        frame = int(match_object.groupdict()["frame"])
        param = match_object.groupdict()["param"]
        if prompt_parser:
            frames[frame] = prompt_parser(param)
        else:
            frames[frame] = param

    if frames == {} and len(string) != 0:
        raise RuntimeError("Key Frame string not correctly formatted")
    return frames


def get_inbetweens(key_frames, integer=False, max_frames=None, interp_spline=None):
    """Given a dict with frame numbers as keys and a parameter value as values,
    return a pandas Series containing the value of the parameter at every frame from 0 to max_frames.
    Any values not provided in the input dict are calculated by linear interpolation between
    the values of the previous and next provided frames. If there is no previous provided frame, then
    the value is equal to the value of the next provided frame, or if there is no next provided frame,
    then the value is equal to the value of the previous provided frame. If no frames are provided,
    all frame values are NaN.

    Parameters
    ----------
    key_frames: dict
        A dict with integer frame numbers as keys and numerical values of a particular parameter as values.
    integer: Bool, optional
        If True, the values of the output series are converted to integers.
        Otherwise, the values are floats.

    Returns
    -------
    pd.Series
        A Series with length max_frames representing the parameter values for each frame.

    Examples
    --------
    >>> max_frames = 5
    >>> get_inbetweens({1: 5, 3: 6})
    0    5.0
    1    5.0
    2    5.5
    3    6.0
    4    6.0
    dtype: float64

    >>> get_inbetweens({1: 5, 3: 6}, integer=True)
    0    5
    1    5
    2    5
    3    6
    4    6
    dtype: int64
    """
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i, value in key_frames.items():
        key_frame_series[i] = value
    key_frame_series = key_frame_series.astype(float)

    interp_method = interp_spline

    if interp_method == "Cubic" and len(key_frames.items()) <= 3:
        interp_method = "Quadratic"

    if interp_method == "Quadratic" and len(key_frames.items()) <= 2:
        interp_method = "Linear"

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    # key_frame_series = key_frame_series.interpolate(method=intrp_method,order=1, limit_direction='both')
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction="both")
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series


def split_prompts(prompts, max_frames=None):
    prompt_series = pd.Series([np.nan for a in range(max_frames)])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


def move_files(start_num, end_num, old_folder, new_folder, batch_name=None, batchNum=None):
    for i in range(start_num, end_num):
        old_file = old_folder + f"/{batch_name}({batchNum})_{i:04}.png"
        new_file = new_folder + f"/{batch_name}({batchNum})_{i:04}.png"
        os.rename(old_file, new_file)


def processKeyFrameProperties(
    max_frames,
    interp_spline,
    angle,
    zoom,
    translation_x,
    translation_y,
    translation_z,
    rotation_3d_x,
    rotation_3d_y,
    rotation_3d_z,
):
    try:
        angle_series = get_inbetweens(parse_key_frames(angle), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `angle` correctly for key frames.\n"
            "Attempting to interpret `angle` as "
            f'"0: ({angle})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        angle = f"0: ({angle})"
        angle_series = get_inbetweens(parse_key_frames(angle), max_frames=max_frames, interp_spline=interp_spline)

    try:
        zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames=max_frames, interp_spline=interp_spline)
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `zoom` correctly for key frames.\n"
            "Attempting to interpret `zoom` as "
            f'"0: ({zoom})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        zoom = f"0: ({zoom})"
        zoom_series = get_inbetweens(parse_key_frames(zoom), max_frames=max_frames, interp_spline=interp_spline)

    try:
        translation_x_series = get_inbetweens(
            parse_key_frames(translation_x),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_x` correctly for key frames.\n"
            "Attempting to interpret `translation_x` as "
            f'"0: ({translation_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_x = f"0: ({translation_x})"
        translation_x_series = get_inbetweens(
            parse_key_frames(translation_x),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )

    try:
        translation_y_series = get_inbetweens(
            parse_key_frames(translation_y),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_y` correctly for key frames.\n"
            "Attempting to interpret `translation_y` as "
            f'"0: ({translation_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_y = f"0: ({translation_y})"
        translation_y_series = get_inbetweens(
            parse_key_frames(translation_y),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )

    try:
        translation_z_series = get_inbetweens(
            parse_key_frames(translation_z),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `translation_z` correctly for key frames.\n"
            "Attempting to interpret `translation_z` as "
            f'"0: ({translation_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        translation_z = f"0: ({translation_z})"
        translation_z_series = get_inbetweens(
            parse_key_frames(translation_z),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )

    try:
        rotation_3d_x_series = get_inbetweens(
            parse_key_frames(rotation_3d_x),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_x` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_x` as "
            f'"0: ({rotation_3d_x})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_x = f"0: ({rotation_3d_x})"
        rotation_3d_x_series = get_inbetweens(
            parse_key_frames(rotation_3d_x),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )

    try:
        rotation_3d_y_series = get_inbetweens(
            parse_key_frames(rotation_3d_y),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_y` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_y` as "
            f'"0: ({rotation_3d_y})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_y = f"0: ({rotation_3d_y})"
        rotation_3d_y_series = get_inbetweens(
            parse_key_frames(rotation_3d_y),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )
    try:
        rotation_3d_z_series = get_inbetweens(
            parse_key_frames(rotation_3d_z),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )
    except RuntimeError as e:
        print(
            "WARNING: You have selected to use key frames, but you have not "
            "formatted `rotation_3d_z` correctly for key frames.\n"
            "Attempting to interpret `rotation_3d_z` as "
            f'"0: ({rotation_3d_z})"\n'
            "Please read the instructions to find out how to use key frames "
            "correctly.\n"
        )
        rotation_3d_z = f"0: ({rotation_3d_z})"
        rotation_3d_z_series = get_inbetweens(
            parse_key_frames(rotation_3d_z),
            max_frames=max_frames,
            interp_spline=interp_spline,
        )
    return (
        angle,
        zoom,
        translation_x,
        translation_y,
        translation_z,
        rotation_3d_x,
        rotation_3d_y,
        rotation_3d_z,
        angle_series,
        zoom_series,
        translation_x_series,
        translation_y_series,
        translation_z_series,
        rotation_3d_x_series,
        rotation_3d_y_series,
        rotation_3d_z_series,
    )


def do_run(
    args=None,
    device=None,
    is_colab=False,
    batchNum=None,
    start_frame=None,
):
    print(f"💻 Starting Run: {args.batch_name}({batchNum}) at frame {start_frame}")
    print("Prepping models...")
    model_config = model_and_diffusion_defaults()
    # Update Model Settings
    if args.diffusion_model == "512x512_diffusion_uncond_finetune_008100":
        model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,  # No need to edit this, it is taken care of later.
                "rescale_timesteps": True,
                "timestep_respacing": 250,  # No need to edit this, it is taken care of later.
                "image_size": 512,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_checkpoint": args.use_checkpoint,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )
    elif args.diffusion_model == "256x256_diffusion_uncond":
        model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,  # No need to edit this, it is taken care of later.
                "rescale_timesteps": True,
                "timestep_respacing": 250,  # No need to edit this, it is taken care of later.
                "image_size": 256,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_checkpoint": args.use_checkpoint,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )
    timestep_respacing = f"ddim{args.steps}"
    diffusion_steps = (1000 // args.steps) * args.steps if args.steps < 1000 else args.steps
    model_config.update({"timestep_respacing": timestep_respacing, "diffusion_steps": diffusion_steps})
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(f"{args.model_path}/{args.diffusion_model}.pt", map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()

    gc.collect()
    torch.cuda.empty_cache()
    seed = args.seed
    # for param in args:
    #    locals()[param] = args[param]
    clip_models = []

    def clipLoad(model_name):
        print(f"🤖 Loading model '{model_name}'...")
        model = clip.load(model_name, jit=False)[0].eval().requires_grad_(False).to(device)
        clip_models.append(model)

    if args.ViTB32 is True:
        clipLoad("ViT-B/32")
    if args.ViTB16 is True:
        clipLoad("ViT-B/16")
    if args.ViTL14 is True:
        clipLoad("ViT-L/14")
    if args.ViTL14_336 is True:
        clipLoad("ViT-L/14@336px")
    if args.RN50 is True:
        clipLoad("RN50")
    if args.RN50x4 is True:
        clipLoad("RN50x4")
    if args.RN50x16 is True:
        clipLoad("RN50x16")
    if args.RN50x64 is True:
        clipLoad("RN50x64")
    if args.RN101 is True:
        clipLoad("RN101")

    if args.use_secondary_model:
        print("🤖 Loading secondary model...")
        secondary_model = SecondaryDiffusionImageNet2()
        secondary_model.load_state_dict(torch.load(f"{args.model_path}/secondary_model_imagenet_2.pth", map_location="cpu"))
        secondary_model.eval().requires_grad_(False).to(device)

    print(f"🤖 Loading LPIPS...")
    lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device)

    print("🌱 Seed used:", args.seed)

    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    angle = None
    zoom = None
    translation_x = None
    translation_y = None

    if args.key_frames:
        (
            angle,
            zoom,
            translation_x,
            translation_y,
            translation_z,
            rotation_3d_x,
            rotation_3d_y,
            rotation_3d_z,
            angle_series,
            zoom_series,
            translation_x_series,
            translation_y_series,
            translation_z_series,
            rotation_3d_x_series,
            rotation_3d_y_series,
            rotation_3d_z_series,
        ) = processKeyFrameProperties(
            args.max_frames,
            args.interp_spline,
            args.angle,
            args.zoom,
            args.translation_x,
            args.translation_y,
            args.translation_z,
            args.rotation_3d_x,
            args.rotation_3d_y,
            args.rotation_3d_z,
        )

    else:
        angle = float(angle)
        zoom = float(zoom)
        translation_x = float(translation_x)
        translation_y = float(translation_y)
        translation_z = float(translation_z)
        rotation_3d_x = float(rotation_3d_x)
        rotation_3d_y = float(rotation_3d_y)
        rotation_3d_z = float(rotation_3d_z)

    # print(range(args.start_frame, args.max_frames))

    if (args.animation_mode == "3D") and (args.midas_weight > 0.0):
        (
            midas_model,
            midas_transform,
            midas_net_w,
            midas_net_h,
            midas_resize_mode,
            midas_normalization,
        ) = init_midas_depth_model(args.midas_depth_model, model_path=args.model_path, device=device)
    for frame_num in range(args.start_frame, args.max_frames):
        # if stop_on_next_loop:
        #  break
        if is_in_notebook():
            display.clear_output(wait=True)

        # Print Frame progress if animation mode is on
        if args.animation_mode != "None":
            batchBar = tqdm(range(args.max_frames), ncols=40, dynamic_ncols=True, desc="Frames", position=0, leave=True)
            batchBar.n = frame_num
            batchBar.refresh()

        # Inits if not video frames
        if args.animation_mode != "Video Input":
            if args.init_image == "":
                init_image = None
            else:
                init_image = args.init_image
            init_scale = args.init_scale
            skip_steps = args.skip_steps

        if args.animation_mode == "2D":
            if args.key_frames:
                angle = angle_series[frame_num]
                zoom = zoom_series[frame_num]
                translation_x = translation_x_series[frame_num]
                translation_y = translation_y_series[frame_num]
                print(
                    f"angle: {angle}",
                    f"zoom: {zoom}",
                    f"translation_x: {translation_x}",
                    f"translation_y: {translation_y}",
                )

            if frame_num > 0:
                seed += 1
                if args.resume_run and frame_num == args.start_frame:
                    img_0 = cv2.imread(args.batchFolder + f"/{args.batch_name}({args.batchNum})_{args.start_frame-1:04}.png")
                else:
                    img_0 = cv2.imread("prevFrame.png")
                center = (1 * img_0.shape[1] // 2, 1 * img_0.shape[0] // 2)
                trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
                rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
                trans_mat = np.vstack([trans_mat, [0, 0, 1]])
                rot_mat = np.vstack([rot_mat, [0, 0, 1]])
                transformation_matrix = np.matmul(rot_mat, trans_mat)
                img_0 = cv2.warpPerspective(
                    img_0,
                    transformation_matrix,
                    (img_0.shape[1], img_0.shape[0]),
                    borderMode=cv2.BORDER_WRAP,
                )

                cv2.imwrite("prevFrameScaled.png", img_0)
                init_image = "prevFrameScaled.png"
                init_scale = args.frames_scale
                skip_steps = args.calc_frames_skip_steps

        if args.animation_mode == "3D":
            if frame_num > 0:
                seed += 1
                if args.resume_run and frame_num == args.start_frame:
                    img_filepath = args.batchFolder + f"/{args.batch_name}({args.batchNum})_{args.start_frame-1:04}.png"
                    if args.turbo_mode and frame_num > args.turbo_preroll:
                        shutil.copyfile(img_filepath, "oldFrameScaled.png")
                else:
                    img_filepath = "/content/prevFrame.png" if is_colab else "prevFrame.png"

                next_step_pil = do_3d_step(
                    img_filepath,
                    frame_num,
                    midas_model,
                    midas_transform,
                    translations={
                        "angle_series": angle_series,
                        "zoom_series": zoom_series,
                        "translation_x_series": translation_x_series,
                        "translation_y_series": translation_y_series,
                        "translation_z_series": translation_z_series,
                        "rotation_3d_x_series": rotation_3d_x_series,
                        "rotation_3d_y_series": rotation_3d_y_series,
                        "rotation_3d_z_series": rotation_3d_z_series,
                    },
                    device=device,
                    TRANSLATION_SCALE=args.TRANSLATION_SCALE,
                    args=args,
                )
                next_step_pil.save("prevFrameScaled.png")

                ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
                if args.turbo_mode:
                    if frame_num == args.turbo_preroll:  # start tracking oldframe
                        next_step_pil.save("oldFrameScaled.png")  # stash for later blending
                    elif frame_num > args.turbo_preroll:
                        # set up 2 warped image sequences, old & new, to blend toward new diff image
                        old_frame = do_3d_step(
                            "oldFrameScaled.png",
                            frame_num,
                            midas_model,
                            midas_transform,
                        )
                        old_frame.save("oldFrameScaled.png")
                        if frame_num % int(args.turbo_steps) != 0:
                            print("turbo skip this frame: skipping clip diffusion steps")
                            filename = f"{args.batch_name}({args.batchNum})_{frame_num:04}.png"
                            blend_factor = ((frame_num % int(args.turbo_steps)) + 1) / int(args.turbo_steps)
                            print("turbo skip this frame: skipping clip diffusion steps and saving blended frame")
                            newWarpedImg = cv2.imread("prevFrameScaled.png")  # this is already updated..
                            oldWarpedImg = cv2.imread("oldFrameScaled.png")
                            blendedImage = cv2.addWeighted(
                                newWarpedImg,
                                blend_factor,
                                oldWarpedImg,
                                1 - blend_factor,
                                0.0,
                            )
                            cv2.imwrite(f"{args.batchFolder}/{filename}", blendedImage)
                            next_step_pil.save(f"{img_filepath}")  # save it also as prev_frame to feed next iteration
                            continue
                        else:
                            # if not a skip frame, will run diffusion and need to blend.
                            oldWarpedImg = cv2.imread("prevFrameScaled.png")
                            cv2.imwrite(f"oldFrameScaled.png", oldWarpedImg)  # swap in for blending later
                            print("clip/diff this frame - generate clip diff image")

                init_image = "prevFrameScaled.png"
                init_scale = args.frames_scale
                skip_steps = args.calc_frames_skip_steps

        if args.animation_mode == "Video Input":
            if not args.video_init_seed_continuity:
                seed += 1
            init_image = f"{args.videoFramesFolder}/{frame_num+1:04}.jpg"
            init_scale = args.frames_scale
            skip_steps = args.calc_frames_skip_steps

        loss_values = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        target_embeds, weights = [], []

        prompts_series = split_prompts(args.prompts_series, max_frames=args.max_frames) if args.prompts_series else None
        if prompts_series is not None and frame_num >= len(prompts_series):
            frame_prompt = prompts_series[-1]
            # print(f'Text Prompt: {frame_prompt}`')
        elif args.prompts_series is not None:
            frame_prompt = prompts_series[frame_num]
        else:
            frame_prompt = []

        image_prompts_series = (split_prompts(args.image_prompts_series, max_frames=args.max_frames) if args.image_prompts_series else None,)
        if image_prompts_series is not None and frame_num >= len(image_prompts_series):
            image_prompt = image_prompts_series[-1]
            # print(f'🖼️ Image Prompt: {image_prompt}`')
        elif args.image_prompts_series is not None:
            image_prompt = image_prompts_series[frame_num]
        else:
            image_prompt = []

        print(f"Frame {frame_num} 📝 Prompt: {frame_prompt}")

        model_stats = []
        for clip_model in clip_models:
            cutn = 16
            model_stat = {
                "clip_model": None,
                "target_embeds": [],
                "make_cutouts": None,
                "weights": [],
            }
            model_stat["clip_model"] = clip_model

            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append((txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0, 1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=args.skip_augs)
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert("RGB")
                    img = TF.resize(
                        img,
                        min(args.side_x, args.side_y, *img.size),
                        T.InterpolationMode.LANCZOS,
                    )
                    batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                    embed = clip_model.encode_image(normalize(batch)).float()
                    if args.fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * args.rand_mag).clamp(0, 1))
                            weights.extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError("The weights must not sum to 0.")
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert("RGB")
            init = init.resize((args.side_x, args.side_y), resample=Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if args.perlin_init:
            if args.perlin_mode == "color":
                init = create_perlin_noise(
                    [1.5**-i * 0.5 for i in range(12)],
                    1,
                    1,
                    False,
                    side_x=args.side_x,
                    side_y=args.side_y,
                )
                init2 = create_perlin_noise(
                    [1.5**-i * 0.5 for i in range(8)],
                    4,
                    4,
                    False,
                    side_x=args.side_x,
                    side_y=args.side_y,
                )
            elif args.perlin_mode == "gray":
                init = create_perlin_noise(
                    [1.5**-i * 0.5 for i in range(12)],
                    1,
                    1,
                    True,
                    side_x=args.side_x,
                    side_y=args.side_y,
                )
                init2 = create_perlin_noise(
                    [1.5**-i * 0.5 for i in range(8)],
                    4,
                    4,
                    True,
                    side_x=args.side_x,
                    side_y=args.side_y,
                )
            else:
                init = create_perlin_noise(
                    [1.5**-i * 0.5 for i in range(12)],
                    1,
                    1,
                    False,
                    side_x=args.side_x,
                    side_y=args.side_y,
                )
                init2 = create_perlin_noise(
                    [1.5**-i * 0.5 for i in range(8)],
                    4,
                    4,
                    True,
                    side_x=args.side_x,
                    side_y=args.side_y,
                )
            # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
            init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
            del init2

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if args.use_secondary_model is True:
                    alpha = torch.tensor(
                        diffusion.sqrt_alphas_cumprod[cur_t],
                        device=device,
                        dtype=torch.float32,
                    )
                    sigma = torch.tensor(
                        diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                        device=device,
                        dtype=torch.float32,
                    )
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={"y": y})
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = int(t.item()) + 1  # errors on last step without +1, need to find source
                        # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution = model_stat["clip_model"].visual.input_resolution
                        except:
                            input_resolution = 224

                        cuts = MakeCutoutsDango(
                            input_resolution,
                            args=args,
                            Overview=eval(args.cut_overview)[1000 - t_int],
                            InnerCrop=eval(args.cut_innercut)[1000 - t_int],
                            IC_Size_Pow=args.cut_ic_pow,
                            IC_Grey_P=eval(args.cut_icgray_p)[1000 - t_int],
                        )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(
                            image_embeds.unsqueeze(1),
                            model_stat["target_embeds"].unsqueeze(0),
                        )
                        dists = dists.view(
                            [
                                eval(args.cut_overview)[1000 - t_int] + eval(args.cut_innercut)[1000 - t_int],
                                n,
                                -1,
                            ]
                        )
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item())  # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += torch.autograd.grad(losses.sum() * args.clip_guidance_scale, x_in)[0] / args.cutn_batches
                tv_losses = tv_loss(x_in)
                if args.use_secondary_model is True:
                    range_losses = range_loss(out)
                else:
                    range_losses = range_loss(out["pred_xstart"])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = tv_losses.sum() * args.tv_scale + range_losses.sum() * args.range_scale + sat_losses.sum() * args.sat_scale
                if init is not None and args.init_scale:
                    init_losses = lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * args.init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if torch.isnan(x_in_grad).any() == False:
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and x_is_NaN == False:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=args.clamp_max) / magnitude  # min=-0.02, min=-clamp_max,
            return grad

        if args.diffusion_sampling_mode == "ddim":
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.plms_sample_loop_progressive

        image_display = Output()
        for i in range(args.n_batches):
            if args.animation_mode == "None":
                display.clear_output(wait=True)
                batchBar = tqdm(range(args.n_batches), ncols=40, dynamic_ncols=True, desc="Batches", position=0, leave=True)
                batchBar.n = i
                batchBar.refresh()
            # print('')
            if is_in_notebook():
                display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_steps - 1
            total_steps = cur_t

            if args.perlin_init:
                init = regen_perlin(
                    perlin_mode=args.perlin_mode,
                    device=device,
                    batch_size=args.batch_size,
                )

            if args.diffusion_sampling_mode == "ddim":
                samples = sample_fn(
                    model,
                    (args.batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=args.randomize_class,
                    eta=args.eta,
                )
            else:
                samples = sample_fn(
                    model,
                    (args.batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=args.randomize_class,
                    order=2,
                )

            for j, sample in enumerate(samples):
                cur_t -= 1
                intermediateStep = False
                if args.steps_per_checkpoint is not None:
                    if j % args.steps_per_checkpoint == 0 and j > 0:
                        intermediateStep = True
                elif j in args.intermediate_saves:
                    intermediateStep = True
                with image_display:
                    if j % args.display_rate == 0 or cur_t == -1 or intermediateStep == True:
                        for k, image in enumerate(sample["pred_xstart"]):
                            tqdm.write(f"Batch {i}, step {j}, output {k}:")
                            tqdm.write(datetime.now().strftime("%y%m%d-%H%M%S_%f"))
                            percent = math.ceil(j / total_steps * 100)
                            if args.n_batches > 0:
                                # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                                if cur_t == -1 and args.intermediates_in_subfolder is True:
                                    save_num = f"{frame_num:04}" if args.animation_mode != "None" else i
                                    filename = f"{args.batch_name}({args.batchNum})_{save_num}.png"
                                else:
                                    # If we're working with percentages, append it
                                    if args.steps_per_checkpoint is not None:
                                        filename = f"{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png"
                                    # Or else, iIf we're working with specific steps, append those
                                    else:
                                        filename = f"{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png"
                            image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                            if j % args.display_rate == 0 or cur_t == -1:
                                # image.save('progress.png')
                                image.save(f"{args.batchFolder}/progress.png")
                                # prints output on console.
                                if is_in_notebook():
                                    display.clear_output(wait=True)
                                    display.display(image)
                                if args.console_preview:
                                    output = climage.convert(
                                        f"{args.batchFolder}/progress.png",
                                        width=args.console_preview_width,
                                    )
                                    tqdm.write(output)
                            if args.steps_per_checkpoint is not None:
                                if j % args.steps_per_checkpoint == 0 and j > 0:
                                    if args.intermediates_in_subfolder is True:
                                        image.save(f"{args.partialFolder}/{filename}")
                                    else:
                                        image.save(f"{args.batchFolder}/{filename}")
                            else:
                                if j in args.intermediate_saves:
                                    if args.intermediates_in_subfolder is True:
                                        image.save(f"{args.partialFolder}/{filename}")
                                    else:
                                        image.save(f"{args.batchFolder}/{filename}")
                            if cur_t == -1:
                                if frame_num == 0:
                                    save_settings(
                                        args,
                                        batchFolder=args.batchFolder,
                                        batch_name=args.batch_name,
                                        batchNum=args.batchNum,
                                    )
                                if args.animation_mode != "None":
                                    image.save("prevFrame.png")
                                image.save(f"{args.batchFolder}/{filename}")
                                if args.animation_mode == "3D":
                                    # If turbo, save a blended image
                                    if args.turbo_mode and frame_num > 0:
                                        # Mix new image with prevFrameScaled
                                        blend_factor = (1) / int(args.turbo_steps)
                                        newFrame = cv2.imread("prevFrame.png")  # This is already updated..
                                        prev_frame_warped = cv2.imread("prevFrameScaled.png")
                                        blendedImage = cv2.addWeighted(
                                            newFrame,
                                            blend_factor,
                                            prev_frame_warped,
                                            (1 - blend_factor),
                                            0.0,
                                        )
                                        cv2.imwrite(
                                            f"{args.batchFolder}/{filename}",
                                            blendedImage,
                                        )
                                    else:
                                        image.save(f"{args.batchFolder}/{filename}")

                                    if args.vr_mode:
                                        generate_eye_views(
                                            args.TRANSLATION_SCALE,
                                            args.batchFolder,
                                            filename,
                                            frame_num,
                                            midas_model,
                                            midas_transform,
                                            vr_eye_angle=args.vr_eye_angle,
                                            vr_ipd=args.vr_ipd,
                                            device=device,
                                            args=args,
                                        )

                                # if frame_num != args.max_frames-1:
                                #   display.clear_output()

            # plt.plot(np.array(loss_values), 'r')


def createVideo(args):
    latest_run = args.batchNum
    folder = args.batch_name  # @param
    run = latest_run  # @param
    final_frame = "final_frame"

    init_frame = 1  # @param {type:"number"} This is the frame where the video will start
    last_frame = (
        final_frame  # @param {type:"number"} You can change i to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.
    )
    fps = 12  # @param {type:"number"}
    # view_video_in_cell = True #@param {type: 'boolean'}
    frames = []
    # tqdm.write('Generating video...')
    if last_frame == "final_frame":
        last_frame = len(glob(args.batchFolder + f"/{folder}({run})_*.png"))
        print(f"Total frames: {last_frame}")

    image_path = f"{args.batchFolder}/({run})_%04d.png"
    filepath = f"{args.batchFolder}({run}).mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-start_number",
        str(init_frame),
        "-i",
        image_path,
        "-frames:v",
        str(last_frame + 1),
        "-c:v",
        "libx264",
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "17",
        "-preset",
        "veryslow",
        filepath,
    ]

    process = subprocess.Popen(cmd, cwd=f"{args.batchFolder}", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    else:
        print("The video is ready and saved to the images folder")


def setupFolders(is_colab=False, PROJECT_DIR=None, pargs=None):
    from pydotted import pydot

    batch_name = pargs.batch_name
    videoFramesFolder = None
    partialFolder = None
    folders = pydot(
        {
            "root_path": os.getcwd(),
            "batch_folder": f"{PROJECT_DIR}/images_out/{batch_name}",
            "initDirPath": f"{PROJECT_DIR}/init_images",
            "outDirPath": f"{PROJECT_DIR}/images_out",
            "model_path": f"{PROJECT_DIR}/models",
            "pretrain_path": f"{PROJECT_DIR}/pretrained",
        }
    )

    createPath(folders.batch_folder)
    createPath(folders.initDirPath)
    createPath(folders.outDirPath)
    createPath(folders.model_path)
    createPath(folders.pretrain_path)

    return folders


def loadModels(folders):
    import wget

    # Download models if not present
    for m in [
        {
            "file": f"{folders.model_path}/dpt_large-midas-2f21e586.pt",
            "url": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        },
        {
            "file": f"{folders.model_path}/512x512_diffusion_uncond_finetune_008100.pt",
            "url": "https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt",
        },
        {
            "file": f"{folders.model_path}/256x256_diffusion_uncond.pt",
            "url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
        },
        {
            "file": f"{folders.model_path}/secondary_model_imagenet_2.pth",
            "url": "https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth",
        },
        {
            "file": f"{folders.pretrain_path}/AdaBins_nyu.pt",
            "url": "https://cloudflare-ipfs.com/ipfs/Qmd2mMnDLWePKmgfS8m6ntAg4nhV5VkUyAydYBp8cWWeB7/AdaBins_nyu.pt",
        },
    ]:
        if not os.path.exists(f'{m["file"]}'):
            print(f'🌍 (First time setup): Downloading model from {m["url"]} to {m["file"]}')
            wget.download(m["url"], m["file"])
        else:
            print(f'✅ Model already downloaded: {m["file"]}')


def start_run(pargs=None, folders=None, device=None, is_colab=False):
    import sys
    USE_ADABINS = True
    TRANSLATION_SCALE = 1.0 / 200.0
    MAX_ADABINS_AREA = 500000
    videoFramesFolder = None
    partialFolder = None
    # Get corrected sizes
    side_x = (pargs.width_height[0] // 64) * 64
    side_y = (pargs.width_height[1] // 64) * 64
    if side_x != pargs.width_height[0] or side_y != pargs.width_height[1]:
        print(f"Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.")

    if pargs.animation_mode == "Video Input":
        videoFramesFolder = f"videoFrames"
        createPath(videoFramesFolder)
        print(f"Exporting Video Frames (1 every {pargs.extract_nth_frame})...")
        pargs.max_frames = len(glob(f"{videoFramesFolder}/*.jpg"))
        try:
            for f in pathlib.Path(f"{videoFramesFolder}").glob("*.jpg"):
                f.unlink()
        except:
            print("")
        vf = f"select=not(mod(n\,{pargs.extract_nth_frame}))"
        subprocess.run(
            f"ffmpeg -i {pargs.video_init_path} -vf f{vf} -vsync -vfr -q:v 2 -loglevel error -stats {videoFramesFolder}/%04d.jpg".split(" "), stdout=subprocess.PIPE
        ).stdout.decode("utf-8")

    # Insist turbo be used only w 3d anim.
    if pargs.animation_mode != "3D" and (pargs.turbo_mode or pargs.vr_mode):
        print("⚠️ Turbo/VR modes only available with 3D animations. Disabling... ⚠️")
        pargs.turbo_mode = False
        pargs.vr_mode = False

    if type(pargs.intermediate_saves) is not list:
        if pargs.intermediate_saves:
            steps_per_checkpoint = math.floor((pargs.steps - pargs.skip_steps - 1) // (pargs.intermediate_saves + 1))
            steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
            print(f"Will save every {steps_per_checkpoint} steps")
        else:
            steps_per_checkpoint = pargs.steps + 10
    else:
        steps_per_checkpoint = None

    if pargs.intermediate_saves and pargs.intermediates_in_subfolder is True:
        partialFolder = f"{folders.batch_folder}/partials"
        createPath(partialFolder)

    if pargs.retain_overwritten_frames is True:
        retainFolder = f"{folders.batch_folder}/retained"
        createPath(retainFolder)

    if pargs.cutout_debug is True:
        cutoutDebugFolder = f"{folders.batch_folder}/debug"
        createPath(cutoutDebugFolder)

    skip_step_ratio = int(pargs.frames_skip_steps.rstrip("%")) / 100
    calc_frames_skip_steps = math.floor(pargs.steps * skip_step_ratio)

    if pargs.steps <= calc_frames_skip_steps:
        sys.exit("⚠️ ERROR: You can't skip more steps than your total steps ⚠️")

    if pargs.resume_run:
        if pargs.run_to_resume == "latest":
            try:
                batchNum  # type: ignore
            except:
                batchNum = len(glob(f"{folders.batch_folder}/{pargs.batch_name}(*)_settings.txt")) - 1
        else:
            batchNum = int(pargs.run_to_resume)
        if pargs.resume_from_frame == "latest":
            start_frame = len(glob(folders.batch_folder + f"/{pargs.batch_name}({batchNum})_*.png"))
            if pargs.animation_mode != "3D" and pargs.turbo_mode == True and start_frame > pargs.turbo_preroll and start_frame % int(pargs.turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(pargs.turbo_steps))
        else:
            start_frame = int(pargs.resume_from_frame) + 1
            if pargs.animation_mode != "3D" and pargs.turbo_mode == True and start_frame > pargs.urbo_preroll and start_frame % int(pargs.turbo_steps) != 0:
                start_frame = start_frame - (start_frame % int(pargs.turbo_steps))
            if pargs.retain_overwritten_frames is True:
                existing_frames = len(glob(folders.batch_folder + f"/{pargs.batch_name}({batchNum})_*.png"))
                frames_to_save = existing_frames - start_frame
                print(f"Moving {frames_to_save} frames to the Retained folder")
                move_files(
                    start_frame,
                    existing_frames,
                    folders.batchFolder,
                    retainFolder,
                    batch_name=pargs.batch_name,
                    batchNum=batchNum,
                )
    else:
        start_frame = 0
        batchNum = len(glob(folders.batch_folder + "/*.txt"))
        while (
            os.path.isfile(f"{folders.batch_folder}/{pargs.batch_name}({batchNum})_settings.txt") is True
            or os.path.isfile(f"{folders.batch_folder}/{pargs.batch_name}-{batchNum}_settings.txt") is True
        ):
            batchNum += 1

    if pargs.set_seed == "random_seed":
        random.seed()
        seed = random.randint(0, 2**32)
        print(f"🌱 Randomly using seed: {seed}")
    else:
        seed = int(pargs.set_seed)

    args = {
        "use_checkpoint": pargs.use_checkpoint,
        "cutout_debug": pargs.cutout_debug,
        "ViTB32": pargs.ViTB32,
        "ViTB16": pargs.ViTB16,
        "ViTL14": pargs.ViTL14,
        "ViTL14_336": pargs.ViTL14_336,
        "RN50": pargs.RN50,
        "RN50x4": pargs.RN50x4,
        "RN50x16": pargs.RN50x16,
        "RN50x64": pargs.RN50x64,
        "RN101": pargs.RN101,
        "diffusion_sampling_mode": pargs.diffusion_sampling_mode,
        "width_height": pargs.width_height,
        "clip_guidance_scale": pargs.clip_guidance_scale,
        "tv_scale": pargs.tv_scale,
        "range_scale": pargs.range_scale,
        "sat_scale": pargs.sat_scale,
        "cutn_batches": pargs.cutn_batches,
        "use_secondary_model": pargs.use_secondary_model,
        "diffusion_model": pargs.diffusion_model,
        "animation_mode": pargs.animation_mode,
        "batchNum": batchNum,
        "prompts_series": pargs.text_prompts,
        "text_prompts": pargs.text_prompts,
        "console_preview": pargs.console_preview,
        "console_preview_width": pargs.console_preview_width,
        "image_prompts_series": pargs.image_prompts,
        "seed": seed,
        "display_rate": pargs.display_rate,
        "n_batches": pargs.n_batches if pargs.animation_mode == "None" else 1,
        "batch_size": 1,
        "batch_name": pargs.batch_name,
        "steps": pargs.steps,
        "init_image": pargs.init_image,
        "init_scale": pargs.init_scale,
        "skip_steps": pargs.skip_steps,
        "side_x": side_x,
        "side_y": side_y,
        "animation_mode": pargs.animation_mode,
        "video_init_path": pargs.video_init_path,
        "extract_nth_frame": pargs.extract_nth_frame,
        "video_init_seed_continuity": pargs.video_init_seed_continuity,
        "key_frames": pargs.key_frames,
        "max_frames": pargs.max_frames if pargs.animation_mode != "None" else 1,
        "interp_spline": pargs.interp_spline,
        "start_frame": start_frame,
        "angle": pargs.angle,
        "zoom": pargs.zoom,
        "translation_x": pargs.translation_x,
        "translation_y": pargs.translation_y,
        "translation_z": pargs.translation_z,
        "rotation_3d_x": pargs.rotation_3d_x,
        "rotation_3d_y": pargs.rotation_3d_y,
        "rotation_3d_z": pargs.rotation_3d_z,
        "midas_depth_model": pargs.midas_depth_model,
        "midas_weight": pargs.midas_weight,
        "near_plane": pargs.near_plane,
        "far_plane": pargs.far_plane,
        "fov": pargs.fov,
        "padding_mode": pargs.padding_mode,
        "sampling_mode": pargs.sampling_mode,
        "frames_scale": pargs.frames_scale,
        "calc_frames_skip_steps": calc_frames_skip_steps,
        "skip_step_ratio": skip_step_ratio,
        "calc_frames_skip_steps": calc_frames_skip_steps,
        "image_prompts": pargs.image_prompts,
        "cut_overview": pargs.cut_overview,
        "cut_innercut": pargs.cut_innercut,
        "cut_ic_pow": pargs.cut_ic_pow,
        "cut_icgray_p": pargs.cut_icgray_p,
        "intermediate_saves": pargs.intermediate_saves,
        "intermediates_in_subfolder": pargs.intermediates_in_subfolder,
        "steps_per_checkpoint": steps_per_checkpoint,
        "perlin_init": pargs.perlin_init,
        "perlin_mode": pargs.perlin_mode,
        "set_seed": pargs.set_seed,
        "eta": pargs.eta,
        "clamp_grad": pargs.clamp_grad,
        "clamp_max": pargs.clamp_max,
        "skip_augs": pargs.skip_augs,
        "randomize_class": pargs.randomize_class,
        "clip_denoised": pargs.clip_denoised,
        "fuzzy_prompt": pargs.fuzzy_prompt,
        "rand_mag": pargs.rand_mag,
        "turbo_mode": pargs.turbo_mode,
        "turbo_preroll": pargs.turbo_preroll,
        "turbo_steps": pargs.turbo_steps,
        "video_init_seed_continuity": pargs.video_init_seed_continuity,
        "videoFramesFolder": videoFramesFolder,
        "TRANSLATION_SCALE": TRANSLATION_SCALE,
        "partialFolder": partialFolder,
        "model_path": folders.model_path,
        "batchFolder": folders.batch_folder,
        "resume_run": pargs.resume_run,
    }
    # args = SimpleNamespace(**args)
    args = pydot(args)  # Thx Zippy
    try:
        do_run(
            args=args,
            device=device,
            is_colab=is_colab,
            batchNum=batchNum,
            start_frame=start_frame,
        )
    except KeyboardInterrupt:
        print("🛑 Run interrupted by user.")
        pass
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    if pargs.animation_mode != "None":
        if pargs.skip_video_for_run_all == True:
            print("⚠️ Skipping video creation, uncheck skip_video_for_run_all if you want to run it")
        else:
            createVideo(args)


def systemDetails(pargs):
    if pargs.simple_nvidia_smi_display:
        nvidiasmi_output = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE).stdout.decode("utf-8")
        print(f"🔎 {nvidiasmi_output}")
    else:
        nvidiasmi_output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE).stdout.decode("utf-8")
        print(nvidiasmi_output)
        # nvidiasmi_ecc_note = subprocess.run(["nvidia-smi", "-i", "0"], stdout=subprocess.PIPE).stdout.decode("utf-8")
        # print(nvidiasmi_ecc_note)


def getDevice(pargs):
    import sys

    DEVICE = torch.device(pargs.cuda_device if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", DEVICE)
    device = DEVICE  # At least one of the modules expects this name..
    try:
        # Fails if CPU is set
        if torch.cuda.get_device_capability(DEVICE) == (8, 0):  ## A100 fix thanks to Emad
            print("Disabling CUDNN for A100 gpu", file=sys.stderr)
            torch.backends.cudnn.enabled = False
    except:
        print("Are you using a CPU?  Check your PyTorch version if you get errors.")
        # torch.backends.cudnn.enabled = False
        pass
    return device


def detectColab():
    try:
        from google.colab import drive  # type: ignore

        return True
    except:
        return False


def is_in_notebook():
    import traceback

    rstk = traceback.extract_stack(limit=1)[0]
    return rstk[0].startswith("<ipython")


def fix_later():
    print("Gotta parse Notebook Args")
