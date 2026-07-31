"""
wide_resnet.py
==============
WideResNet architecture adapted for DP-SGD on CIFAR-10.

Source: https://github.com/MihaelaHudisteanu/DP-Micro-Adam/blob/main/models/wide_resnet.py
(itself from https://github.com/meliketoy/wide-resnet.pytorch)

DP compatibility notes:
- WSConv2d (Weight Standardization) replaces standard Conv2d. It normalises
  the weights themselves, providing activation normalisation without batch
  statistics — intrinsically compatible with Opacus per-sample gradients.
- GroupNorm is used throughout (not BatchNorm).
- All ReLU calls use F.relu() (not inplace).
- Residual addition is non-inplace: out = out + shortcut(x).
- ModuleValidator.fix() is NOT called because WSConv2d is not a standard
  nn.Conv2d — Opacus would not know how to handle it. Instead we rely on
  WSConv2d + GroupNorm being already compatible.
- ModuleValidator.validate() is called at the end to surface any issues.

Available variants (depth, widen_factor):
  wide_resnet_16_4  : depth=16, k=4  (~2.7M params, standard CIFAR-10 DP benchmark)
  wide_resnet_16_8  : depth=16, k=8  (~11M params, comparable to ResNet18)
  wide_resnet_40_4  : depth=40, k=4  (~8.9M params)
"""

from __future__ import annotations

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.validators import ModuleValidator

logger = logging.getLogger(__name__)


class WSConv2d(nn.Module):
    """Conv2d with Weight Standardization.

    Standardises the convolutional filters to zero mean and unit variance
    (per output channel), providing a normalisation effect similar to
    BatchNorm but applied to weights rather than activations — no batch
    statistics required, hence compatible with per-sample gradient computation.

    Forward:
        w_std[c] = (w[c] - mean(w[c])) / sqrt(Var(w[c]) * fan_in + eps) * gain[c]
        out = conv2d(x, w_std) + bias
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.gain = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        fan_in = in_channels * kernel_size * kernel_size
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(1.0 / fan_in))

    def forward(self, x, eps: float = 1e-4):
        w = self.weight
        mean = w.mean(dim=(1, 2, 3), keepdim=True)
        var = w.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        fan_in = w.shape[1] * w.shape[2] * w.shape[3]
        scale = torch.rsqrt(torch.clamp(var * fan_in, min=eps)) \
                * self.gain[:, None, None, None]
        shift = mean * scale
        w_std = w * scale - shift
        return F.conv2d(x, w_std, self.bias, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        groups=self.groups)


def _norm(num_channels: int, groups: int = 16) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=groups, num_channels=num_channels, affine=True)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> WSConv2d:
    return WSConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> WSConv2d:
    return WSConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0)


class WideBasic(nn.Module):
    """Pre-activation residual block for WideResNet.

    Pre-activation order (BN→ReLU→Conv, as in ResNet-v2):
        out = Conv2(ReLU(GN2(Conv1(ReLU(GN1(x))))))  +  shortcut(x)

    This differs from ResNet18's post-activation BasicBlock:
        out = ReLU(Conv2(BN2(ReLU(Conv1(BN1(x))))) + x)
    """

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 groups: int = 16):
        super().__init__()
        self.bn1 = _norm(in_planes, groups=groups)
        self.conv1 = _conv3x3(in_planes, planes, stride=stride)
        self.bn2 = _norm(planes, groups=groups)
        self.conv2 = _conv3x3(planes, planes, stride=1)

        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.ReLU(inplace=False),
                _norm(in_planes, groups=groups),
                _conv1x1(in_planes, planes, stride=stride),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        out = self.bn1(out)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.bn2(out)
        out = self.conv2(out)
        return out + self.shortcut(x)   # non-inplace


class WideResNet(nn.Module):
    """WideResNet for CIFAR-10 DP-SGD.

    Architecture (depth=16, k=4 → WRN-16-4):
        Input:   3 × 32 × 32
        conv1:   16 × 32 × 32   (3×3, stride=1, no MaxPool — CIFAR stem)
        layer1:  64 × 32 × 32   (n=2 blocks, stride=1, k=4 → 16×4=64 channels)
        layer2:  128 × 16 × 16  (n=2 blocks, stride=2, k=4 → 32×4=128 channels)
        layer3:  256 × 8 × 8    (n=2 blocks, stride=2, k=4 → 64×4=256 channels)
        bn1+pool: 256 × 1 × 1
        linear:  256 → num_classes

    n = (depth - 4) / 6  blocks per stage.
    nStages = [16, 16k, 32k, 64k].
    """

    def __init__(self, depth: int, widen_factor: int, num_classes: int,
                 groups: int = 16):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self._in_planes = 16
        self.conv1 = _conv3x3(3, nStages[0])
        self.layer1 = self._make_layer(nStages[1], n, stride=1, groups=groups)
        self.layer2 = self._make_layer(nStages[2], n, stride=2, groups=groups)
        self.layer3 = self._make_layer(nStages[3], n, stride=2, groups=groups)
        self.bn1 = _norm(nStages[3], groups=groups)
        self.linear = nn.Linear(nStages[3], num_classes)

        nn.init.normal_(self.linear.weight, 0.0, self.linear.in_features ** -0.5)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def _make_layer(self, planes: int, num_blocks: int, stride: int,
                    groups: int = 16) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasic(self._in_planes, planes, stride=s, groups=groups))
            self._in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = self.bn1(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


def build_wide_resnet_dp_compatible(
    depth: int = 16,
    widen_factor: int = 4,
    num_classes: int = 10,
    groups: int = 16,
    validate: bool = True,
) -> nn.Module:
    """Build a WideResNet ready for DP-SGD.

    WSConv2d + GroupNorm make this architecture intrinsically DP-compatible
    without needing ModuleValidator.fix() (which only handles nn.Conv2d /
    nn.BatchNorm2d). We validate anyway to surface unexpected issues.
    """
    model = WideResNet(depth=depth, widen_factor=widen_factor,
                       num_classes=num_classes, groups=groups)
    if validate:
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.warning("WideResNet Opacus compatibility warnings:")
            for e in errors:
                logger.warning("  - %s", e)
        else:
            logger.info("WideResNet: Opacus validation passed.")
    return model