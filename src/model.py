"""
model.py
========
Definition of the "DP-compatible" ResNet18 used both for training (baseline
and DP-SGD) and for CKA analysis.

Centralizing this definition here guarantees that the training scripts and
the CKA script instantiate EXACTLY the same architecture, which is required
for `load_state_dict` to succeed without errors and for the representation
comparison to be valid.

Modifications applied relative to the standard torchvision ResNet18:
- BatchNorm -> GroupNorm (BatchNorm is not compatible with the per-sample
  gradient computation used by Opacus)
- ReLU(inplace=False) everywhere (inplace operations break the autograd
  graph needed for per-sample gradients)
- Non-inplace residual addition in BasicBlock (`out = out + identity`
  instead of `out += identity`)
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock
from opacus.validators import ModuleValidator

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def _safe_basicblock_forward(self, x):
    """BasicBlock forward pass without any inplace operations."""
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out = out + identity  # non-inplace, required for Opacus
    out = self.relu(out)

    return out


def _ensure_basicblock_patched() -> None:
    """Globally patch BasicBlock.forward (idempotent)."""
    global _PATCH_APPLIED
    if not _PATCH_APPLIED:
        BasicBlock.forward = _safe_basicblock_forward
        _PATCH_APPLIED = True
        logger.debug("BasicBlock.forward patched (non-inplace residual).")


def _disable_inplace_relu(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def build_resnet18_dp_compatible(num_classes: int = 10, validate: bool = True) -> nn.Module:
    """
    Build a ResNet18 ready for DP-SGD (and therefore reloadable identically
    on the CKA side).

    Args:
        num_classes: number of output classes (10 for CIFAR-10).
        validate: if True, surfaces any remaining Opacus compatibility
            warnings (should normally print nothing once the model is fixed).

    Returns:
        An nn.Module ResNet18 with GroupNorm, non-inplace ReLU, and a
        non-inplace residual addition.
    """
    _ensure_basicblock_patched()

    model = models.resnet18(num_classes=num_classes)
    model = ModuleValidator.fix(model)  # BatchNorm -> GroupNorm
    _disable_inplace_relu(model)

    if validate:
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.warning("Remaining Opacus compatibility warnings:")
            for err in errors:
                logger.warning("  - %s", err)

    return model