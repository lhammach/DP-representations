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

Optional CIFAR-10 stem modification (--cifar-stem flag):
- The default ResNet18 stem (conv1 7x7 stride=2 + MaxPool) was designed for
  ImageNet images (224x224). On CIFAR-10 (32x32) it immediately shrinks the
  spatial resolution to 7x7 before any residual block, destroying most of
  the spatial information. Replacing it with a 3x3 stride=1 conv + no
  MaxPool keeps the resolution at 32x32 through the stem, which is the
  standard approach used in the DP-SGD literature on CIFAR-10
  (e.g. Wide-ResNet papers, DP-MicroAdam).
  This flag MUST be set consistently between training and CKA loading.
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


def _apply_cifar_stem(model: nn.Module) -> nn.Module:
    """Replace the ImageNet stem with a CIFAR-10-friendly stem.

    Default stem: Conv2d(3, 64, kernel_size=7, stride=2, padding=3) + MaxPool2d
    → halves spatial resolution twice → 32x32 input becomes 7x7 after stem.

    CIFAR stem: Conv2d(3, 64, kernel_size=3, stride=1, padding=1) + Identity
    → preserves resolution → 32x32 input stays 32x32 after stem.

    Note: applied BEFORE ModuleValidator.fix() to ensure the new conv1 gets
    its GroupNorm replacement handled by Opacus correctly.
    """
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    logger.info("CIFAR-10 stem applied: conv1 3x3 stride=1, maxpool disabled.")
    return model


def build_resnet18_dp_compatible(
    num_classes: int = 10,
    cifar_stem: bool = False,
    validate: bool = True,
) -> nn.Module:
    """
    Build a ResNet18 ready for DP-SGD (and therefore reloadable identically
    on the CKA side).

    Args:
        num_classes: number of output classes (10 for CIFAR-10).
        cifar_stem: if True, replaces the ImageNet stem (7x7 conv + MaxPool)
            with a CIFAR-10-adapted stem (3x3 conv, no MaxPool) that preserves
            the 32x32 resolution. MUST be set identically at training time and
            at CKA loading time — the checkpoint name includes 'cifarSTEM' when
            this is active to avoid silent mismatches.
        validate: if True, surfaces any remaining Opacus compatibility
            warnings (should normally print nothing once the model is fixed).

    Returns:
        An nn.Module ResNet18 with GroupNorm, non-inplace ReLU, and a
        non-inplace residual addition.
    """
    _ensure_basicblock_patched()

    model = models.resnet18(num_classes=num_classes)

    if cifar_stem:
        model = _apply_cifar_stem(model)

    model = ModuleValidator.fix(model)  # BatchNorm -> GroupNorm
    _disable_inplace_relu(model)

    if validate:
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.warning("Remaining Opacus compatibility warnings:")
            for err in errors:
                logger.warning("  - %s", err)

    return model


def build_model(
    model_name: str,
    num_classes: int = 10,
    cifar_stem: bool = False,
    validate: bool = True,
) -> nn.Module:
    """Build the requested model architecture, DP-compatible.

    Args:
        model_name: one of:
            "resnet18"           — ResNet18 with GroupNorm (default)
            "wideresnet-16-4"   — WideResNet depth=16, k=4  (~2.7M params)
            "wideresnet-16-8"   — WideResNet depth=16, k=8  (~11M params)
            "wideresnet-40-4"   — WideResNet depth=40, k=4  (~8.9M params)
        cifar_stem: only applies to resnet18 (WideResNet always uses a 3×3
            stem without MaxPool natively).
        num_classes: number of output classes.

    Returns:
        A DP-compatible nn.Module.
    """
    name = model_name.lower().strip()
    if name == "resnet18":
        return build_resnet18_dp_compatible(num_classes=num_classes,
                                            cifar_stem=cifar_stem,
                                            validate=validate)
    elif name in ("wideresnet-16-4", "wrn-16-4"):
        from wide_resnet import build_wide_resnet_dp_compatible
        return build_wide_resnet_dp_compatible(depth=16, widen_factor=4,
                                               num_classes=num_classes,
                                               validate=validate)
    elif name in ("wideresnet-16-8", "wrn-16-8"):
        from wide_resnet import build_wide_resnet_dp_compatible
        return build_wide_resnet_dp_compatible(depth=16, widen_factor=8,
                                               num_classes=num_classes,
                                               validate=validate)
    elif name in ("wideresnet-40-4", "wrn-40-4"):
        from wide_resnet import build_wide_resnet_dp_compatible
        return build_wide_resnet_dp_compatible(depth=40, widen_factor=4,
                                               num_classes=num_classes,
                                               validate=validate)
    elif name in ("vit-t", "vit-s"):
        try:
            from vit import build_vit_dp_compatible
        except ImportError:
            from vit_cifar import build_vit_dp_compatible
        # Dropout causes RuntimeError with Opacus vmap per-sample gradients.
        # Setting dropout=0 fixes this without changing the architecture.
        return build_vit_dp_compatible(vit_type=name, num_classes=num_classes,
                                       dropout=0.0, emb_dropout=0.0,
                                       validate=validate)
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose from: resnet18, wideresnet-16-4, wideresnet-16-8, "
            "wideresnet-40-4, vit-t, vit-s"
        )


def list_learnable_layers(
    model: nn.Module,
    include_types: tuple[type, ...] = (nn.Conv2d, nn.Linear, nn.GroupNorm),
) -> list[str]:
    """List the dotted names of all sub-modules with learnable weights.

    By default includes Conv2d (all conv1/conv2/downsample projections),
    Linear (the final fc layer), and GroupNorm (the affine scale/shift
    parameters that replaced BatchNorm). This avoids hand-maintaining a
    fixed layer list (like the old DAMIER_LAYERS) that silently goes stale
    if the architecture changes — it always reflects the actual model.

    Pass a narrower `include_types` (e.g. `(nn.Conv2d, nn.Linear)`) to
    exclude normalization layers if you only want the "spatial" weights.

    Returns:
        Layer names in model definition order (top to bottom), e.g.
        ["conv1", "bn1", "layer1.0.conv1", "layer1.0.bn1", ..., "fc"].
    """
    return [name for name, module in model.named_modules() if isinstance(module, include_types) and name]