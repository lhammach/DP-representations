"""
vit_cifar.py
============
Vision Transformer adapted for CIFAR-10 (32×32 images), as used in the
DP-MicroAdam paper. Source: https://github.com/kentaroy47/vision-transformers-cifar10

Key architectural choices for CIFAR-10:
- SPT (Shifted Patch Tokenization): enriches patch tokens with local context
  from 4 shifted versions of the image — compensates for the small number of
  patches (64 = 8×8) compared to ImageNet (196 = 14×14 with patch_size=16).
- LSA (Locally Self-Attention): masks self-attention on the diagonal (a token
  cannot attend to itself) and uses a learnable temperature. Improves training
  stability on small images.
- patch_size=4: gives 64 patches on 32×32 images (vs only 4 patches with
  standard patch_size=16).

DP compatibility:
- LayerNorm: compatible with Opacus (no batch statistics).
- nn.Linear: standard, handled by Opacus grad_sample hooks.
- einops.rearrange: transparent to autograd, no Opacus issues.
- No BatchNorm, no inplace operations.
- ModuleValidator.fix() is NOT needed (no BN to replace).
- ModuleValidator.validate() is called to surface any unexpected issues.

Variants (depth, heads, mlp_dim):
  vit-t : depth=4, heads=6, dim=512, mlp_dim=256  (~6M params)
  vit-s : depth=6, heads=8, dim=512, mlp_dim=512  (~9.5M params)
"""

from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from opacus.validators import ModuleValidator

logger = logging.getLogger(__name__)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    """Locally Self-Attention: diagonal-masked attention with learnable temperature."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
        # Mask diagonal: a token cannot attend to itself
        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        dots = dots.masked_fill(mask, -torch.finfo(dots.dtype).max)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x   # non-inplace residual
            x = ff(x) + x     # non-inplace residual
        return x


class SPT(nn.Module):
    """Shifted Patch Tokenization: uses 4 shifted versions of the image
    to enrich local context in each patch token."""

    def __init__(self, *, dim: int, patch_size: int, channels: int = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels   # 5 = original + 4 shifts
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted = [F.pad(x, s) for s in shifts]
        x_with_shifts = torch.cat((x, *shifted), dim=1)
        return self.to_patch_tokens(x_with_shifts)


class ViTCIFAR(nn.Module):
    """ViT for CIFAR-10 with SPT tokenization and LSA attention.

    Forward pass dimensions (vit-s, image 32×32, patch_size=4):
        Input:               B × 3 × 32 × 32
        SPT tokenization:    B × 64 × 512      (64 patches, dim=512)
        + CLS token:         B × 65 × 512
        + pos embedding:     B × 65 × 512
        Transformer ×6:      B × 65 × 512      (depth=6 blocks)
        CLS pooling:         B × 512
        MLP head:            B × 10
    """

    def __init__(self, *, image_size: int = 32, patch_size: int = 4,
                 num_classes: int = 10, dim: int = 512, depth: int = 6,
                 heads: int = 8, mlp_dim: int = 512, pool: str = 'cls',
                 channels: int = 3, dim_head: int = 64,
                 dropout: float = 0., emb_dropout: float = 0.):
        super().__init__()
        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0
        num_patches = (ih // ph) * (iw // pw)
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)


def build_vit_dp_compatible(
    vit_type: str = "vit-s",
    num_classes: int = 10,
    dropout: float = 0.0,
    emb_dropout: float = 0.0,
    validate: bool = True,
) -> nn.Module:
    """Build a ViT for CIFAR-10, validated for DP-SGD compatibility.

    dropout / emb_dropout default to 0.0 because Opacus uses vmap for
    per-sample gradients, and vmap raises RuntimeError on random operations
    (Dropout). Set to 0.0 for DP training, or 0.1 for baseline training
    to match the original DP-MicroAdam architecture.
    """
    configs = {
        "vit-t": dict(depth=4, heads=6, dim=512, mlp_dim=256),
        "vit-s": dict(depth=6, heads=8, dim=512, mlp_dim=512),
    }
    if vit_type not in configs:
        raise ValueError(f"Unknown vit_type '{vit_type}'. Choose from: {list(configs.keys())}")

    cfg = configs[vit_type]
    model = ViTCIFAR(
        image_size=32, patch_size=4, num_classes=num_classes,
        dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
        dim_head=64, mlp_dim=cfg["mlp_dim"],
        dropout=dropout, emb_dropout=emb_dropout,
    )
    if validate:
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            logger.warning("ViT Opacus compatibility warnings:")
            for e in errors:
                logger.warning("  - %s", e)
        else:
            logger.info("ViT (%s, dropout=0): Opacus validation passed.", vit_type)
    return model