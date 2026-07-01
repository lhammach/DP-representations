"""
config.py
=========
Centralized configuration management (hyperparameters, paths).

Typical usage:
    cfg = load_config("configs/default.yaml")
    cfg = apply_overrides(cfg, args)  # args = argparse.Namespace

The YAML file remains the "long-term" source of truth (what you want to
reuse from one run to the next); CLI arguments are for one-off overrides
(e.g. quickly trying epsilon=2 without touching the file).
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # --- Data ---
    data_root: str = "./cifar10"
    cifar10_url: str = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"

    # --- Common hyperparameters ---
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 128
    seed: int = 42

    # --- DP-SGD hyperparameters ---
    epsilon: float = 10.0
    delta: float = 1e-8
    max_grad_norm: float = 1.2
    max_physical_batch_size: int = 32
    accountant: str = "rdp"

    # --- CKA ---
    cka_batch_size: int = 256
    cka_num_batches: int = 30
    cka_kernel: str = "linear"  # "linear" or "gaussian"
    cka_layers: list[str] = field(
        default_factory=lambda: ["layer1", "layer2", "layer3", "layer4"]
    )

    # --- Outputs ---
    networks_dir: str = "./networks"
    results_dir: str = "./results"
    logs_dir: str = "./logs"
    experiment: str = "quick_tests"  # subfolder under networks/ and results/, e.g. "seed_sweep"

    # --- Misc ---
    num_classes: int = 10

    def networks_path(self) -> str:
        """networks_dir, with the experiment subfolder appended if set."""
        return str(Path(self.networks_dir) / self.experiment) if self.experiment else self.networks_dir

    def results_path(self) -> str:
        """results_dir, with the experiment subfolder appended if set."""
        return str(Path(self.results_dir) / self.experiment) if self.experiment else self.results_dir

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        unknown = set(d) - valid_keys
        if unknown:
            import logging

            logging.getLogger(__name__).warning(
                "Ignored config keys (unknown): %s", sorted(unknown)
            )
        return cls(**filtered)


def load_config(path: str | Path | None) -> Config:
    """Load a Config from a YAML file, or return the default values."""
    if path is None:
        return Config()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config.from_dict(raw)


def add_config_overrides_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI arguments allowing any Config field to be overridden."""
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file")
    for f in dataclasses.fields(Config):
        if f.name == "cka_layers":
            parser.add_argument(
                "--cka-layers", type=str, default=None,
                help="Comma-separated list of layers, e.g. layer1,layer2,layer3,layer4",
            )
            continue
        flag = "--" + f.name.replace("_", "-")
        arg_type = type(f.default) if f.default is not None else str
        parser.add_argument(flag, type=arg_type, default=None, help=f"Override for '{f.name}'")
    return parser


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply non-None CLI overrides on top of an existing Config."""
    cfg = dataclasses.replace(cfg)
    args_dict = vars(args)

    if args_dict.get("cka_layers"):
        cfg.cka_layers = [s.strip() for s in args_dict["cka_layers"].split(",") if s.strip()]

    for f in dataclasses.fields(Config):
        if f.name == "cka_layers":
            continue
        val = args_dict.get(f.name)
        if val is not None:
            setattr(cfg, f.name, val)
    return cfg