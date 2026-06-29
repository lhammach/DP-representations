"""
checkpoint.py
=============
Consistent, "safe" checkpoint naming, saving, and reloading.

Naming convention:
    {prefix}_eps{epsilon}_delta{delta}_epoch{epochs}_C{max_grad_norm}_seed{seed}_{timestamp}.pth

The timestamp (format YYYYmmdd-HHMMSS) guarantees that an old run is never
accidentally overwritten, while keeping a readable, chronologically
sortable name. A `metadata.json` next to the `.pth` makes every run
explorable without having to load the full tensor file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _format_delta(delta: float) -> str:
    """1e-08 -> '1e08' (filename-friendly, no minus sign)."""
    return f"{delta:.0e}".replace("-", "")


def make_run_id(
    prefix: str,
    epsilon: float | str,
    delta: float,
    epochs: int,
    max_grad_norm: float,
    seed: int,
    timestamp: str | None = None,
) -> str:
    """Build the run identifier used to name checkpoints and logs."""
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    delta_str = _format_delta(delta)
    eps_str = epsilon if isinstance(epsilon, str) else f"{epsilon:g}"
    return f"{prefix}_eps{eps_str}_delta{delta_str}_epoch{epochs}_C{max_grad_norm}_seed{seed}_{ts}"


def get_checkpoint_path(
    prefix: str,
    epsilon: float | str,
    delta: float,
    epochs: int,
    max_grad_norm: float,
    seed: int,
    save_dir: str | Path,
    ext: str = "pth",
    timestamp: str | None = None,
) -> Path:
    """Generate a unique checkpoint path (timestamp included, so never a collision)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(prefix, epsilon, delta, epochs, max_grad_norm, seed, timestamp)
    return save_dir / f"{run_id}.{ext}"


def save_checkpoint(path: str | Path, payload: dict[str, Any], extra_metadata: dict[str, Any] | None = None) -> None:
    """Save a .pth checkpoint plus a human-readable .json metadata file next to it.

    The .json does NOT contain the weights (only scalars/lists/histories),
    which lets you browse all past runs (e.g. comparing epsilons) without
    loading every .pth into memory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(payload, path)
    logger.info("Checkpoint saved: %s", path)

    meta = {k: v for k, v in payload.items() if k not in ("model_state_dict",)}
    if extra_metadata:
        meta.update(extra_metadata)
    meta["checkpoint_file"] = path.name
    meta["saved_at"] = datetime.now().isoformat(timespec="seconds")

    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Metadata saved: %s", meta_path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)


def find_latest_checkpoint(
    networks_dir: str | Path,
    prefix: str,
    epsilon: float | str | None = None,
    seed: int | None = None,
) -> Path | None:
    """Find the most recent checkpoint matching the given filters.

    Handy for day-to-day use: no need to copy-paste yesterday's exact
    generated filename, just ask for "the latest baseline with seed=42".
    """
    networks_dir = Path(networks_dir)
    if not networks_dir.exists():
        return None

    candidates = sorted(networks_dir.glob(f"{prefix}_*.pth"))

    def matches(p: Path) -> bool:
        name = p.stem
        if epsilon is not None:
            eps_str = epsilon if isinstance(epsilon, str) else f"{epsilon:g}"
            if f"_eps{eps_str}_" not in name:
                return False
        if seed is not None and f"_seed{seed}_" not in name:
            return False
        return True

    matching = [p for p in candidates if matches(p)]
    if not matching:
        return None

    # The timestamp in the name sorts lexicographically = sorts chronologically.
    return matching[-1]