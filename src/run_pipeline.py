#!/usr/bin/env python
"""
run_pipeline.py
================
Optional orchestrator: chains baseline training -> DP training -> "compare"
CKA analysis in a single command. Handy for a full "default" run, but for
day-to-day use (iterating on a single epsilon, re-running just the CKA
step, etc.) the separate scripts (train_baseline.py / train_dp.py /
run_cka.py) remain more flexible — see the README.

Example:
    python run_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, add_config_overrides_args, apply_overrides, load_config
from checkpoint import find_latest_checkpoint, make_run_id
from logging_utils import setup_logging

logger = logging.getLogger(__name__)
SRC_DIR = Path(__file__).resolve().parent


def config_to_cli_args(cfg: Config) -> list[str]:
    """Rebuild --xxx flags from an already-resolved Config (YAML file +
    CLI overrides merged), to pass them through as-is to the sub-scripts.
    This way train_baseline.py / train_dp.py / run_cka.py receive exactly
    the same configuration as the orchestrator, without depending on the
    presence of a --config file."""
    flags: list[str] = []
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        value = getattr(cfg, f.name)
        if f.name == "cka_layers":
            flags += [flag, ",".join(value)]
        else:
            flags += [flag, str(value)]
    return flags


def run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit code {result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline: baseline -> DP -> CKA")
    add_config_overrides_args(parser)
    parser.add_argument("--skip-baseline", action="store_true", help="Don't (re)run baseline training")
    parser.add_argument("--skip-dp", action="store_true", help="Don't (re)run DP training")
    parser.add_argument("--skip-cka", action="store_true", help="Don't run the final CKA analysis")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    # Resolve to absolute paths BEFORE passing them to sub-scripts: the
    # latter are launched with cwd=SRC_DIR, so a relative path like
    # "./cifar10" would otherwise be interpreted relative to src/ instead
    # of relative to the directory the user ran run_pipeline.py from.
    cfg.data_root = str(Path(cfg.data_root).resolve())
    cfg.networks_dir = str(Path(cfg.networks_dir).resolve())
    cfg.results_dir = str(Path(cfg.results_dir).resolve())
    cfg.logs_dir = str(Path(cfg.logs_dir).resolve())

    run_id = make_run_id("pipeline", cfg.epsilon, cfg.delta, cfg.epochs, cfg.max_grad_norm, cfg.seed)
    setup_logging(cfg.logs_dir, run_id)
    logger.info("=== Full pipeline: %s ===", run_id)

    config_args = config_to_cli_args(cfg)

    if not args.skip_baseline:
        run([sys.executable, "train_baseline.py", *config_args])
    else:
        logger.info("Baseline step skipped (--skip-baseline).")

    if not args.skip_dp:
        run([sys.executable, "train_dp.py", *config_args])
    else:
        logger.info("DP step skipped (--skip-dp).")

    if not args.skip_cka:
        baseline_ckpt = find_latest_checkpoint(cfg.networks_dir, "baseline_resnet18", seed=cfg.seed)
        dp_ckpt = find_latest_checkpoint(cfg.networks_dir, "dp_resnet18", epsilon=cfg.epsilon, seed=cfg.seed)

        if baseline_ckpt is None or dp_ckpt is None:
            logger.error(
                "Could not locate checkpoints for the CKA analysis "
                "(baseline=%s, dp=%s). Run run_cka.py manually.",
                baseline_ckpt, dp_ckpt,
            )
        else:
            run([
                sys.executable, "run_cka.py", *config_args, "compare",
                "--ckpt-a", str(baseline_ckpt), "--ckpt-b", str(dp_ckpt),
                "--label-a", "Baseline", "--label-b", f"DP_eps{cfg.epsilon:g}",
            ])
    else:
        logger.info("CKA step skipped (--skip-cka).")

    logger.info("=== Pipeline finished ===")


if __name__ == "__main__":
    main()