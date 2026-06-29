"""
logging_utils.py
=================
Consistent logging configuration for all scripts: console output + a log
file per run (named with the same run_id as the checkpoints, so everything
can be easily traced back).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(logs_dir: str | Path, run_id: str, level: int = logging.INFO) -> Path:
    """Configure the root logger: console + `{logs_dir}/{run_id}.log` file."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_id}.log"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return log_path