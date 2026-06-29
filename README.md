# DP-representations — reproducible pipeline

Comparison of internal representations (via **CKA** — Centered Kernel
Alignment) between a normally trained ResNet18 and a ResNet18 trained with
**DP-SGD** (Opacus) on CIFAR-10.

This repository replaces the two original Colab notebooks
(`dpsgd_baseline.ipynb` and `cka_analysis.ipynb`) with a pipeline of Python
scripts, so experiments can be rerun reproducibly, without copy-pasting
cells or risking inconsistent file naming between two runs.

## Project structure

```
dp_representations/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml          # default config (epsilon=10, seed=42, 10 epochs)
│   └── eps_low.yaml           # example variant (epsilon=2)
├── src/
│   ├── __init__.py            # makes src/ importable as a package (no logic)
│   ├── model.py               # DP-compatible ResNet18 definition (shared architecture)
│   ├── data.py                 # CIFAR-10 download + loading
│   ├── config.py               # Config dataclass + YAML loading + CLI overrides
│   ├── checkpoint.py           # file naming (with timestamp) + save/load
│   ├── training.py             # baseline and DP-SGD training loops
│   ├── cka.py                   # CKA formulas + activation extraction
│   ├── visualization.py        # matplotlib plots (curves, heatmaps)
│   ├── logging_utils.py        # console + file logging
│   ├── train_baseline.py       # CLI script: training without DP
│   ├── train_dp.py              # CLI script: DP-SGD training
│   ├── run_cka.py                # CLI script: CKA analysis (3 subcommands)
│   └── run_pipeline.py          # optional orchestrator (baseline → DP → CKA)
├── networks/                    # checkpoints .pth + .json metadata (generated)
│   └── .gitkeep                 # empty placeholder, see note below
├── results/                      # CKA figures and summaries (generated)
│   └── .gitkeep
└── logs/                          # per-run text logs (generated)
    └── .gitkeep
```

**About `__init__.py`**: an empty (or near-empty) file that marks `src/` as
a regular Python package. It has no role at runtime here (every script
adds `src/` to `sys.path` directly), but it's good practice to keep it so
`src` can also be imported normally (e.g. `from src.model import ...`)
if you ever run things from the repo root instead of from inside `src/`.

**About `.gitkeep`**: an empty file with no special meaning to Git itself
— it's just a convention. Git does not track empty folders, so without
something inside them, `networks/`, `results/`, and `logs/` would simply
not exist after cloning the repo (the scripts do create them automatically
on first run via `Path.mkdir(parents=True, exist_ok=True)`, but committing
a placeholder keeps the expected structure visible from the start). The
`.gitignore` below is written to ignore everything generated inside these
folders **except** `.gitkeep`, so the actual checkpoints/figures/logs never
get committed, but the empty folders themselves do.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
# Create the virtual environment and install all dependencies from requirements.txt
uv venv
uv pip install -r requirements.txt

# Activate it (only needed if you want to run python directly, e.g. python src/train_baseline.py)
source .venv/bin/activate         # or .venv\Scripts\activate on Windows
```

Alternatively, skip the explicit activation step and just prefix any
command with `uv run`, which automatically uses the project's `.venv`:

```bash
uv run src/train_baseline.py --config configs/default.yaml
```

CIFAR-10 is downloaded automatically (from the FastAI mirror) the first
time a training script or `run_cka.py` is run, into `./cifar10`.

## What the pipeline does

### 1. Shared architecture (`model.py`)

The critical point in the original notebooks: to compare the
representations of two models with CKA, **both checkpoints must be
reloadable into a strictly identical architecture**, otherwise
`load_state_dict` either fails outright or — worse — succeeds silently on
a slightly different architecture, and the CKA no longer compares the
same thing.

`model.py` therefore centralizes the construction of the "DP-compatible"
ResNet18 (GroupNorm instead of BatchNorm, non-inplace ReLU, non-inplace
residual addition) in a single place, used both by training (baseline
**and** DP) and by checkpoint loading for CKA. There is no longer any risk
of divergence between the two original notebooks.

### 2. Training (`train_baseline.py` / `train_dp.py`)

Two symmetric scripts:
- `train_baseline.py`: standard optimization (RMSprop), no DP.
- `train_dp.py`: DP-SGD via Opacus. `PrivacyEngine.make_private_with_epsilon`
  automatically computes the noise (sigma) needed to reach the requested
  `(epsilon, delta)` budget over the chosen number of epochs.
  `BatchMemoryManager` splits the logical batch into smaller physical
  batches to fit in memory despite the per-sample gradient computation.

Each script saves:
- a `.pth` file (weights + accuracy history + hyperparameters),
- a sibling `.json` file (same metadata, without the weights) — handy for
  quickly browsing all past runs without loading any tensors.

### 3. Checkpoint naming (`checkpoint.py`)

Convention:
```
{prefix}_eps{epsilon}_delta{delta}_epoch{epochs}_C{max_grad_norm}_seed{seed}_{timestamp}.pth
```
Example: `dp_resnet18_eps10_delta1e08_epoch10_C1.2_seed42_20260628-143012.pth`

The **timestamp** (`YYYYmmdd-HHMMSS`) is an addition relative to the
original notebooks: it guarantees an old run is never accidentally
overwritten, even when rerunning with exactly the same hyperparameters.
Since the format sorts lexicographically, `find_latest_checkpoint()`
easily retrieves "the latest baseline with seed=42" without having to
copy-paste a filename.

### 4. CKA analysis (`run_cka.py`)

Three subcommands, matching the analyses from notebook 2:

- **`compare`**: CKA between any two checkpoints (baseline vs DP, baseline
  vs baseline with a different seed, DP vs DP...) on the high-level layers
  (`layer1` through `layer4`). This is the most common comparison.
- **`multi-eps`**: compares a reference baseline against several DP
  checkpoints trained with different epsilons, to visualize how
  representation similarity evolves with the privacy budget.
- **`fine-grained`**: the "checkerboard" heatmap at a finer grain (conv1/
  conv2 sub-layers of each residual block), to visualize finer internal
  patterns than at the level of the 4 main stages.

Each subcommand saves a `.png` figure and the raw numeric values (`.json`
or `.npy`) in `results/`, with a timestamp.

### 5. Orchestrator (`run_pipeline.py`)

Chains `train_baseline.py` → `train_dp.py` → `run_cka.py compare`
automatically, reusing the most recent checkpoints. Handy for a full
"default" run, but the separate scripts remain recommended for day-to-day
use (see below) since you rarely need to rerun everything each time.

## Configuration

All configuration lives in a YAML file (see `configs/default.yaml`), and
every field can be overridden on the command line:

```bash
# Uses the full default config
python src/train_dp.py --config configs/default.yaml

# Same, but with a different epsilon just for this trial
python src/train_dp.py --config configs/default.yaml --epsilon 2

# Without any config file at all (Config dataclass defaults)
python src/train_baseline.py --epochs 5 --seed 43
```

## File checklist

Use this to double-check your local copy matches the expected layout —
21 files total (13 in `src/` including `__init__.py`, 2 in `configs/`,
3 `.gitkeep` placeholders, plus 3 at the repo root):

| Path | Purpose |
|---|---|
| `.gitignore` | ignores generated data/outputs, keeps `.gitkeep` placeholders |
| `README.md` | this file |
| `requirements.txt` | pinned-free dependency list for `uv pip install` |
| `configs/default.yaml` | default hyperparameters |
| `configs/eps_low.yaml` | example override (epsilon=2) |
| `src/__init__.py` | empty, marks `src/` as a package |
| `src/model.py` | shared DP-compatible ResNet18 |
| `src/data.py` | CIFAR-10 download + loading |
| `src/config.py` | `Config` dataclass, YAML loading, CLI overrides |
| `src/checkpoint.py` | checkpoint naming/saving/loading helpers |
| `src/training.py` | training loops (baseline + DP-SGD) + `evaluate` |
| `src/cka.py` | CKA math + activation hooks |
| `src/visualization.py` | all matplotlib plotting functions |
| `src/logging_utils.py` | console + file logging setup |
| `src/train_baseline.py` | CLI entry point: baseline training |
| `src/train_dp.py` | CLI entry point: DP-SGD training |
| `src/run_cka.py` | CLI entry point: CKA analysis (3 subcommands) |
| `src/run_pipeline.py` | CLI entry point: full orchestrator |
| `networks/.gitkeep` | keeps the (otherwise empty) checkpoints folder in git |
| `results/.gitkeep` | keeps the (otherwise empty) results folder in git |
| `logs/.gitkeep` | keeps the (otherwise empty) logs folder in git |

If `networks/`, `results/`, or `logs/` are simply missing rather than
empty, that's fine too — every script creates them on demand.

---

## Day-to-day workflow (personal use)

Typical order for a working session. The commands below assume the
virtual environment is activated (`source .venv/bin/activate`); if you'd
rather not activate it, prefix each one with `uv run` instead
(e.g. `uv run train_baseline.py ...`).

### A. First time / setting up a new epsilon to test

```bash
cd src

# 1. Train the baseline once (reusable across all epsilons)
python train_baseline.py --config ../configs/default.yaml

# 2. Train the DP model for the desired epsilon
python train_dp.py --config ../configs/default.yaml --epsilon 8

# 3. Compare the two (find the exact filenames with `ls ../networks/`,
#    or use find_latest_checkpoint from a Python shell if you prefer)
python run_cka.py compare \
    --ckpt-a ../networks/baseline_resnet18_..._seed42_<timestamp>.pth \
    --ckpt-b ../networks/dp_resnet18_eps8_..._seed42_<timestamp>.pth \
