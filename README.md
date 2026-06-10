# Star-Mamba Vision Experiments

PyTorch code for **Star-Mamba**, a patch-based vision classifier that combines a global Mamba operator with directional local Mamba scans over image patches.

This directory is intended to be the GitHub repository root. It contains the Star-Mamba training code, experiment presets for CIFAR-10, CIFAR-100, Fashion-MNIST, Day2Night, Tiny ImageNet, Tiny ImageNet ablations, patch-size ablations, preserved training logs, baseline references, and a Vim-vs-Star-Mamba saliency comparison script for paper figures.

## Repository Layout

```text
.
├── train.py                         # Main experiment entrypoint
├── starmamba/                       # Reusable training/model package
│   ├── models.py                    # VisionModel, StarMambaBlock, ablation blocks
│   ├── data.py                      # Dataset loaders and Tiny ImageNet preparation
│   ├── augmentations.py             # Cutout and repeated-augmentation sampler
│   ├── training.py                  # Shared train/eval loop and configs
│   ├── experiments.py               # Named experiment presets
│   └── utils.py                     # Logging, checkpoints, mixup, metrics, seed setup
├── scripts/
│   ├── compare_vim_starmamba_gradcam.py
│   └── run_patch_size_ablation.sh
├── baselines/                       # Lightweight EfficientVMamba and Vim artifacts
├── logs/                            # Preserved logs from completed runs
├── checkpoints/                     # Local model checkpoints, not for GitHub commits
├── data/                            # Local datasets, not for GitHub commits
├── BASELINES.md                     # Baseline metadata and comparison notes
├── RESULTS.md                       # Reported results extracted from logs
├── requirements.txt                 # Runtime dependencies
├── pyproject.toml                   # Formatting/lint configuration
└── .github/workflows/syntax.yml     # Lightweight syntax check workflow
```

Generated datasets, checkpoints, maps, and runtime outputs should stay out of normal Git commits. Publish large `.pth` files separately with GitHub Releases, Hugging Face, Zenodo, or another artifact store.

## GitHub Upload Notes

Use this directory as the repository root when publishing. The project does not require any sibling folders for normal training.

Do not commit generated datasets, local checkpoints, virtual environments, or large runtime outputs. The included `.gitignore` excludes `data/`, `checkpoints/`, archive files, Python caches, and common experiment-output directories. Preserved `logs/` and baseline logs are kept as lightweight experiment evidence.

## Model Summary

Each full Star-Mamba block applies:

- `LayerNorm` over patch tokens.
- one global `Mamba` pass over the full sequence.
- four directional local Mamba passes: north, south, west, and east.
- SoftPlus-gated aggregation of the directional outputs.
- residual connection and MLP refinement.

The vision model uses a convolutional patch embedding layer, learned positional embeddings, stacked Star-Mamba blocks, mean token pooling, and a linear classification head.

## Setup

Create an environment with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`mamba-ssm` is CUDA-sensitive. If installation fails, install PyTorch for your CUDA version first, then install `mamba-ssm` following its upstream installation notes. The pinned stack in `requirements.txt` is:

- `torch==2.3.1`
- `torchvision==0.18.1`
- `mamba-ssm==2.2.2`

## Running Experiments

Run commands from the project root:

```bash
python train.py cifar10
python train.py cifar100
python train.py fashionmnist
python train.py day2night
python train.py tiny-imagenet
```

Tiny ImageNet ablations:

```bash
python train.py tiny-imagenet-no-global
python train.py tiny-imagenet-no-ew
python train.py tiny-imagenet-no-ns
python train.py tiny-imagenet-only-global
```

Patch-size ablations:

```bash
python train.py cifar10-patch4
python train.py cifar10-patch8
python train.py cifar100-patch4
python train.py cifar100-patch8
python train.py fashionmnist-patch4
python train.py fashionmnist-patch8
python train.py tiny-imagenet-patch8
python train.py tiny-imagenet-patch16
```

Tiny ImageNet uses 64x64 images, so the patch sizes are doubled from the CIFAR/Fashion-MNIST ablation sizes: patch 4 becomes 8, and patch 8 becomes 16. To run the full patch-size ablation sequence with one command:

```bash
scripts/run_patch_size_ablation.sh
```

The runner skips experiments whose log already contains `Training finished.`. If an incomplete log already exists, it is moved to a timestamped backup before rerunning that experiment. If one experiment crashes, the runner records the failure and continues to the next experiment. Set `RERUN_COMPLETED=1` if you intentionally want to rerun everything.

Any training override accepted by `train.py` can be passed through the script:

```bash
scripts/run_patch_size_ablation.sh --epochs 50 --batch-size 64
```

Set `PYTHON=/path/to/python` before the command if your environment does not expose `python3`.

For long CUDA runs with native extension crashes, you can trade a little speed for cleaner epoch boundaries:

```bash
scripts/run_patch_size_ablation.sh --empty-cache-each-epoch
```

Useful overrides:

```bash
python train.py tiny-imagenet-no-ew --epochs 50
python train.py cifar100 --batch-size 64 --workers 0
python train.py cifar10-patch4 --optimizer adamw
```

The default optimizer is `adamw-safe`, which uses AdamW with PyTorch fused/foreach optimizer paths disabled. Use `--optimizer adamw` only if you want PyTorch's default AdamW implementation.

The default DataLoader worker count is `0` for stability on CUDA systems where worker subprocesses may segfault. If you see an error like `DataLoader worker ... is killed by signal: Segmentation fault`, rerun with `--workers 0`. Increase workers only on systems where the DataLoader is stable.

## Logs And Checkpoints

`train.py` does **not** create a log file unless `--log-file` is provided. Without this flag, training output only prints to the terminal.

```bash
python train.py tiny-imagenet --log-file logs/log_tiny_imagenet_rerun.txt
```

The log file path is exactly the value passed to `--log-file`. If the file already exists, it is overwritten.

Best checkpoints are saved automatically under `checkpoints/` with names like:

```text
checkpoints/best_starmamba_tiny_imagenet_acc67.66_ep287.pth
```

The prefix comes from `starmamba/experiments.py`, and the accuracy/epoch are added by `starmamba/utils.py`.

## Data

CIFAR-10, CIFAR-100, and Fashion-MNIST are downloaded through `torchvision`.

Day2Night is expected at `data/day2night/` with the existing split folders `trainA/`, `trainB/`, `testA/`, and `testB/`. The `day2night` experiment resizes images from 256x256 to 64x64, uses patch size 4, has two classes, and disables train-time augmentation including mixup.

Tiny ImageNet is downloaded and reorganized by `starmamba.data.prepare_tiny_imagenet()` when `data/tiny-imagenet-200/` is missing. The Tiny ImageNet experiment uses the validation folder as the evaluation split.

## Analysis

The current `scripts/` directory contains a Vim-vs-Star-Mamba Grad-CAM comparison utility. Mamba models do not expose transformer attention weights, so this script uses patch Grad-CAM saliency maps as the comparable visualization.

Compare a small contiguous set of Tiny ImageNet validation images:

```bash
python scripts/compare_vim_starmamba_gradcam.py --vim-root /path/to/vim/root --index 0 --count 8 --target-source true
```

Compare the first 10 images from every Tiny ImageNet class, 2000 images total:

```bash
python scripts/compare_vim_starmamba_gradcam.py --vim-root /path/to/vim/root --samples-per-class 10 --target-source true
```

The Vim source tree and Vim checkpoints are not included in this repository. To run the comparison, pass `--vim-root` pointing to an external Vim checkout that contains `Vim/vim/models_mamba.py` and, unless `--vim-checkpoint` is given, `checkpoints_tinyimagenet/`.

By default, the script:

- auto-selects the highest-accuracy Vim Tiny ImageNet checkpoint,
- auto-selects the highest-accuracy Star-Mamba Tiny ImageNet checkpoint,
- uses this repository's `data/` directory as the Tiny ImageNet data root unless `--data-root` is provided,
- saves outputs to `comparison_maps/vim_vs_starmamba_tinyimagenet/`.

Each sample gets individual heatmaps/overlays, a side-by-side comparison image, raw `.npz` maps, and `comparison_metrics.csv` with Pearson correlation, cosine similarity, and top-20-percent saliency IoU.

## Results And Baselines

See [RESULTS.md](RESULTS.md) for reported Star-Mamba accuracies extracted from preserved logs.

See [BASELINES.md](BASELINES.md) for EfficientVMamba and Vim baseline metadata, local artifact paths, and comparison tables.

## Reproducibility Notes

- Training sets Python, NumPy, and PyTorch seeds.
- CUDA deterministic mode is enabled where supported.
- Results can still vary across GPU, CUDA, PyTorch, and `mamba-ssm` versions.
- Repeated augmentation uses repeated samples per batch; with `batch_size=128` and `repeats=5`, the actual yielded batch has `25 * 5 = 125` samples.
- Preserved logs are historical run artifacts. New runs should use new `--log-file` names unless you intentionally want to overwrite an old log.

