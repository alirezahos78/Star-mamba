# Star-Mamba: Efficient Visual State Space Modeling with Independent Axial Scanning

PyTorch implementation of **Star-Mamba**, a pure state space model (SSM) architecture for image classification that recovers spatial locality through scan ordering, without convolution or attention.

This repository contains the official code for the paper *"Star-Mamba: Efficient Visual State Space Modeling with Independent Axial Scanning."* It includes the Star-Mamba training code, experiment presets for CIFAR-10, CIFAR-100, Fashion-MNIST, Tiny-ImageNet, the Day2Night probe, the component and patch-size ablations, preserved training logs, baseline references, and a Grad-CAM comparison script used for the paper figures.

## Overview

Adapting one-dimensional SSMs such as Mamba to two-dimensional vision requires flattening an image into a sequence, which separates spatially adjacent pixels. We formally characterize the resulting loss of locality, which we term **global dominance bias**, and show that under raster scanning the influence between vertically adjacent pixels decays exponentially with the image width.

**Independent Axial Star Scanning (IASS)** addresses this by processing each row and each column as an autonomous sequence, keeping spatially adjacent pixels close within the scan and reducing the decay so that it no longer depends on the image width. The four directional outputs (north, south, west, east) are combined by an interpretable **Expansive Gating** mechanism with learnable per-channel weights, alongside a parallel global Mamba pass that captures long-range context.

## Repository Layout

```
.
├── train.py                         # Main experiment entrypoint
├── starmamba/                       # Training/model package
│   ├── models.py                    # VisionModel, StarMambaBlock, ablation blocks
│   ├── data.py                      # Dataset loaders and Tiny-ImageNet preparation
│   ├── augmentations.py             # Cutout and repeated-augmentation sampler
│   ├── training.py                  # Shared train/eval loop and configs
│   ├── experiments.py               # Named experiment presets
│   └── utils.py                     # Logging, checkpoints, mixup, metrics, seeding
├── scripts/
│   ├── compare_vim_starmamba_gradcam.py
│   └── run_patch_size_ablation.sh
├── baselines/                       # Lightweight EfficientVMamba and Vim artifacts
├── logs/                            # Preserved logs from completed runs
├── BASELINES.md                     # Baseline metadata and comparison notes
├── RESULTS.md                       # Reported results extracted from logs
├── requirements.txt                 # Runtime dependencies
├── pyproject.toml                   # Formatting/lint configuration
└── .github/workflows/syntax.yml     # Lightweight syntax check workflow
```

Generated datasets, checkpoints, and runtime outputs are excluded from version control via `.gitignore`. Large `.pth` checkpoints should be published separately through GitHub Releases, Hugging Face, or Zenodo.

## Model Summary

Each Star-Mamba block applies:

- `LayerNorm` over the patch tokens.
- A **Global Context Path**: a single Mamba pass over the full token sequence.
- A **Local Axial Path (IASS)**: four directional Mamba passes (north, south, west, east) that scan rows and columns as independent sequences.
- **Expansive Gating**: SoftPlus-based, learnable per-channel weights that combine the four directional outputs.
- A residual connection and an MLP refinement block.

The full model uses a convolutional patch embedding, learned positional embeddings, a stack of Star-Mamba blocks, mean token pooling, and a linear classification head. The reported configuration has 3.4M parameters.

## Setup

Create an environment with Python 3.10 or newer:

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`mamba-ssm` is CUDA-sensitive. If installation fails, install PyTorch for your CUDA version first, then install `mamba-ssm` following its upstream notes. The pinned stack in `requirements.txt` is:

- `torch==2.3.1`
- `torchvision==0.18.1`
- `mamba-ssm==2.2.2`

## Running Experiments

Run commands from the project root:

```
python train.py cifar10
python train.py cifar100
python train.py fashionmnist
python train.py tiny-imagenet
python train.py day2night
```

**Component ablations** (Tiny-ImageNet):

```
python train.py tiny-imagenet-no-global     # remove the Global Context Path
python train.py tiny-imagenet-no-ew          # remove the East-West scans
python train.py tiny-imagenet-no-ns          # remove the North-South scans
python train.py tiny-imagenet-only-global    # global path only
```

**Patch-size ablations:**

```
python train.py cifar10-patch4
python train.py cifar10-patch8
python train.py cifar100-patch4
python train.py cifar100-patch8
python train.py fashionmnist-patch4
python train.py fashionmnist-patch8
python train.py tiny-imagenet-patch8
python train.py tiny-imagenet-patch16
```

Tiny-ImageNet uses 64x64 images, so its patch sizes are doubled relative to the CIFAR/Fashion-MNIST ablations (patch 4 becomes 8, patch 8 becomes 16). To run the full patch-size sweep in one command:

```
scripts/run_patch_size_ablation.sh
```

The runner skips experiments whose log already contains `Training finished.`, backs up incomplete logs before rerunning, and continues past individual failures. Set `RERUN_COMPLETED=1` to force a full rerun. Any `train.py` override can be passed through:

```
scripts/run_patch_size_ablation.sh --epochs 50 --batch-size 64
```

Useful overrides:

```
python train.py tiny-imagenet-no-ew --epochs 50
python train.py cifar100 --batch-size 64 --workers 0
python train.py cifar10-patch4 --optimizer adamw
```

The default optimizer is `adamw-safe` (AdamW with fused/foreach paths disabled for stability). The default DataLoader worker count is `0`; if you see a `DataLoader worker ... Segmentation fault`, keep `--workers 0`.

## Logs and Checkpoints

`train.py` only writes a log file when `--log-file` is provided:

```
python train.py tiny-imagenet --log-file logs/log_tiny_imagenet.txt
```

Best checkpoints are saved under `checkpoints/` with names like:

```
checkpoints/best_starmamba_tiny_imagenet_acc67.66_ep287.pth
```

## Data

CIFAR-10, CIFAR-100, and Fashion-MNIST are downloaded through `torchvision`.

Tiny-ImageNet is downloaded and reorganized by `starmamba.data.prepare_tiny_imagenet()` when `data/tiny-imagenet-200/` is missing; the validation folder is used as the evaluation split.

Day2Night is used in the paper as a **controlled probe** for a globally-cued task. It is expected at `data/day2night/` with split folders `trainA/`, `trainB/`, `testA/`, `testB/`. The experiment resizes images to 64x64, uses patch size 4, has two classes, and disables train-time augmentation.

## Analysis

`scripts/compare_vim_starmamba_gradcam.py` produces the Grad-CAM saliency comparison between Vim and Star-Mamba used in the paper. Since Mamba models do not expose attention weights, patch Grad-CAM saliency is used as the comparable visualization.

```
python scripts/compare_vim_starmamba_gradcam.py --vim-root /path/to/vim/root --index 0 --count 8 --target-source true
```

The Vim source tree and checkpoints are not included; pass `--vim-root` pointing to an external Vim checkout. Outputs (heatmaps, side-by-side overlays, raw `.npz` maps, and `comparison_metrics.csv` with Pearson correlation, cosine similarity, and top-20% saliency IoU) are saved to `comparison_maps/`.

## Results and Baselines

See [RESULTS.md](RESULTS.md) for reported Star-Mamba accuracies extracted from the preserved logs, and [BASELINES.md](BASELINES.md) for EfficientVMamba and Vim baseline metadata and comparison tables.

## Reproducibility Notes

- Training seeds Python, NumPy, and PyTorch.
- CUDA deterministic mode is enabled where supported.
- Results can still vary across GPU, CUDA, PyTorch, and `mamba-ssm` versions.
- Preserved logs are historical run artifacts; new runs should use new `--log-file` names.

## License

This project is released under the [MIT License](LICENSE).
