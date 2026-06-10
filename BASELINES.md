# Baselines

This file documents the baseline evidence used for comparison with Star-Mamba. The baseline folders in this repository are lightweight artifacts only: local run scripts and preserved logs. They are not full copies of the upstream projects.

For paper or release use, cite the original papers/repositories, record the exact upstream commit, and keep large datasets/checkpoints outside this repository.

## External Sources

| Baseline | Source status | Upstream repository | Commit used |
| --- | --- | --- | --- |
| EfficientVMamba | external upstream project; lightweight scripts/logs only are preserved here | `https://github.com/TerryPei/EfficientVMamba` | `0bc5ee288b402d648641f5494b73e9d152e0c62b` |
| Vim | external upstream project; lightweight scripts/logs only are preserved here | `https://github.com/hustvl/Vim.git` | `dd0358ad1e42701f22afbefa0717cc8825cf9f45` |

## Local Baseline Artifacts

EfficientVMamba logs:

- `baselines/efficient_vmamba/logs/log_efficient_vmamba_cifar10_scratch.txt`
- `baselines/efficient_vmamba/logs/log_efficient_vmamba_cifar100_scratch.txt`
- `baselines/efficient_vmamba/logs/log_efficient_vmamba_fashion_mnist_scratch.txt`
- `baselines/efficient_vmamba/logs/log_efficient_vmamba_tiny_imagenet_scratch.txt`

EfficientVMamba scripts:

- `baselines/efficient_vmamba/scripts/train_efficient_vmamba_cifar10_scratch.py`
- `baselines/efficient_vmamba/scripts/train_efficient_vmamba_cifar100_scratch.py`
- `baselines/efficient_vmamba/scripts/train_efficient_vmamba_fashion_mnist_scratch.py`
- `baselines/efficient_vmamba/scripts/train_efficient_vmamba_tiny_imagenet_scratch.py`

Vim logs:

- `baselines/vim/logs/log_vim_cifar10_scratch.txt`
- `baselines/vim/logs/log_vim_cifar100_scratch.txt`
- `baselines/vim/logs/log_vim_fashionmnist_scratch.txt`
- `baselines/vim/logs/log_vim_tinyimagenet_scratch.txt`

Vim scripts:

- `baselines/vim/scripts/train_vim_cifar10_scratch.py`
- `baselines/vim/scripts/train_vim_cifar100_scratch.py`
- `baselines/vim/scripts/train_vim_fashionmnist_scratch.py`
- `baselines/vim/scripts/train_vim_tinyimagenet_scratch.py`

## Star-Mamba Logs Used For Comparison

The current Star-Mamba logs are:

- `logs/log_cifar10.txt`
- `logs/log_cifar100.txt`
- `logs/log_fashionmnist.txt`
- `logs/log_tiny_imagenet.txt`
- `logs/log_tiny_imagenet_ablation_no_global.txt`
- `logs/log_tiny_imagenet_ablation_no_ew.txt`
- `logs/log_tiny_imagenet_ablation_no_ns.txt`
- `logs/log_tiny_imagenet_ablation_only_global.txt`

There is no default log-file name in `train.py`; these names are preserved artifacts from completed runs. New logs are written only when `--log-file <path>` is passed.

## Reported Comparison Results

The following values are extracted from the local logs above.

| Model | CIFAR-10 | CIFAR-100 | Fashion-MNIST | Tiny ImageNet |
| --- | ---: | ---: | ---: | ---: |
| EfficientVMamba, scratch | 93.43% | 74.89% | 94.85% | 60.39% |
| Vim, scratch | 95.57% | 71.26% | 96.07% | 54.56% |
| Star-Mamba | 96.81% | 80.93% | 96.15% | 67.76%  |

The Star-Mamba Tiny ImageNet value comes from `logs/log_tiny_imagenet.txt`, which reports best accuracy `67.76%`.

## Tiny ImageNet Ablations

The ablation values below are extracted from the current Star-Mamba ablation logs.

| Variant | Log file | Tiny ImageNet accuracy |
| --- | --- | ---: |
| full local/global, strict validation | `logs/log_tiny_imagenet_strict.txt` | 67.76% |
| no global path | `logs/log_tiny_imagenet_ablation_no_global.txt` | 65.76% |
| no east/west local scans | `logs/log_tiny_imagenet_ablation_no_ew.txt` | 64.02% |
| no north/south local scans | `logs/log_tiny_imagenet_ablation_no_ns.txt` | 61.38% |
| only global path | `logs/log_tiny_imagenet_ablation_only_global.txt` | 57.09% |

## Vim Vs Star-Mamba Saliency Maps

The current repository includes a paper-figure utility for comparing Vim and Star-Mamba patch Grad-CAM maps:

```bash
python scripts/compare_vim_starmamba_gradcam.py --vim-root /path/to/vim/root --samples-per-class 10 --target-source true
```

This uses:

- Vim root: external Vim checkout passed with `--vim-root`
- Vim checkpoints: highest-accuracy checkpoint under `<vim-root>/checkpoints_tinyimagenet/`, unless `--vim-checkpoint` is provided
- Star-Mamba checkpoints: `checkpoints/`
- default output: `comparison_maps/vim_vs_starmamba_tinyimagenet/`

Because Mamba models do not expose transformer attention matrices, these visualizations should be described as patch Grad-CAM saliency maps rather than literal attention maps.

## Reproducibility Notes

- Full baseline source trees are not included because they belong to external projects.
- The comparison table should be accompanied by citations to the original EfficientVMamba and Vim papers/repositories.
- If publishing this repository, do not include full external baseline source trees unless their licenses allow redistribution and attribution is preserved.
- If baseline scripts were modified for local training, document those modifications in the corresponding baseline folder or publish a separate fork/patch.
- Large baseline checkpoints, generated maps, and downloaded datasets should not be committed to this project repository.
