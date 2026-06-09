# Results

Reported accuracies below are extracted from the preserved files in `logs/`.

## Main Results

| Experiment | Log file | Best reported accuracy | Best epoch |
| --- | --- | ---: | ---: |
| CIFAR-10 | `logs/log_cifar10.txt` | 96.81% test | 285 |
| CIFAR-100 | `logs/log_cifar100.txt` | 80.93% test | 290 |
| Fashion-MNIST | `logs/log_fashionmnist.txt` | 96.15% test | 269 |
| Tiny ImageNet, strict validation | `logs/log_tiny_imagenet.txt` | 67.76% validation | 297 |



## Tiny ImageNet Ablations

| Variant | Log file | Best reported accuracy | Best epoch |
| --- | --- | ---: | ---: |
| full local/global | `logs/log_tiny_imagenet.txt` | 67.76% validation | 297 |
| no global path | `logs/log_tiny_imagenet_ablation_no_global.txt` | 65.76% test | 269 |
| no east/west local scans | `logs/log_tiny_imagenet_ablation_no_ew.txt` | 64.02% test | 285 |
| no north/south local scans | `logs/log_tiny_imagenet_ablation_no_ns.txt` | 61.38% test | 299 |
| only global path | `logs/log_tiny_imagenet_ablation_only_global.txt` | 57.09% test | 293 |

## Reading The Ablations

- The full local/global model is strongest in the current Tiny ImageNet logs: `67.76%` validation.
- Removing the global path drops to `65.76%`.
- Removing east/west local scans drops to `64.02%`.
- Removing north/south local scans drops to `61.38%`.
- Using only the global path drops to `57.09%`.

These runs suggest the local directional paths are important, with north/south scans contributing strongly in the current Tiny ImageNet ablations.


