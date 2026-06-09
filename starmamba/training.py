import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from .models import StarMambaBlock, VisionModel
from .utils import count_parameters, evaluate, mixup_criterion, mixup_data, move_batch_to_device, release_cuda_memory, save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 300
    lr_max: float = 1e-3
    weight_decay: float = 0.05
    optimizer: str = "adamw-safe"
    mixup_alpha: float = 1.0
    label_smoothing: float = 0.1
    warmup_epochs: int = 10
    grad_clip_norm: float = 1.0
    checkpoint_prefix: str = "best_starmamba"
    checkpoint_dir: str = "checkpoints"
    empty_cache_each_epoch: bool = False


@dataclass
class ModelConfig:
    img_size: int = 32
    patch_size: int = 2
    in_channels: int = 3
    dim: int = 112
    depth: int = 6
    num_classes: int = 10
    block_type: type = StarMambaBlock


def build_model(config: ModelConfig):
    return VisionModel(
        block_type=config.block_type,
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        dim=config.dim,
        depth=config.depth,
        num_classes=config.num_classes,
    )


def build_cosine_warmup_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs):
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        t_max = max(1, total_steps - warmup_steps)
        t_cur = step - warmup_steps
        return 0.5 * (1.0 + math.cos(math.pi * t_cur / t_max))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_optimizer(model, config):
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adamw-safe":
        return optim.AdamW(
            params,
            lr=config.lr_max,
            weight_decay=config.weight_decay,
            foreach=False,
            fused=False,
        )
    if config.optimizer == "adamw":
        return optim.AdamW(params, lr=config.lr_max, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer '{config.optimizer}'. Use 'adamw-safe' or 'adamw'.")


def format_flops(flops):
    if flops >= 1e12:
        return f"{flops / 1e12:.2f}T"
    if flops >= 1e9:
        return f"{flops / 1e9:.2f}G"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f}M"
    if flops >= 1e3:
        return f"{flops / 1e3:.2f}K"
    return f"{flops:.0f}"


def model_input_shape(model):
    patch_size = model.patch_embed.kernel_size[0]
    img_h = model.h * patch_size
    img_w = model.w * patch_size
    return 1, model.patch_embed.in_channels, img_h, img_w


def flop_count_summary(model):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return "unavailable (fvcore is not installed)"

    was_training = model.training
    model.eval()
    try:
        param = next(model.parameters())
        inputs = torch.zeros(model_input_shape(model), device=param.device, dtype=param.dtype)
        with torch.no_grad():
            flops = FlopCountAnalysis(model, inputs).unsupported_ops_warnings(False)
            total = flops.total()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"
    finally:
        model.train(was_training)

    return f"{format_flops(total)} ({total:.0f})"


def directional_weight_summary(model):
    """Return readable mean directional gate weights for the first block."""

    if not hasattr(model, "layers") or not model.layers:
        return ""

    block = model.layers[0]
    weights = []
    for name in ("n", "s", "w", "e"):
        attr = f"lam_{name}"
        if hasattr(block, attr):
            value = torch.nn.functional.softplus(getattr(block, attr)).mean().item()
            weights.append(f"{name.upper()}={value:.4f}")
    return " | ".join(weights)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, mixup_alpha=1.0, grad_clip_norm=1.0):
    model.train()
    train_loss = 0.0
    train_correct = 0.0
    total_samples = 0

    for x, y in loader:
        x, y = move_batch_to_device(x, y, device)
        optimizer.zero_grad(set_to_none=True)

        if mixup_alpha and mixup_alpha > 0:
            x_in, y_a, y_b, lam = mixup_data(x, y, mixup_alpha, device)
            out = model(x_in)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
        else:
            out = model(x)
            loss = criterion(out, y)
            y_a, y_b, lam = y, y, 1.0

        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        try:
            optimizer.step()
        except SystemError as exc:
            raise RuntimeError(
                "Optimizer step failed inside PyTorch. This is commonly caused by a "
                "PyTorch/CUDA/mamba-ssm version mismatch. Try the default "
                "'adamw-safe' optimizer, rerun with '--workers 0' if a DataLoader "
                "worker crashed, or install the repo-pinned torch/mamba-ssm versions."
            ) from exc
        scheduler.step()

        train_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        _, predicted = out.max(1)
        train_correct += (predicted.eq(y_a) * lam + predicted.eq(y_b) * (1 - lam)).sum().item()
        del x, y, out, loss, predicted, y_a, y_b
        if mixup_alpha and mixup_alpha > 0:
            del x_in

    return {
        "loss": train_loss / total_samples,
        "acc": 100.0 * train_correct / total_samples,
    }


def train_classifier(
    model,
    train_loader,
    val_loader,
    device,
    config=None,
    metric_name="Test Acc",
    print_fn=print,
):
    """Train a classifier and save the best checkpoint.

    The function intentionally mirrors the original experiment scripts while
    making the loop reusable across CIFAR, FashionMNIST, and Tiny-ImageNet.
    """

    config = config or TrainConfig()
    model = model.to(device)

    print_fn(f"Model Parameters: {count_parameters(model) / 1e6:.2f}M")
    print_fn(f"Model FLOPs: {flop_count_summary(model)}")

    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scheduler = build_cosine_warmup_scheduler(
        optimizer,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=config.warmup_epochs,
    )

    best_acc = 0.0
    best_checkpoint = None

    for epoch in range(config.epochs):
        start = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            mixup_alpha=config.mixup_alpha,
            grad_clip_norm=config.grad_clip_norm,
        )
        release_cuda_memory(device)
        val_acc = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        weights = directional_weight_summary(model)
        elapsed = time.time() - start

        print_fn(
            f"Ep {epoch + 1:03d} | Train Acc={train_metrics['acc']:.2f}% | "
            f"{metric_name}={val_acc:.2f}% | Loss={train_metrics['loss']:.3f} | "
            f"LR={current_lr:.6f} | Time={elapsed:.1f}s"
        )
        if weights:
            print_fn(f"      Weights (Mean): {weights}")

        if val_acc > best_acc:
            print_fn(f"   New Best! ({best_acc:.2f}% -> {val_acc:.2f}%) Saving checkpoint...")
            best_acc = val_acc
            best_checkpoint = save_checkpoint(
                model,
                best_acc,
                epoch + 1,
                config.checkpoint_prefix,
                checkpoint_dir=config.checkpoint_dir,
            )
            print_fn(f"Checkpoint saved: {best_checkpoint}")
        if config.empty_cache_each_epoch:
            release_cuda_memory(device, empty_cache=True)

    return {
        "best_acc": best_acc,
        "best_checkpoint": best_checkpoint,
    }
