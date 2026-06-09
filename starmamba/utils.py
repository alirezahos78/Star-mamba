import os
import random
import sys
import gc

import numpy as np
import torch


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Logger:
    """Mirror stdout to a log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        log_dir = os.path.dirname(filename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def is_cuda_device(device):
    return torch.device(device).type == "cuda"


def move_batch_to_device(x, y, device):
    non_blocking = is_cuda_device(device)
    return x.to(device, non_blocking=non_blocking), y.to(device, non_blocking=non_blocking)


def release_cuda_memory(device, empty_cache=False):
    gc.collect()
    if empty_cache and is_cuda_device(device):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def save_checkpoint(model, acc, epoch, filename_prefix, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, f"{filename_prefix}_acc{acc:.2f}_ep{epoch}.pth")
    torch.save(model.state_dict(), filename)
    return filename


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mixup_data(x, y, alpha=1.0, device="cuda"):
    lam = random.betavariate(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = move_batch_to_device(x, y, device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            del x, y, outputs, predicted
    return 100.0 * correct / total
