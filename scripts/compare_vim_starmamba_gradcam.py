import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from starmamba.data import IMAGENET_STATS, prepare_tiny_imagenet  # noqa: E402
from starmamba.experiments import get_experiment  # noqa: E402
from starmamba.training import build_model  # noqa: E402

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare patch Grad-CAM maps for Vim and Star-Mamba on the same "
            "Tiny ImageNet validation images."
        )
    )
    parser.add_argument("--vim-root", default=None, help="Path to an external Vim checkout root. Required.")
    parser.add_argument("--vim-checkpoint", default=None)
    parser.add_argument("--star-experiment", default="tiny-imagenet")
    parser.add_argument("--star-checkpoint", default=None)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="If set, compare the first N images from every class and ignore --index/--count.",
    )
    parser.add_argument(
        "--target-source",
        default="true",
        choices=["true", "vim-pred", "star-pred"],
        help="Class used for both Grad-CAM calls.",
    )
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "comparison_maps", "vim_vs_starmamba_tinyimagenet"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--metric-size", type=int, default=64)
    return parser.parse_args()


def checkpoint_score(path):
    name = os.path.basename(path)
    match = re.search(r"(?:^|_)acc([0-9]+(?:\.[0-9]+)?)(?:_ep([0-9]+))?", name)
    if match is None:
        return (float("-inf"), -1, os.path.getmtime(path))
    epoch = -1 if match.group(2) is None else int(match.group(2))
    return (float(match.group(1)), epoch, os.path.getmtime(path))


def find_vim_checkpoint(vim_root):
    pattern = os.path.join(vim_root, "checkpoints_tinyimagenet", "vim_tinyimagenet_best_acc*.pth")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No Vim Tiny ImageNet checkpoint found with pattern: {pattern}")
    return sorted(matches, key=checkpoint_score, reverse=True)[0]


def find_star_checkpoint(experiment):
    exp = get_experiment(experiment)
    prefix = exp["train"].checkpoint_prefix
    pattern = os.path.join(PROJECT_ROOT, "checkpoints", f"{prefix}_*.pth")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No Star-Mamba checkpoint found with pattern: {pattern}")
    return sorted(matches, key=checkpoint_score, reverse=True)[0]


def add_vim_to_path(vim_root):
    vim_module_path = os.path.join(vim_root, "Vim", "vim")
    if not os.path.isdir(vim_module_path):
        raise FileNotFoundError(f"Vim model directory not found: {vim_module_path}")
    if vim_module_path not in sys.path:
        sys.path.insert(0, vim_module_path)


def load_vim_model(vim_root, checkpoint_path, device):
    add_vim_to_path(vim_root)
    from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

    model = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
        pretrained=False,
        num_classes=200,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_star_model(experiment, checkpoint_path, device):
    exp = get_experiment(experiment)
    model = build_model(exp["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


class DualTransformTinyImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, vim_transform, star_transform):
        super().__init__(root=root, transform=None)
        self.vim_transform = vim_transform
        self.star_transform = star_transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path).convert("RGB")
        return {
            "path": path,
            "target": target,
            "raw": image,
            "vim": self.vim_transform(image),
            "star": self.star_transform(image),
        }


def build_dataset(data_root, split):
    mean, std = IMAGENET_STATS
    tiny_dir = prepare_tiny_imagenet(data_root)
    folder = "train" if split == "train" else "val"
    vim_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    star_transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return DualTransformTinyImageNet(os.path.join(tiny_dir, folder), vim_transform, star_transform)


def normalize_cam(cam):
    cam = cam.astype(np.float32)
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam


def cam_to_image(cam, image_size):
    cam = normalize_cam(cam)
    gray = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
    gray = gray.resize(image_size, resample=RESAMPLE_BILINEAR)
    return np.asarray(gray).astype(np.float32) / 255.0


def colorize_cam(cam, image_size):
    cam = cam_to_image(cam, image_size)
    heatmap = np.zeros((*cam.shape, 3), dtype=np.float32)
    heatmap[..., 0] = np.clip(1.5 * cam, 0, 1)
    heatmap[..., 1] = np.clip(1.5 * (1.0 - np.abs(cam - 0.55) * 2.0), 0, 1)
    heatmap[..., 2] = np.clip(1.5 * (1.0 - cam), 0, 1)
    return Image.fromarray((heatmap * 255).astype(np.uint8))


def overlay_heatmap(image, heatmap, alpha):
    image_np = np.asarray(image).astype(np.float32)
    heatmap_np = np.asarray(heatmap).astype(np.float32)
    overlay = (1.0 - alpha) * image_np + alpha * heatmap_np
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


class PatchGradCAM:
    def __init__(self, model, target_module, grid_size=None):
        self.model = model
        self.grid_size = grid_size
        self.activations = None
        self.handle = target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        if output.requires_grad:
            output.retain_grad()

    def remove(self):
        self.handle.remove()

    def _grid_size(self, num_tokens):
        if self.grid_size is not None:
            return self.grid_size
        side = int(round(num_tokens ** 0.5))
        if side * side != num_tokens:
            raise ValueError(f"Cannot infer square patch grid from {num_tokens} tokens.")
        return side, side

    def _cam_from_activations(self):
        acts = self.activations
        grads = self.activations.grad
        if acts.dim() == 4:
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1)
            return torch.relu(cam)[0].detach().cpu().numpy()
        if acts.dim() == 3:
            weights = grads.mean(dim=1, keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=2))[0]
            h, w = self._grid_size(cam.numel())
            return cam.reshape(h, w).detach().cpu().numpy()
        raise ValueError(f"Unsupported activation shape: {tuple(acts.shape)}")

    def __call__(self, image, target_class):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        pred = int(logits.argmax(dim=1).item())
        prob = float(torch.softmax(logits.detach(), dim=1)[0, pred].item())
        logits[:, target_class].sum().backward()
        return self._cam_from_activations(), pred, prob


def compare_maps(vim_cam, star_cam, metric_size):
    vim = cam_to_image(vim_cam, (metric_size, metric_size)).reshape(-1)
    star = cam_to_image(star_cam, (metric_size, metric_size)).reshape(-1)
    vim_centered = vim - vim.mean()
    star_centered = star - star.mean()
    corr = float((vim_centered * star_centered).sum() / ((np.linalg.norm(vim_centered) * np.linalg.norm(star_centered)) + 1e-8))
    cosine = float((vim * star).sum() / ((np.linalg.norm(vim) * np.linalg.norm(star)) + 1e-8))
    vim_top = vim >= np.quantile(vim, 0.8)
    star_top = star >= np.quantile(star, 0.8)
    intersection = np.logical_and(vim_top, star_top).sum()
    union = np.logical_or(vim_top, star_top).sum()
    top20_iou = float(intersection / max(1, union))
    return corr, cosine, top20_iou


def label_image(image, label):
    pad = 24
    output = Image.new("RGB", (image.width, image.height + pad), "white")
    output.paste(image, (0, pad))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    draw.text((4, 6), label, fill=(0, 0, 0), font=font)
    return output


def make_contact_sheet(columns):
    labeled = [label_image(image, label) for label, image in columns]
    width = sum(image.width for image in labeled)
    height = max(image.height for image in labeled)
    sheet = Image.new("RGB", (width, height), "white")
    x = 0
    for image in labeled:
        sheet.paste(image, (x, 0))
        x += image.width
    return sheet


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def first_n_indices_per_class(dataset, samples_per_class):
    counts = {class_id: 0 for class_id in range(len(dataset.classes))}
    selected = []
    for index, (_path, target) in enumerate(dataset.samples):
        if counts[target] < samples_per_class:
            selected.append(index)
            counts[target] += 1
        if all(count >= samples_per_class for count in counts.values()):
            break

    missing = {dataset.classes[class_id]: samples_per_class - count for class_id, count in counts.items() if count < samples_per_class}
    if missing:
        first_missing = ", ".join(f"{name}:{count}" for name, count in list(missing.items())[:10])
        raise RuntimeError(
            f"Not enough images for {len(missing)} class(es) in this split. Missing counts: {first_missing}"
        )
    return selected


def main():
    args = parse_args()
    if args.vim_root is None:
        raise ValueError(
            "--vim-root is required because the external Vim source tree is not included in this repository."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    vim_checkpoint = args.vim_checkpoint or find_vim_checkpoint(args.vim_root)
    star_checkpoint = args.star_checkpoint or find_star_checkpoint(args.star_experiment)
    dataset = build_dataset(args.data_root, args.split)

    vim_model = load_vim_model(args.vim_root, vim_checkpoint, device)
    star_model = load_star_model(args.star_experiment, star_checkpoint, device)

    vim_grid = getattr(vim_model.patch_embed, "grid_size", None)
    vim_cam = PatchGradCAM(vim_model, vim_model.patch_embed, grid_size=vim_grid)
    star_cam = PatchGradCAM(star_model, star_model.patch_embed, grid_size=(star_model.h, star_model.w))

    rows = []
    if args.samples_per_class is None:
        dataset_indices = list(range(args.index, min(args.index + args.count, len(dataset))))
    else:
        dataset_indices = first_n_indices_per_class(dataset, args.samples_per_class)
    print(f"Vim checkpoint: {vim_checkpoint}")
    print(f"Star-Mamba checkpoint: {star_checkpoint}")
    print(f"Saving comparison maps to: {args.output_dir}")
    print(f"Comparing {len(dataset_indices)} image(s)")

    try:
        for output_index, dataset_index in enumerate(dataset_indices, start=1):
            item = dataset[dataset_index]
            target = int(item["target"])
            raw = item["raw"].resize((224, 224), resample=RESAMPLE_BILINEAR)
            vim_input = item["vim"].unsqueeze(0).to(device)
            star_input = item["star"].unsqueeze(0).to(device)

            with torch.no_grad():
                vim_logits = vim_model(vim_input)
                star_logits = star_model(star_input)
                vim_pred = int(vim_logits.argmax(dim=1).item())
                star_pred = int(star_logits.argmax(dim=1).item())

            if args.target_source == "vim-pred":
                cam_class = vim_pred
            elif args.target_source == "star-pred":
                cam_class = star_pred
            else:
                cam_class = target

            vim_map, vim_pred, vim_prob = vim_cam(vim_input, cam_class)
            star_map, star_pred, star_prob = star_cam(star_input, cam_class)
            corr, cosine, top20_iou = compare_maps(vim_map, star_map, args.metric_size)

            stem = f"tinyimagenet_{args.split}_{dataset_index:05d}"
            vim_heat = colorize_cam(vim_map, raw.size)
            star_heat = colorize_cam(star_map, raw.size)
            vim_overlay = overlay_heatmap(raw, vim_heat, args.alpha)
            star_overlay = overlay_heatmap(raw, star_heat, args.alpha)
            sheet = make_contact_sheet(
                [
                    ("image", raw),
                    ("Vim heat", vim_heat),
                    ("Vim overlay", vim_overlay),
                    ("Star heat", star_heat),
                    ("Star overlay", star_overlay),
                ]
            )

            raw.save(os.path.join(args.output_dir, f"{stem}_image.png"))
            vim_heat.save(os.path.join(args.output_dir, f"{stem}_vim_gradcam.png"))
            vim_overlay.save(os.path.join(args.output_dir, f"{stem}_vim_overlay.png"))
            star_heat.save(os.path.join(args.output_dir, f"{stem}_starmamba_gradcam.png"))
            star_overlay.save(os.path.join(args.output_dir, f"{stem}_starmamba_overlay.png"))
            sheet.save(os.path.join(args.output_dir, f"{stem}_comparison.png"))
            np.savez_compressed(
                os.path.join(args.output_dir, f"{stem}_maps.npz"),
                vim_gradcam=vim_map,
                starmamba_gradcam=star_map,
                target=np.array(target),
                class_id=np.array(cam_class),
                vim_prediction=np.array(vim_pred),
                vim_probability=np.array(vim_prob),
                starmamba_prediction=np.array(star_pred),
                starmamba_probability=np.array(star_prob),
                pearson=np.array(corr),
                cosine=np.array(cosine),
                top20_iou=np.array(top20_iou),
            )

            row = {
                "index": dataset_index,
                "path": item["path"],
                "target": target,
                "cam_class": cam_class,
                "vim_prediction": vim_pred,
                "vim_probability": f"{vim_prob:.6f}",
                "starmamba_prediction": star_pred,
                "starmamba_probability": f"{star_prob:.6f}",
                "pearson": f"{corr:.6f}",
                "cosine": f"{cosine:.6f}",
                "top20_iou": f"{top20_iou:.6f}",
            }
            rows.append(row)
            print(
                f"[{output_index}/{len(dataset_indices)}] index={dataset_index} target={target} cam_class={cam_class} "
                f"vim_pred={vim_pred} star_pred={star_pred} pearson={corr:.3f} top20_iou={top20_iou:.3f}"
            )
    finally:
        vim_cam.remove()
        star_cam.remove()

    save_csv(rows, os.path.join(args.output_dir, "comparison_metrics.csv"))


if __name__ == "__main__":
    main()
