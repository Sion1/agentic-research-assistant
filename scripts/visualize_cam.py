#!/usr/bin/env python3
"""Grad-CAM visualization for a CIFAR-10 ResNet-34 checkpoint.

Picks N random test images, runs forward + backward to get Grad-CAM heatmaps
on the last conv layer (layer4), overlays them on the original image, and
saves a single grid PNG.

Usage:
  python3 scripts/visualize_cam.py \\
      --ckpt runs/cifar10_baseline/best.pth \\
      --out  figs/cam_baseline.png \\
      --num 8
"""
from __future__ import annotations
import argparse, os, pathlib, sys
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.cifar_demo.data import build_cifar10, CIFAR10_MEAN, CIFAR10_STD, CLASSES
from src.cifar_demo.model import build_resnet34
from src.cifar_demo.utils import pick_device


def denormalize(img_t: torch.Tensor) -> np.ndarray:
    """(C,H,W) tensor in normalized space → (H,W,3) uint8 RGB."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = (img_t.cpu() * std + mean).clamp(0, 1).numpy()
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def gradcam_one(model, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """Returns a (H, W) heatmap normalized to [0, 1] for a single image."""
    model.zero_grad(set_to_none=True)
    x = x.requires_grad_(True)
    logits = model(x.unsqueeze(0))                       # (1, num_classes)
    score  = logits[0, class_idx]
    score.backward()
    feat = model._cam_feat[0]                            # (C, h, w)
    grad = model._cam_grad[0]                            # (C, h, w)
    weights = grad.mean(dim=(1, 2))                      # (C,)
    cam = (weights[:, None, None] * feat).sum(dim=0)     # (h, w)
    cam = F.relu(cam)
    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    # Upsample to input resolution (32x32)
    H, W = x.shape[-2:]
    cam_t = torch.from_numpy(cam)[None, None, :, :].float()
    cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return cam_up.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="figs/cam.png")
    ap.add_argument("--data-root", default=os.environ.get("AUTORES_DATA_ROOT", "./data/cifar10"))
    ap.add_argument("--num", type=int, default=8, help="number of sample images")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("ERROR: matplotlib required. Install: pip install matplotlib")

    device = pick_device()
    print(f"[cam] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_resnet34(num_classes=10).to(device).eval()
    model.load_state_dict(ckpt["model"])

    print("[cam] building CIFAR-10 test loader")
    _, test_loader = build_cifar10(root=args.data_root, augmentation="none",
                                    batch_size=args.num, num_workers=2, download=True)
    rng = np.random.default_rng(args.seed)
    # Pick a single batch and shuffle to get diverse samples
    x, y = next(iter(test_loader))
    perm = rng.permutation(len(x))[: args.num]
    x = x[perm].to(device); y = y[perm]

    rows = 2
    cols = (args.num + rows - 1) // rows
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2.0, rows * 4.2), dpi=120)
    if axes.ndim == 1: axes = axes[None, :]

    # Predict to label correctness
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).cpu()

    for i in range(args.num):
        # cam targets the predicted class
        cam = gradcam_one(model, x[i], int(pred[i]))
        rgb = denormalize(x[i])

        r_img, c_img = (i // cols) * 2,     i % cols
        r_cam        = (i // cols) * 2 + 1
        ax_img = axes[r_img, c_img]
        ax_cam = axes[r_cam, c_img]
        ax_img.imshow(rgb)
        correct = "✓" if pred[i].item() == y[i].item() else "✗"
        title_color = "#1D9E75" if correct == "✓" else "#E24B4A"
        ax_img.set_title(f"{CLASSES[y[i].item()]}\npred {CLASSES[pred[i].item()]} {correct}",
                         fontsize=8, color=title_color)
        ax_img.axis("off")
        ax_cam.imshow(rgb)
        ax_cam.imshow(cam, cmap="jet", alpha=0.5)
        ax_cam.axis("off")

    # Hide unused cells
    for j in range(args.num, rows * cols):
        for r in range((j // cols) * 2, (j // cols) * 2 + 2):
            axes[r, j % cols].axis("off")

    fig.suptitle(f"Grad-CAM · {pathlib.Path(args.ckpt).name}", fontsize=11)
    fig.tight_layout()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"[cam] saved {out_path}")


if __name__ == "__main__":
    main()
