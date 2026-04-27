#!/usr/bin/env python3
"""Feature t-SNE visualization for a CIFAR-10 ResNet-34 checkpoint.

Loads a checkpoint, extracts penultimate features (after GAP, before FC) on a
random subset of the CIFAR-10 test set, projects to 2D via sklearn t-SNE,
plots colored by ground-truth class, and saves the PNG.

Usage:
  python3 scripts/visualize_tsne.py \\
      --ckpt runs/cifar10_baseline/best.pth \\
      --out  figs/tsne_baseline.png \\
      --num-samples 1000
"""
from __future__ import annotations
import argparse, os, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.cifar_demo.data import build_cifar10, CLASSES
from src.cifar_demo.model import build_resnet34
from src.cifar_demo.utils import pick_device


@torch.no_grad()
def extract_features(model, loader, device, num_samples: int = 1000):
    model.eval()
    feats, labels = [], []
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pooled, _ = model.forward_features(x)
        feats.append(pooled.cpu().numpy())
        labels.append(y.numpy())
        n += y.size(0)
        if n >= num_samples:
            break
    feats = np.concatenate(feats, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]
    return feats, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="figs/tsne.png")
    ap.add_argument("--data-root", default=os.environ.get("AUTORES_DATA_ROOT", "./data/cifar10"))
    ap.add_argument("--num-samples", type=int, default=1500)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        sys.exit("ERROR: scikit-learn required for t-SNE. Install: pip install scikit-learn")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("ERROR: matplotlib required. Install: pip install matplotlib")

    device = pick_device()
    print(f"[tsne] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_resnet34(num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])

    print("[tsne] building CIFAR-10 test loader (no augmentation)")
    _, test_loader = build_cifar10(root=args.data_root, augmentation="none",
                                    batch_size=128, num_workers=4, download=True)

    print(f"[tsne] extracting {args.num_samples} feature vectors")
    feats, labels = extract_features(model, test_loader, device, num_samples=args.num_samples)
    print(f"[tsne] features shape={feats.shape}")

    print(f"[tsne] running TSNE (perplexity={args.perplexity})")
    embed = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed,
                  init="pca", learning_rate="auto").fit_transform(feats)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        mask = labels == c
        ax.scatter(embed[mask, 0], embed[mask, 1], s=10, alpha=0.7,
                   color=cmap(c), label=CLASSES[c])
    ax.set_title(f"CIFAR-10 ResNet-34 features · t-SNE\n{pathlib.Path(args.ckpt).name}",
                  fontsize=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=9, markerscale=2.0, frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"[tsne] saved {out_path}")


if __name__ == "__main__":
    main()
