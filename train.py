#!/usr/bin/env python3
"""Train ResNet-34 on CIFAR-10. Loop-friendly entry point.

Reads a YAML config, trains, saves best/final checkpoints, writes the
FINISH-line that loop.sh's reaper looks for.

Usage:
  python3 train.py --config configs/cifar10_resnet34.yaml --iter-num 1

The wrapper run_experiment.sh adds extra safety (GPU selection, OOM preflight,
state.tsv update). See run_experiment.sh for the loop-integrated invocation.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time
import yaml, torch, torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from src.cifar_demo.data import build_cifar10
from src.cifar_demo.model import build_resnet34
from src.cifar_demo.trainer import build_optimizer, build_scheduler, train_one_epoch, evaluate
from src.cifar_demo.utils import set_seed, write_finish_line, pick_device

# Optional wandb integration. Activate by either setting WANDB_PROJECT env var
# or adding `wandb: { project: <name> }` to your config. If wandb isn't
# installed or no project is configured, training silently skips logging.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def _load_config(path: str) -> dict:
    raw = yaml.safe_load(pathlib.Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("config must be a YAML mapping")
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--iter-num", type=int, default=0, help="iter number for state.tsv / FINISH line")
    ap.add_argument("--out-root", default="runs", help="checkpoint output root (overrides config.output.root)")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = pick_device()
    print(f"[train] device={device} config={args.config}", flush=True)

    # ---- data ----
    data_cfg = cfg.get("data", {})
    train_loader, test_loader = build_cifar10(
        root=data_cfg.get("root", "/data/cifar10"),
        augmentation=data_cfg.get("augmentation", "standard"),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        download=bool(data_cfg.get("download", True)),
    )

    # ---- model ----
    model = build_resnet34(num_classes=int(cfg.get("model", {}).get("num_classes", 10))).to(device)

    # ---- optim + sched ----
    train_cfg = cfg.get("training", {})
    epochs = int(train_cfg.get("epochs", 20))
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, epochs=epochs)
    criterion = nn.CrossEntropyLoss()

    # ---- output dir ----
    out_root = pathlib.Path(cfg.get("output", {}).get("root", args.out_root))
    exp_name = cfg.get("exp_name", pathlib.Path(args.config).stem)
    run_dir = out_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] run_dir={run_dir} epochs={epochs}", flush=True)

    # ---- optional wandb run ----
    wandb_run = None
    wandb_cfg = cfg.get("wandb", {}) if isinstance(cfg.get("wandb"), dict) else {}
    wandb_project = wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT")
    if _WANDB_AVAILABLE and wandb_project:
        try:
            wandb_run = wandb.init(
                project=wandb_project,
                name=exp_name,
                config=cfg,
                reinit=True,
                dir=str(run_dir),
                tags=wandb_cfg.get("tags") or [],
            )
            print(f"[train] wandb run = {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[train] wandb init failed ({e!r}); continuing without wandb", flush=True)
            wandb_run = None
    elif _WANDB_AVAILABLE and not wandb_project:
        print("[train] wandb installed but WANDB_PROJECT not set — skipping (set env or cfg.wandb.project)", flush=True)

    # ---- train ----
    history = []
    best = {"acc": 0.0, "epoch": -1}
    rc = 0
    try:
        for ep in range(epochs):
            train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
            eval_stats  = evaluate(model, test_loader, criterion, device)
            if scheduler is not None:
                scheduler.step()
            row = {"epoch": ep, **{f"train_{k}": v for k, v in train_stats.items()},
                   **{f"test_{k}": v for k, v in eval_stats.items()}}
            history.append(row)
            print(f"[ep {ep:>3}/{epochs}]  train_loss={train_stats['loss']:.4f}  "
                  f"train_acc={train_stats['acc']:.4f}  test_loss={eval_stats['loss']:.4f}  "
                  f"test_acc={eval_stats['acc']:.4f}  ({train_stats['elapsed']:.1f}s)",
                  flush=True)
            if wandb_run is not None:
                wandb_run.log(row, step=ep)

            if eval_stats["acc"] > best["acc"]:
                best = {"acc": eval_stats["acc"], "epoch": ep}
                torch.save({
                    "model": model.state_dict(),
                    "metrics": {"acc": eval_stats["acc"], "loss": eval_stats["loss"], "epoch": ep},
                    "config": cfg,
                }, run_dir / "best.pth")
    except KeyboardInterrupt:
        rc = 130
        print("[train] interrupted", flush=True)
    except Exception as e:
        rc = 1
        print(f"[train] FAILED: {e!r}", flush=True)
        raise
    finally:
        # Always save final.pth + history
        torch.save({
            "model": model.state_dict(),
            "metrics": {"acc": eval_stats.get("acc", 0.0) if 'eval_stats' in locals() else 0.0,
                        "loss": eval_stats.get("loss", 0.0) if 'eval_stats' in locals() else 0.0,
                        "best_acc": best["acc"], "best_epoch": best["epoch"]},
            "config": cfg,
            "history": history,
        }, run_dir / "final.pth")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"[train] saved {run_dir/'final.pth'}  best_acc={best['acc']:.4f} (epoch {best['epoch']})", flush=True)
        if wandb_run is not None:
            wandb_run.summary["best_acc"]   = best["acc"]
            wandb_run.summary["best_epoch"] = best["epoch"]
            wandb_run.finish()
        write_finish_line(args.iter_num, rc=rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
