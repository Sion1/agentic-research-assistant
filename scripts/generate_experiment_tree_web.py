#!/usr/bin/env python3
"""Generate an interactive autoresearch experiment-tree webpage.

The page is static and self-contained except for relative links to existing
logs/figs/configs. It is safe to regenerate after every loop tick.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import mimetypes
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "autoresearch_dashboard" / "index.html"
USER_SUMMARY = ROOT / "state" / "user_summary.md"
USER_SUMMARIES = ROOT / "state" / "user_summaries.md"

# Filename-extension allow-lists for `iter_fig_artifacts()`. Heavyweight
# tensors (.pt) and other cache files are skipped on purpose — the dashboard
# is meant for inspection, not bulk artifact distribution.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
FIG_ARTIFACT_EXTS = IMAGE_EXTS | {".csv", ".json", ".html", ".txt", ".md"}


def _git_show_blob(branch: str, path: str) -> bytes | None:
    """Read `path` at `branch` in this repo via `git show`. Returns None on
    any failure (branch missing, file missing, git not available, etc.).

    Used to recover iter outputs that git_iter_commit.sh has wiped from
    main's working tree but still exist on autoresearch/iter-NNN."""
    try:
        result = subprocess.run(
            ["git", "show", f"{branch}:{path}"],
            cwd=str(ROOT),
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def _read_iter_artifact(rel_path: str) -> bytes | None:
    """Read an iter artifact (figs/iter_NNN/* or logs/iteration_NNN.md).

    Tries main's working tree first; falls back to autoresearch/iter-NNN
    via git show. The fallback is necessary because git_iter_commit.sh
    only restores the most-recent iter's outputs — earlier iters' figs
    and reports stay only on their per-iter branch."""
    full = ROOT / rel_path
    if full.exists() and full.is_file():
        try:
            return full.read_bytes()
        except OSError:
            pass
    # Try git fallback. Only viable for path patterns that name the iter.
    m = re.search(r"(?:figs/iter_|logs/iteration_)(\d+)", rel_path)
    if not m:
        return None
    iter_pad = m.group(1)
    return _git_show_blob(f"autoresearch/iter-{iter_pad}", rel_path)


def _read_iter_text(rel_path: str) -> str:
    """Text-mode wrapper around _read_iter_artifact."""
    blob = _read_iter_artifact(rel_path)
    if blob is None:
        return ""
    return blob.decode("utf-8", errors="replace")


@dataclass
class IterRow:
    iter_id: int
    status: str
    exp_name: str
    config: str
    gpu: str
    pid: str
    started_at: str
    finished_at: str
    best_metric: str   # column 9 of state.tsv — name varies by project (best_metric / best_acc / best_f1 / ...)
    verdict: str


@dataclass
class IterNode:
    id: str
    name: str
    detail: str
    status: str
    x: int
    y: int
    metric: str = ""
    idea: str = ""
    reason: str = ""
    notes: str = ""
    links: list[dict[str, str]] = field(default_factory=list)
    visuals: list[dict[str, str]] = field(default_factory=list)
    analysis: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$", re.MULTILINE)


def read_state(path: Path) -> list[IterRow]:
    if not path.exists():
        raise FileNotFoundError(f"state file not found: {path}")
    rows: list[IterRow] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            if not raw.get("iter") or not raw["iter"].isdigit():
                continue
            rows.append(
                IterRow(
                    iter_id=int(raw.get("iter", "0")),
                    status=raw.get("status", ""),
                    exp_name=raw.get("exp_name", ""),
                    config=raw.get("config", ""),
                    gpu=raw.get("gpu", ""),
                    pid=raw.get("pid", ""),
                    started_at=raw.get("started_at", ""),
                    finished_at=raw.get("finished_at", ""),
                    # state.tsv column 9: try common names, fall back to whatever the launcher used
                    best_metric=(raw.get("best_metric")
                                 or raw.get("best_acc")
                                 or raw.get("best_f1")
                                 or list(raw.values())[8] if len(raw) > 8 else ""),
                    verdict=raw.get("verdict", ""),
                )
            )
    return sorted(rows, key=lambda r: (r.iter_id == 999, r.iter_id))


def section(text: str, number: int) -> str:
    matches = list(SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        if int(m.group(1)) != number:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        return text[start:end].strip()
    return ""


def compact(text: str, max_len: int = 520) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def first_bullets(text: str, max_items: int = 3) -> list[str]:
    bullets = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            bullets.append(compact(line[2:], 180))
        if len(bullets) >= max_items:
            break
    return bullets


def parse_metric_table(report: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    results = section(report, 4)
    for line in results.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"---", "Metric"}:
            continue
        key = cells[0].strip("` ")
        # Prefer the "This run" column if present, otherwise use the last
        # numeric-looking value in the row.
        value = ""
        header_line = next((l for l in results.splitlines() if l.startswith("| Metric ")), "")
        headers = [c.strip() for c in header_line.strip().strip("|").split("|")] if header_line else []
        if headers and "This run" in " ".join(headers):
            for idx, h in enumerate(headers):
                if idx < len(cells) and "This run" in h:
                    value = cells[idx]
                    break
        if not value:
            for c in reversed(cells[1:]):
                if re.search(r"\d", c):
                    value = c
                    break
        value = re.sub(r"\*\*", "", value).strip()
        if value:
            metrics[key] = value
    return metrics


def load_ckpt_metrics(exp_name: str) -> dict[str, Any]:
    """Best-effort checkpoint metric loading for running jobs.

    If torch is unavailable or checkpoint loading fails, return {}.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return {}
    # Try common output-root layouts. Adapt for your launcher if it writes
    # checkpoints elsewhere.
    candidates: list[Path] = []
    for root in ("runs",):
        candidates += sorted((ROOT / root / exp_name).glob("*/best.pth"))
        candidates += sorted((ROOT / root / exp_name).glob("best.pth"))
        candidates += sorted((ROOT / root / exp_name).glob("*/final.pth"))
        candidates += sorted((ROOT / root / exp_name).glob("final.pth"))
    if not candidates:
        return {}
    try:
        ckpt = torch.load(candidates[-1], map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            return ckpt.get("metrics", {}) or {}
    except Exception:
        return {}
    return {}


def config_values(config_path: str) -> dict[str, str]:
    path = ROOT / config_path
    if not config_path or not path.exists():
        return {}
    vals: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    key = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m_key = re.match(r"^([A-Za-z0-9_]+):\s*$", line)
        if m_key:
            key = m_key.group(1)
            continue
        m_val = re.match(r"^\s+value:\s*(.+?)\s*$", line)
        if key and m_val:
            vals[key] = m_val.group(1).strip().strip("'\"")
            key = None
    return vals


def visual_label(rel_path: str) -> str:
    """Human-friendly tab label from a viz file path under figs/iter_NNN/.

    Maps known stems to nicer labels; falls back to the filename stem with
    underscores/dashes turned into spaces. Adapt the `names` dict for your
    project's viz scripts (e.g. add `"confusion_normalized": "confusion (norm)"`)."""
    path = Path(rel_path)
    stem = path.stem.lower()
    names = {
        "tsne": "t-SNE",
        "cam": "Grad-CAM",
        "per_class": "per-class",
        "per_class_delta": "delta",
        "confusion": "confusion",
    }
    label = names.get(stem, path.stem.replace("_", " ").replace("-", " "))
    if len(path.parts) > 1:
        parent = path.parent.as_posix().replace("_", " ").strip(" ./")
        if parent and parent != ".":
            label = f"{label} · {parent}"
    return label


def iter_fig_artifacts(iter_pad: str) -> list[tuple[str, str, bool]]:
    """Return [(rel_path_under_iter_dir, label, is_image), ...] for one iter.

    Source priority:
      1. Working tree (figs/iter_NNN/) — recursive scan, picks up any viz
         the project's scripts produced, not a hardcoded filename list.
      2. autoresearch/iter-NNN branch — when git_iter_commit.sh has wiped
         the iter's outputs from main but they still live on the branch.
         Uses `git ls-tree -r` (cheap) and yields the same shape.

    Heavyweight cache files (.pt) are skipped on purpose."""
    fig_rel = f"figs/iter_{iter_pad}"
    fig_dir = ROOT / fig_rel
    artifacts: list[tuple[str, str, bool]] = []

    # 1. Working tree scan (preferred — handles the latest iter)
    if fig_dir.exists():
        for path in sorted(fig_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in FIG_ARTIFACT_EXTS:
                continue
            rel = path.relative_to(fig_dir).as_posix()
            artifacts.append((rel, visual_label(rel), suffix in IMAGE_EXTS))
        if artifacts:
            return artifacts

    # 2. Git-branch fallback (recovers older iters wiped by git_iter_commit.sh)
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only",
             f"autoresearch/iter-{iter_pad}", "--", fig_rel],
            cwd=str(ROOT), capture_output=True, timeout=10, text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line.startswith(fig_rel + "/"):
                    continue
                suffix = Path(line).suffix.lower()
                if suffix not in FIG_ARTIFACT_EXTS:
                    continue
                rel = line[len(fig_rel) + 1:]
                artifacts.append((rel, visual_label(rel), suffix in IMAGE_EXTS))
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return artifacts


def group_for(exp_name: str, config: dict[str, str], iter_id: int) -> tuple[str, str, str]:
    """Map an experiment to a group node in the tree.

    The framework's dashboard groups iters by the *axis* their hypothesis
    targets, NOT by the absolute config. Adapt the keyword rules below for
    your project — the only contract is that every iter ends up in some
    group, and the group order in `build_tree()` reflects how you want the
    tree to read top-to-bottom.

    Default rules below match the bundled CIFAR-10 + ResNet-34 demo. The
    exp_name keywords come from the demo's ablation/<axis>.yaml files.
    """
    name = exp_name.lower()
    if iter_id == 999 or "smoketest" in name:
        return "sanity", "Sanity / Smoke", "Pipeline health checks & smoketests"
    if "no_aug" in name or "noaug" in name:
        return "baseline", "Baseline (no aug)", "Bare baseline — no data augmentation"
    if "autoaug" in name or "auto_aug" in name:
        return "aug",      "Augmentation",     "Augmentation strength sweep"
    if "adamw" in name or "adam" in name:
        return "opt",      "Optimizer",        "Optimizer family / hyperparameter sweep"
    if "multistep" in name or "schedule" in name:
        return "sched",    "LR schedule",      "Learning-rate schedule variants"
    if "long" in name or "epochs" in name:
        return "budget",   "Training budget",  "Epoch / step-budget variants"
    if "wd" in name or "weight_decay" in name:
        return "reg",      "Regularization",   "Weight decay / dropout / smoothing variants"
    if "baseline" in name or "default" in name:
        return "baseline", "Baseline",         "Reference baseline"
    return "other",        "Other",            "Unclassified experiments"


def display_status(row: IterRow) -> str:
    verdict = row.verdict.lower()
    if row.status == "running":
        return "running"
    if verdict == "success":
        return "success"
    if verdict in {"failure", "bug"} or row.status == "failed":
        return "failed"
    if verdict in {"partial", "noise"}:
        return "warning"
    return "info"


def fmt_num(v: Any) -> str:
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def build_iter_node(row: IterRow, parent_x: int, parent_y: int, idx: int, sibling_count: int) -> IterNode:
    iter_pad = f"{row.iter_id:03d}"
    # Use the helper so we fall back to the iter's branch when git_iter_commit
    # has wiped older iters' reports from main's working tree.
    report = _read_iter_text(f"logs/iteration_{iter_pad}.md")
    report_path = ROOT / "logs" / f"iteration_{iter_pad}.md"
    cfg = config_values(row.config)
    ckpt = load_ckpt_metrics(row.exp_name) if row.status == "running" or not row.best_metric else {}
    parsed = parse_metric_table(report) if report else {}

    # Headline metric pulled from (in order): state.tsv column 9 (anything the
    # launcher writes), the iter report's parsed table, and the saved
    # checkpoint's `metrics` dict. Adapt the alternate keys below for your
    # project — the demo uses `acc` and `best_acc`.
    primary = (row.best_metric
               or parsed.get("acc") or parsed.get("test_acc") or parsed.get("metric")
               or fmt_num(ckpt.get("acc") or ckpt.get("best_acc") or ckpt.get("test_acc")))
    secondary = parsed.get("loss") or fmt_num(ckpt.get("loss") or ckpt.get("best_loss"))
    extra1 = parsed.get("top5") or fmt_num(ckpt.get("top5"))
    extra2 = parsed.get("epoch") or fmt_num(ckpt.get("best_epoch"))

    pieces = []
    if primary:
        pieces.append(f"acc={primary}")
    if secondary:
        pieces.append(f"loss={secondary}")
    if extra1:
        pieces.append(f"top5={extra1}")
    metric = "  ".join(pieces) if pieces else ("running" if row.status == "running" else "")

    hypothesis = compact(section(report, 1), 620)
    verdict_text = compact(section(report, 6), 460)
    decision = compact(section(report, 7), 460)
    next_step = compact(section(report, 8), 460)
    notes = verdict_text or decision or ("训练仍在运行，等待下一次 analyze tick。" if row.status == "running" else "")

    detail_bits = []
    if row.verdict:
        detail_bits.append(row.verdict)
    elif row.status:
        detail_bits.append(row.status)
    # Project-specific quick-glance config bits (override these for your
    # domain — pick the 1-2 hyperparameters that distinguish ablation cells).
    # Demo: optimizer + augmentation.
    if cfg.get("optimizer"):
        detail_bits.append(cfg["optimizer"])
    if cfg.get("augmentation"):
        detail_bits.append(f"aug={cfg['augmentation']}")
    detail = " · ".join(detail_bits)

    links = []
    # Report link is added if the report exists EITHER in the working tree
    # OR on the iter branch (the helper's fallback). copy_one() handles the
    # actual bundling via the same fallback, so a working-tree miss still
    # produces a clickable artifact in docs/autoresearch_dashboard/assets/.
    if report:
        links.append({"label": "report", "href": f"../logs/iteration_{iter_pad}.md"})
    if row.config and (ROOT / row.config).exists():
        links.append({"label": "config", "href": f"../{row.config}"})
    visuals = []

    # Recursively scan figs/iter_NNN/ (working tree first, autoresearch/iter-NNN
    # branch second) for ANY viz file the project produced. Replaces the older
    # hardcoded filename list — adding a new viz to your training pipeline now
    # appears in the dashboard automatically without editing this script.
    for filename, label, is_image in iter_fig_artifacts(iter_pad):
        href = f"../figs/iter_{iter_pad}/{filename}"
        links.append({"label": label, "href": href})
        if is_image:
            visuals.append({"label": label, "href": href})

    x = parent_x + 340
    center = (sibling_count - 1) / 2.0
    y = int(parent_y + (idx - center) * 84)
    return IterNode(
        id=f"iter{iter_pad}",
        name=f"Iter {iter_pad}",
        detail=detail,
        status=display_status(row),
        x=x,
        y=y,
        metric=metric,
        idea=hypothesis,
        reason=decision,
        notes=notes,
        links=links,
        visuals=visuals,
        analysis={
            "issue": verdict_text[:140] if verdict_text else ("Running" if row.status == "running" else "No analysis report yet"),
            "causes": first_bullets(section(report, 5), 3) or first_bullets(section(report, 4), 3),
            "fixes": first_bullets(section(report, 8), 3),
        }
        if display_status(row) in {"failed", "warning", "running"}
        else None,
        meta={
            "exp_name": row.exp_name,
            "gpu": row.gpu,
            "pid": row.pid,
            "started_at": row.started_at,
            "finished_at": row.finished_at,
            "acc": primary,
            "loss": secondary,
            "top5": extra1,
            "best_epoch": extra2,
        },
    )


def build_tree(rows: list[IterRow]) -> dict[str, Any]:
    rows = [r for r in rows if r.exp_name]
    configs = {r.iter_id: config_values(r.config) for r in rows}
    groups: dict[str, dict[str, Any]] = {}
    for r in rows:
        key, label, detail = group_for(r.exp_name, configs.get(r.iter_id, {}), r.iter_id)
        groups.setdefault(key, {"id": key, "name": label, "detail": detail, "rows": []})
        groups[key]["rows"].append(r)

    order = ["baseline", "aug", "opt", "sched", "reg", "budget", "sanity", "other"]
    group_items = [groups[k] for k in order if k in groups]
    group_count = len(group_items)
    x_group = 390
    span = 560
    start_y = 360 - span / 2

    nodes = []
    for gi, group in enumerate(group_items):
        y = int(start_y + gi * (span / max(group_count - 1, 1)))
        row_statuses = [display_status(r) for r in group["rows"]]
        if "running" in row_statuses:
            status = "running"
        elif "success" in row_statuses:
            status = "success"
        elif row_statuses and all(s == "failed" for s in row_statuses):
            status = "failed"
        elif "warning" in row_statuses:
            status = "warning"
        else:
            status = "info"
        children = [
            build_iter_node(r, x_group, y, idx, len(group["rows"]))
            for idx, r in enumerate(sorted(group["rows"], key=lambda rr: rr.iter_id))
        ]
        metric_vals = []
        for child in children:
            try:
                metric_vals.append(float(child.meta.get("acc") or "nan"))
            except Exception:
                pass
        metric_vals = [v for v in metric_vals if not math.isnan(v)]
        metric = f"best acc={max(metric_vals):.2f}" if metric_vals else ""
        nodes.append(
            {
                "id": group["id"],
                "name": group["name"],
                "detail": group["detail"],
                "status": status,
                "x": x_group,
                "y": y,
                "metric": metric,
                "notes": f"{len(children)} experiments",
                "subs": [child.__dict__ for child in children],
            }
        )

    all_metrics = []
    for group in nodes:
        for child in group["subs"]:
            try:
                all_metrics.append((float(child["meta"].get("acc")), child["name"]))
            except Exception:
                pass
    best = max(all_metrics, default=None)
    task_metric = f"current best {best[1]} · acc={best[0]:.2f}" if best else ""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    node_summaries = read_node_summaries(USER_SUMMARIES)
    return {
        "task": {
            "id": "task",
            "name": "AutoResearch",
            "detail": "实验探索树 · 自动从 state / logs / figs 生成",
            "status": "info",
            "x": 80,
            "y": 360,
            "metric": task_metric,
            "notes": f"Generated at {generated}. Click any node to inspect metrics, rationale, reports and artifacts.",
        },
        "nodes": nodes,
        "user_summary": USER_SUMMARY.read_text(encoding="utf-8", errors="replace") if USER_SUMMARY.exists() else "",
        "user_summaries": node_summaries,
    }


def read_node_summaries(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    parts = re.split(r"(?m)^<!--\s*node:([^>]+?)\s*-->\s*$", text)
    for i in range(1, len(parts), 2):
        node_id = parts[i].strip()
        body = parts[i + 1]
        body = re.sub(r"(?m)^##\s+.*?\n", "", body, count=1).strip()
        if node_id:
            out[node_id] = body
    return out


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoResearch 实验探索树</title>
<style>
:root{
  --bg-page:#fafaf7;--bg-card:#ffffff;--bg-soft:#f4f3ee;
  --text-primary:#1a1a1a;--text-secondary:#6b6a64;--text-tertiary:#9a988f;
  --border:rgba(0,0,0,0.08);--border-strong:rgba(0,0,0,0.16);
  --edge:rgba(120,120,110,0.35);
  --shadow-sm:0 1px 3px rgba(20,20,15,0.06),0 1px 2px rgba(20,20,15,0.04);
  --shadow-lg:0 12px 40px rgba(20,20,15,0.12),0 2px 8px rgba(20,20,15,0.06);
  --font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  --font-mono:"SF Mono",Monaco,Consolas,"Courier New",monospace;
}
@media (prefers-color-scheme: dark){
  :root{--bg-page:#0e0e0c;--bg-card:#1a1a18;--bg-soft:#242421;
    --text-primary:#f0eee6;--text-secondary:#a8a69d;--text-tertiary:#6e6c64;
    --border:rgba(255,255,255,0.08);--border-strong:rgba(255,255,255,0.18);
    --edge:rgba(180,180,170,0.25);--shadow-sm:0 1px 3px rgba(0,0,0,0.4);
    --shadow-lg:0 12px 40px rgba(0,0,0,0.5),0 2px 8px rgba(0,0,0,0.3)}
}
*{box-sizing:border-box} html,body{margin:0;padding:0}
body{font-family:var(--font-sans);background:var(--bg-page);color:var(--text-primary);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased;min-height:100vh;overflow:hidden}
.page{width:100vw;height:100vh;margin:0;padding:0;display:flex;flex-direction:column}
.hero{display:grid;grid-template-columns:minmax(360px,1fr) minmax(320px,520px) auto;gap:18px;align-items:center;
  padding:18px 28px;border-bottom:0.5px solid var(--border);background:rgba(255,255,255,0.72);backdrop-filter:blur(18px)}
.title{font-size:28px;font-weight:600;margin:0 0 3px;letter-spacing:0}
.subtitle{font-size:13px;color:var(--text-tertiary);margin:0}
.hero-actions{display:flex;gap:10px;align-items:center;justify-content:flex-end;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:14px}
.logo{width:42px;height:42px;border-radius:14px;background:radial-gradient(circle at 30% 30%,#eff6ff,#dbeafe);border:1px solid #bfdbfe;position:relative;box-shadow:var(--shadow-sm)}
.logo::before,.logo::after{content:"";position:absolute;border-radius:50%;background:#2563eb}.logo::before{width:8px;height:8px;left:7px;top:7px}.logo::after{width:8px;height:8px;right:7px;bottom:7px}
.search{height:42px;width:100%;border:0.5px solid var(--border);border-radius:11px;background:var(--bg-card);color:var(--text-primary);font-family:inherit;font-size:13px;padding:0 14px;box-shadow:var(--shadow-sm)}
.search:focus{outline:none;border-color:var(--border-strong)}
.lang-select{font-size:12.5px;padding:6px 10px;border-radius:8px;border:0.5px solid var(--border-strong);background:var(--bg-card);color:var(--text-primary);font-family:inherit}
.shell{padding:16px 28px 24px;display:flex;flex-direction:column;min-height:0;flex:1}
.toolbar{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:12px;margin:0 0 14px}
.chip{font-size:12px;padding:14px 16px;border-radius:11px;background:var(--bg-card);color:var(--text-secondary);border:0.5px solid var(--border);box-shadow:var(--shadow-sm);display:flex;align-items:center;justify-content:space-between;min-height:72px}
.chip strong{display:block;font-size:24px;color:var(--text-primary);line-height:1}
.toolbar-actions{display:flex;gap:8px;align-items:center;margin:-4px 0 14px;justify-content:flex-end;flex-wrap:wrap}
.toolbar-actions .chip{min-height:auto;padding:5px 10px;border-radius:999px;display:inline-flex;gap:6px}
.workspace{display:grid;grid-template-columns:1fr;gap:18px;min-height:0;flex:1}
.page.detail-open .workspace{grid-template-columns:minmax(760px,1fr) minmax(420px,34vw)}
.canvas{position:relative;background:var(--bg-card);border-radius:12px;box-shadow:var(--shadow-sm);overflow:hidden;min-height:0;display:flex;flex-direction:column}
.canvas-head{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:14px 18px;border-bottom:0.5px solid var(--border)}
.canvas-title{font-size:16px;font-weight:600}
.tree-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.control-btn,.control-select{height:32px;border:0.5px solid var(--border);background:var(--bg-card);color:var(--text-primary);border-radius:8px;font-family:inherit;font-size:12px;padding:0 11px;box-shadow:var(--shadow-sm);cursor:pointer}
.control-btn:hover,.control-select:hover{background:var(--bg-soft)}
.zoom-group{display:inline-flex;border:0.5px solid var(--border);border-radius:8px;overflow:hidden;box-shadow:var(--shadow-sm);background:var(--bg-card)}
.zoom-group .control-btn{border:0;border-radius:0;box-shadow:none}
.zoom-label{min-width:54px;text-align:center;font-size:12px;padding:6px 10px;border-left:0.5px solid var(--border);border-right:0.5px solid var(--border)}
.tree-body{position:relative;min-height:0;flex:1;overflow:auto;padding:22px}
svg.tree{display:block;width:100%;height:100%;min-width:900px;min-height:720px;user-select:none;transform-origin:0 0}
.legend{position:absolute;left:18px;bottom:18px;background:rgba(255,255,255,.78);border:0.5px solid var(--border);border-radius:10px;box-shadow:var(--shadow-sm);padding:10px 12px;font-size:11.5px;color:var(--text-secondary);backdrop-filter:blur(12px)}
.legend-title{font-weight:600;color:var(--text-primary);margin-bottom:5px}
.legend-row{display:flex;align-items:center;gap:7px;line-height:1.75}
.legend-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.drag-ghost{position:fixed;z-index:200;pointer-events:none;background:var(--text-primary);color:var(--bg-card);border-radius:999px;padding:6px 10px;font-size:12px;box-shadow:var(--shadow-lg);opacity:.92}
.compare-dock.drop-target{outline:2px solid #2563eb;outline-offset:3px}
.detail-panel{display:none;background:var(--bg-card);border:0.5px solid var(--border);border-radius:12px;box-shadow:var(--shadow-sm);padding:18px 20px;min-height:0;overflow:auto}
.page.detail-open .detail-panel{display:block}
.empty-detail{height:100%;display:flex;align-items:center;justify-content:center;text-align:center;color:var(--text-tertiary);font-size:13px}
@media (max-width:1100px){
  body{overflow:auto}.page{height:auto;min-height:100vh}.hero{grid-template-columns:1fr}.shell{padding:16px}.toolbar{grid-template-columns:repeat(2,minmax(120px,1fr))}.workspace{grid-template-columns:1fr}.detail-panel{min-height:520px}
  .page.detail-open .workspace{grid-template-columns:1fr}
}
g[data-id]{cursor:pointer;transition:opacity .15s ease}g[data-id].dim{opacity:.18}.halo{opacity:0;transition:opacity .18s ease}
g[data-id]:hover .halo,g[data-id].sel .halo,g[data-id].cmp .halo{opacity:.18}g[data-id].sel .halo{opacity:.28}g[data-id].cmp .halo{opacity:.22}
.dot-main{transition:r .18s ease}g[data-id]:hover .dot-main,g[data-id].sel .dot-main{r:11}
.ring-sel{opacity:0;transition:opacity .15s ease}g[data-id].sel .ring-sel,g[data-id].cmp .ring-sel{opacity:1}
.node-label{font-size:12.5px;fill:var(--text-primary);font-family:var(--font-sans);pointer-events:none;font-weight:500;letter-spacing:0}
.node-detail{font-size:10.5px;fill:var(--text-tertiary);font-family:var(--font-sans);pointer-events:none}
.edge{fill:none;stroke:var(--edge);stroke-width:1.1;transition:stroke .18s}
.pop-status{display:inline-flex;align-items:center;gap:6px;font-size:11px;padding:3px 9px;border-radius:10px;margin-bottom:10px;font-weight:500}
.pop-status.success{background:#E1F5EE;color:#0F6E56}.pop-status.failed{background:#FCEBEB;color:#791F1F}
.pop-status.warning{background:#FAEEDA;color:#854F0B}.pop-status.running{background:#E6F1FB;color:#0C447C}
.pop-status.info{background:var(--bg-soft);color:var(--text-secondary)}
@media (prefers-color-scheme: dark){
  .pop-status.success{background:#0F3D2E;color:#9FE1CB}.pop-status.failed{background:#3a1a1a;color:#F7C1C1}
  .pop-status.warning{background:#3a2a0a;color:#FAC775}.pop-status.running{background:#0C2C53;color:#B5D4F4}}
.pop-status .pulse{width:6px;height:6px;border-radius:50%;background:currentColor}
.pop-name{font-size:16px;font-weight:500;margin:0;letter-spacing:0}.pop-detail{font-size:13px;color:var(--text-secondary);margin:2px 0 0}
.pop-meta{margin-top:14px;padding:10px 12px;background:var(--bg-soft);border-radius:10px;font-family:var(--font-mono);font-size:12.5px;color:var(--text-primary)}
.pop-line{font-size:12.5px;line-height:1.55;margin-top:10px}.pop-line .lab{color:var(--text-tertiary);font-style:italic;margin-right:4px}
.pop-notes{font-size:12.5px;color:var(--text-secondary);margin-top:12px;line-height:1.6}
.section-h{font-size:11px;color:var(--text-tertiary);margin:16px 0 8px;font-weight:500;letter-spacing:0.04em;text-transform:uppercase}
.analysis{border-left:2px solid #E24B4A;background:#FCEBEB;padding:11px 13px;border-radius:0 10px 10px 0}
.analysis.warn{border-left-color:#EF9F27;background:#FAEEDA}.analysis.running{border-left-color:#378ADD;background:#E6F1FB}
@media (prefers-color-scheme: dark){.analysis{background:#3a1a1a}.analysis.warn{background:#3a2a0a}.analysis.running{background:#0C2C53}}
.analysis-h{font-size:12.5px;font-weight:500;color:#791F1F;margin-bottom:6px}.analysis.warn .analysis-h{color:#854F0B}.analysis.running .analysis-h{color:#0C447C}
.cause{font-size:12px;color:var(--text-secondary);margin:4px 0 8px;padding-left:14px}.cause li{margin:2px 0}
.links{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.link-btn{font-size:11.5px;padding:4px 9px;border-radius:8px;border:0.5px solid var(--border-strong);background:var(--bg-card);color:var(--text-primary);text-decoration:none}
.link-btn:hover{background:var(--bg-soft)}
.visual-tabs{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 10px}
.visual-tab{font-size:11.5px;padding:4px 9px;border-radius:8px;border:0.5px solid var(--border-strong);background:var(--bg-card);color:var(--text-primary);cursor:pointer;font-family:inherit}
.visual-tab.active,.visual-tab:hover{background:var(--bg-soft)}
.visual-frame{background:var(--bg-soft);border:0.5px solid var(--border);border-radius:10px;padding:8px;min-height:220px;display:flex;align-items:center;justify-content:center;overflow:auto}
.visual-frame img{max-width:100%;max-height:420px;border-radius:8px;display:block;object-fit:contain}
.visual-caption{font-size:11.5px;color:var(--text-tertiary);margin-top:6px}
.pop-actions{display:flex;gap:8px;margin-top:14px}
.btn{font-size:12.5px;padding:6px 12px;border-radius:8px;border:0.5px solid var(--border-strong);background:transparent;
  color:var(--text-primary);cursor:pointer;font-family:inherit;transition:background .12s,transform .08s;flex:1}
.btn:hover{background:var(--bg-soft)}.btn:active{transform:scale(.98)}
.btn.primary{background:var(--text-primary);color:var(--bg-card);border-color:var(--text-primary)}
.detail-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
.icon-btn{width:28px;height:28px;border:none;background:transparent;color:var(--text-tertiary);border-radius:7px;cursor:pointer;font-size:18px;line-height:1}
.icon-btn:hover{background:var(--bg-soft);color:var(--text-primary)}
.compare-overlay{position:fixed;inset:18px;background:var(--bg-card);border:0.5px solid var(--border);border-radius:14px;box-shadow:var(--shadow-lg);z-index:50;display:none;flex-direction:column;min-height:0}
.compare-overlay.show{display:flex}
.compare-head{display:flex;align-items:center;justify-content:space-between;padding:16px 18px;border-bottom:0.5px solid var(--border)}
.compare-title{font-size:17px;font-weight:500}
.compare-content{display:flex;flex-direction:column;min-height:0;flex:1}
.compare-body{display:flex;gap:0;padding:0 18px 18px;min-height:0;flex:1;overflow:hidden}
.compare-card{border:0.5px solid var(--border);border-radius:12px;padding:14px;min-width:180px;background:var(--bg-card);overflow:auto}
.compare-card h3{font-size:15px;font-weight:500;margin:0 0 4px}
.compare-card .visual-frame{height:52vh;min-height:320px}
.compare-card .visual-frame img{max-height:none;width:100%;height:100%;object-fit:contain}
.compare-resizer{cursor:col-resize;display:flex;align-items:center;justify-content:center;flex:0 0 16px;touch-action:none}
.compare-resizer::before{content:"";width:3px;height:100%;border-radius:999px;background:var(--border-strong)}
.compare-resizer:hover::before{background:var(--text-tertiary)}
.compare-empty{padding:24px;color:var(--text-tertiary);font-size:13px}
.compare-range{width:180px;accent-color:var(--text-primary)}
.compare-dock{display:none;margin-top:14px;background:var(--bg-card);border:0.5px solid var(--border);border-radius:12px;box-shadow:var(--shadow-sm);padding:12px 14px;min-height:0}
.compare-dock.show{display:block}
.compare-dock-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:10px}
.compare-dock-title{font-size:16px;font-weight:600}
.compare-dock-sub{font-size:12px;color:var(--text-tertiary)}
.compare-dock-actions{display:grid;grid-template-columns:1fr;gap:6px;min-width:76px}
.compare-dock-actions .btn{padding:5px 9px;line-height:1.25}
.compare-dock-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;max-height:230px;overflow:auto}
.compare-drop-empty{border:1px dashed var(--border-strong);border-radius:10px;min-height:76px;display:flex;align-items:center;justify-content:center;color:var(--text-tertiary);font-size:12.5px;background:var(--bg-soft)}
.mini-card{border:0.5px solid var(--border);border-radius:10px;padding:10px;background:var(--bg-card);min-height:150px}
.mini-card-head{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.mini-index{font-size:11px;color:#2563eb;background:#eff6ff;border:0.5px solid #bfdbfe;border-radius:6px;padding:1px 8px}
.mini-card-main{display:grid;grid-template-columns:124px 1fr;gap:10px}
.mini-card-img{background:var(--bg-soft);border:0.5px solid var(--border);border-radius:8px;display:flex;align-items:center;justify-content:center;overflow:hidden;min-height:88px}
.mini-card-img img{width:100%;height:100%;object-fit:contain;display:block}
.mini-card h3{font-size:13px;margin:0;font-weight:600;flex:1}
.mini-metrics{display:grid;grid-template-columns:repeat(3,1fr);border:0.5px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:8px}
.mini-metric{font-size:11px;padding:4px 8px;border-right:0.5px solid var(--border);white-space:nowrap}
.mini-metric:last-child{border-right:0}
.mini-metric span{display:block;color:var(--text-tertiary);font-size:10px}
.mini-card-note{font-size:11.5px;color:var(--text-secondary);line-height:1.45;max-height:48px;overflow:hidden}
.summary-box{width:100%;min-height:120px;resize:vertical;border:0.5px solid var(--border);border-radius:10px;background:var(--bg-card);color:var(--text-primary);font-family:var(--font-sans);font-size:12.5px;line-height:1.55;padding:10px}
@media (max-width:900px){.compare-body{display:block;overflow:auto}.compare-resizer{display:none}.compare-overlay{inset:10px}.compare-card{margin-bottom:12px;min-width:0}.compare-card .visual-frame{height:360px}.mini-card-main{grid-template-columns:1fr}}
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(20px);background:var(--text-primary);
  color:var(--bg-card);padding:9px 16px;border-radius:8px;font-size:12.5px;opacity:0;transition:opacity .2s,transform .2s;pointer-events:none;z-index:100}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
</style>
</head>
<body>
<div class="page">
  <header class="hero">
    <div class="brand">
      <div class="logo"></div>
      <div>
        <h1 class="title" data-i18n="title">AutoResearch 实验探索树</h1>
        <p class="subtitle" data-i18n="subtitle">自动读取 state/iterations.tsv、iteration 报告、figs 产物。点任意节点查看详情。</p>
      </div>
    </div>
    <input class="search" id="search" type="search" placeholder="搜索节点、指标或实验名..." data-i18n-placeholder="search">
    <div class="hero-actions">
      <label class="subtitle" for="lang-select" data-i18n="language">语言</label>
      <select class="lang-select" id="lang-select">
        <option value="zh">中文</option>
        <option value="ja">日本語</option>
        <option value="en">English</option>
      </select>
    </div>
  </header>
  <main class="shell">
    <div class="toolbar" id="toolbar"></div>
    <div class="toolbar-actions" id="toolbar-actions"></div>
    <div class="workspace">
      <div class="canvas" id="canvas">
        <div class="canvas-head">
          <div class="canvas-title">实验探索树</div>
          <div class="tree-controls">
            <select class="control-select" id="tree-filter" aria-label="筛选">
              <option value="all">筛选</option>
              <option value="success">成功</option>
              <option value="failed">失败</option>
              <option value="warning">警告</option>
              <option value="running">运行中</option>
            </select>
            <select class="control-select" id="tree-layout" aria-label="布局">
              <option value="ltr">布局: 从左到右</option>
            </select>
            <span class="zoom-group">
              <button class="control-btn" id="zoom-out" aria-label="缩小">−</button>
              <span class="zoom-label" id="zoom-label">100%</span>
              <button class="control-btn" id="zoom-in" aria-label="放大">＋</button>
            </span>
            <button class="control-btn" id="fit-view">适配视图</button>
            <button class="control-btn" id="fullscreen-tree">⛶</button>
          </div>
        </div>
        <div class="tree-body" id="tree-body">
          <svg class="tree" viewBox="0 0 940 760" preserveAspectRatio="xMidYMid meet" id="svg"></svg>
          <div class="legend">
            <div class="legend-title">状态图例</div>
            <div class="legend-row"><span class="legend-dot" style="background:#1D9E75"></span>成功</div>
            <div class="legend-row"><span class="legend-dot" style="background:#E24B4A"></span>失败</div>
            <div class="legend-row"><span class="legend-dot" style="background:#378ADD"></span>运行中</div>
            <div class="legend-row"><span class="legend-dot" style="background:#EF9F27"></span>警告</div>
            <div class="legend-row"><span class="legend-dot" style="background:#888780"></span>未知</div>
          </div>
        </div>
      </div>
      <aside class="detail-panel" id="detail-panel">
        <div class="empty-detail">选择左侧节点查看指标、分析和可视化实验图。</div>
      </aside>
    </div>
    <section class="compare-dock" id="compare-dock"></section>
  </main>
</div>
<div class="compare-overlay" id="compare-overlay" role="dialog">
  <div class="compare-head">
    <div>
      <div class="compare-title" data-i18n="compare">节点对比</div>
      <div class="subtitle" style="margin:2px 0 0" data-i18n="compareSubtitle">并排查看实验节点的指标和可视化图。</div>
    </div>
    <div style="display:flex;align-items:center;gap:12px">
      <input class="compare-range" id="compare-width-range" type="range" min="28" max="72" value="50" aria-label="调整左右宽度">
      <button class="icon-btn" id="compare-close" aria-label="关闭">×</button>
    </div>
  </div>
  <div class="compare-content" id="compare-content"></div>
</div>
<div class="toast" id="toast"></div>
<script id="tree-data" type="application/json">__DATA_JSON__</script>
<script>
(function(){
  const I18N={
    zh:{title:'AutoResearch 实验探索树',subtitle:'自动读取 state/iterations.tsv、iteration 报告、figs 产物。点任意节点查看详情。',search:'搜索节点、指标或实验名...',language:'语言',success:'成功',failed:'失败',warning:'警告',running:'运行中',info:'任务',export:'导出',compare:'节点对比',compareSubtitle:'并排查看实验节点的指标和可视化图。',openCompare:'打开对比',clearCompare:'清空对比',nodeSummary:'节点摘要',saveSummary:'保存节点摘要',downloadSummary:'下载摘要文件',addCompare:'加入对比',copy:'复制摘要',visuals:'Visualizations',artifacts:'Artifacts',analysis:'分析摘录',empty:'选择左侧节点查看指标、分析和可视化实验图。',compareEmpty:'请先选择至少两个节点加入对比。',compareReady:'已选择节点，可打开对比',compareAdded:'已加入对比',compareLimit:'最多对比 4 个节点；已移除最早选择。',saved:'摘要已保存',downloaded:'当前页面不能直接写磁盘，已下载摘要文件'},
    ja:{title:'AutoResearch 実験探索ツリー',subtitle:'state、iteration レポート、figs から自動生成。ノードをクリックして詳細を表示。',search:'ノード、指標、実験名を検索...',language:'言語',success:'成功',failed:'失敗',warning:'警告',running:'実行中',info:'タスク',export:'エクスポート',compare:'ノード比較',compareSubtitle:'実験ノードの指標と可視化を並べて確認します。',openCompare:'比較を開く',clearCompare:'比較をクリア',nodeSummary:'ノード要約',saveSummary:'ノード要約を保存',downloadSummary:'要約をダウンロード',addCompare:'比較に追加',copy:'要約をコピー',visuals:'Visualizations',artifacts:'Artifacts',analysis:'分析メモ',empty:'左のノードを選択すると、指標・分析・可視化を表示します。',compareEmpty:'比較するノードを2つ以上選択してください。',compareReady:'比較を開けます',compareAdded:'比較に追加しました',compareLimit:'比較は最大4ノードです。最初の選択を外しました。',saved:'要約を保存しました',downloaded:'このページから直接保存できないため、要約ファイルをダウンロードしました'},
    en:{title:'AutoResearch Experiment Tree',subtitle:'Generated from state, iteration reports, and figs. Click any node to inspect details.',search:'Search nodes, metrics, or experiment name...',language:'Language',success:'Success',failed:'Failed',warning:'Warning',running:'Running',info:'Task',export:'Export',compare:'Node comparison',compareSubtitle:'Compare metrics and visualizations across experiment nodes.',openCompare:'Open compare',clearCompare:'Clear compare',nodeSummary:'Node summary',saveSummary:'Save node summary',downloadSummary:'Download summary',addCompare:'Add to compare',copy:'Copy summary',visuals:'Visualizations',artifacts:'Artifacts',analysis:'Analysis excerpt',empty:'Select a node to inspect metrics, analysis, and visualizations.',compareEmpty:'Select at least two nodes for comparison.',compareReady:'Comparison is ready',compareAdded:'Added to comparison',compareLimit:'Comparison is limited to 4 nodes; removed the oldest selection.',saved:'Summary saved',downloaded:'Cannot write to disk from this page; downloaded the summary file'}
  };
  let lang=localStorage.getItem('dashboard_lang')||'zh';
  function t(k){return (I18N[lang]&&I18N[lang][k])||I18N.zh[k]||k}
  const STATUS_KEYS=['success','failed','warning','running','info'];
  const COLOR={success:'#1D9E75',failed:'#E24B4A',warning:'#EF9F27',running:'#378ADD',info:'#888780'};
  const data=JSON.parse(document.getElementById('tree-data').textContent);
  let state={selected:null,visualIndex:0,compare:[],compareVisualIndex:0,compareOpen:false,compareWidth:50,compareWeights:[],search:'',filter:'all',zoom:1,drag:null};
  function applyLanguage(){
    document.documentElement.lang=lang==='ja'?'ja':lang==='en'?'en':'zh-CN';
    document.querySelectorAll('[data-i18n]').forEach(el=>{el.textContent=t(el.getAttribute('data-i18n'))});
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el=>{el.setAttribute('placeholder',t(el.getAttribute('data-i18n-placeholder')))});
    const sel=document.getElementById('lang-select'); if(sel)sel.value=lang;
  }
  function allNodes(){const out=[data.task];for(const n of data.nodes){out.push(n);for(const s of (n.subs||[]))out.push(s)}return out}
  function findNode(id){return allNodes().find(n=>n.id===id)||null}
  function curve(x1,y1,x2,y2,gap){const xe=x2-(gap||0),cx1=x1+(xe-x1)*0.55,cx2=xe-(xe-x1)*0.55;return `M${x1},${y1} C${cx1},${y1} ${cx2},${y2} ${xe},${y2}`}
  function escapeHtml(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
  function renderToolbar(){
    const counts={}; for(const n of allNodes()) counts[n.status]=(counts[n.status]||0)+1;
    document.getElementById('toolbar').innerHTML=STATUS_KEYS.map(k=>`<span class="chip"><span>${t(k)}</span><strong>${counts[k]||0}</strong></span>`).join('');
    const cmp=state.compare.length?`<span class="chip">${t('compare')} ${state.compare.length}/4 · ${state.compare.map(id=>escapeHtml(findNode(id)?.name||id)).join(' vs ')}</span>`:'';
    const openDisabled=state.compare.length<2?' disabled':'';
    document.getElementById('toolbar-actions').innerHTML=`${cmp}
      <button class="btn" id="summary-open" style="flex:0">${t('nodeSummary')}</button>
      <button class="btn" id="export-html" style="flex:0">${t('export')}</button>
      <button class="btn primary" id="compare-open" style="flex:0"${openDisabled}>${t('openCompare')} (${state.compare.length})</button>
      ${state.compare.length?`<button class="btn" id="compare-clear" style="flex:0">${t('clearCompare')}</button>`:''}`;
    renderCompareDock();
  }
  function nodeSearchText(node){
    const m=node.meta||{};
    return [node.name,node.detail,node.metric,node.idea,node.reason,node.notes,m.exp_name,m.acc,m.loss,m.top5].join(' ').toLowerCase();
  }
  function nodeMatchesSearch(node){
    const q=state.search.trim().toLowerCase();
    return !q || nodeSearchText(node).includes(q);
  }
  function renderSvg(){
    const svg=document.getElementById('svg'); const t=data.task; let edges='',dots='';
    svg.style.transform=`scale(${state.zoom})`;
    svg.style.width=`${state.zoom*100}%`;
    svg.style.height=`${state.zoom*100}%`;
    const zl=document.getElementById('zoom-label'); if(zl)zl.textContent=`${Math.round(state.zoom*100)}%`;
    for(const n of data.nodes){
      edges+=`<path class="edge" d="${curve(t.x+10,t.y,n.x-10,n.y)}"/>`;
      for(const s of (n.subs||[])) edges+=`<path class="edge" d="${curve(n.x+10,n.y,s.x-10,s.y)}"/>`;
    }
    function drawDot(node){
      const isSel=state.selected===node.id, isCmp=state.compare.includes(node.id), c=COLOR[node.status]||COLOR.info;
      const statusMatch=state.filter==='all'||node.status===state.filter;
      const isDim=!nodeMatchesSearch(node)||!statusMatch;
      const pulse=node.status==='running'?`<circle cx="${node.x}" cy="${node.y}" r="9" fill="${c}" opacity="0.25"><animate attributeName="r" values="9;18;9" dur="1.8s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.4;0;0.4" dur="1.8s" repeatCount="indefinite"/></circle>`:'';
      return `<g data-id="${node.id}" class="${isSel?'sel':''} ${isCmp?'cmp':''} ${isDim?'dim':''}">
        ${pulse}<circle class="halo" cx="${node.x}" cy="${node.y}" r="22" fill="${c}"/>
        <circle class="dot-main" cx="${node.x}" cy="${node.y}" r="9" fill="${c}"/>
        <circle class="ring-sel" cx="${node.x}" cy="${node.y}" r="16" fill="none" stroke="${c}" stroke-width="1.2"/>
        <circle cx="${node.x}" cy="${node.y}" r="26" fill="transparent"/>
        <text class="node-label" x="${node.x}" y="${node.y+34}" text-anchor="middle">${escapeHtml(node.name)}</text>
        ${node.metric?`<text class="node-detail" x="${node.x}" y="${node.y+49}" text-anchor="middle">${escapeHtml(node.metric.slice(0,30))}</text>`:''}
      </g>`;
    }
    dots+=drawDot(t); for(const n of data.nodes){dots+=drawDot(n);for(const s of (n.subs||[]))dots+=drawDot(s)}
    svg.innerHTML=edges+dots;
  }
  function metaLines(node){
    const m=node.meta||{}; const keys=['exp_name','acc','loss','top5','best_epoch','gpu','pid','started_at','finished_at'];
    return keys.filter(k=>m[k]).map(k=>`${k}: ${m[k]}`).join('\n');
  }
  function renderDetailContent(){
    const node=findNode(state.selected); if(!node) return '';
    const status=node.status||'info';
    const links=(node.links||[]).map(l=>`<a class="link-btn" href="${escapeHtml(l.href)}" target="_blank" rel="noreferrer">${escapeHtml(l.label)}</a>`).join('');
    const visuals=node.visuals||[];
    if(state.visualIndex>=visuals.length) state.visualIndex=0;
    const activeVisual=visuals[state.visualIndex];
    const visualHtml=visuals.length?`
      <div class="section-h">${t('visuals')}</div>
      <div class="visual-tabs">
        ${visuals.map((v,i)=>`<button class="visual-tab ${i===state.visualIndex?'active':''}" data-visual-index="${i}">${escapeHtml(v.label)}</button>`).join('')}
      </div>
      <div class="visual-frame">
        <img src="${escapeHtml(activeVisual.href)}" alt="${escapeHtml(activeVisual.label)}">
      </div>
      <div class="visual-caption">${lang==='en'?'Previewed inline; use artifact links for source files.':lang==='ja'?'画像はこのパネル内でプレビューされます。元ファイルは Artifacts から開けます。':'图片在当前面板内预览；下方 Artifacts 可打开原始文件。'}</div>`:'';
    let analysisHtml='';
    if(node.analysis){
      const cls=status==='warning'?'warn':status==='running'?'running':'';
      const causes=(node.analysis.causes||[]).map(c=>`<li>${escapeHtml(c)}</li>`).join('');
      const fixes=(node.analysis.fixes||[]).map(f=>`<li>${escapeHtml(f)}</li>`).join('');
      analysisHtml=`<div class="section-h">${t('analysis')}</div><div class="analysis ${cls}">
        <div class="analysis-h">${escapeHtml(node.analysis.issue||'')}</div>
        ${causes?`<ul class="cause">${causes}</ul>`:''}
        ${fixes?`<div class="section-h" style="margin-top:8px">Next</div><ul class="cause">${fixes}</ul>`:''}
      </div>`;
    }
    return `<div class="detail-head"><div>
        <span class="pop-status ${status}"><i class="pulse"></i>${t(status)||status}</span>
        <h3 class="pop-name">${escapeHtml(node.name)}</h3>
        <p class="pop-detail">${escapeHtml(node.detail||'')}</p>
      </div><button class="icon-btn" id="b-close-detail" aria-label="关闭">×</button></div>
      ${node.metric?`<div class="pop-meta">${escapeHtml(node.metric)}</div>`:''}
      ${metaLines(node)?`<div class="pop-meta">${escapeHtml(metaLines(node))}</div>`:''}
      ${node.idea?`<div class="pop-line"><span class="lab">hypothesis ·</span>${escapeHtml(node.idea)}</div>`:''}
      ${node.reason?`<div class="pop-line"><span class="lab">decision ·</span>${escapeHtml(node.reason)}</div>`:''}
      ${node.notes?`<div class="pop-notes">${escapeHtml(node.notes)}</div>`:''}
      ${visualHtml}
      ${analysisHtml}
      ${renderNodeSummary(node)}
      ${links?`<div class="section-h">${t('artifacts')}</div><div class="links">${links}</div>`:''}
      <div class="pop-actions"><button class="btn primary" id="b-add-compare">${t('addCompare')}</button><button class="btn" id="b-copy">${t('copy')}</button></div>`;
  }
  function renderNodeSummary(node){
    const val=(data.user_summaries&&data.user_summaries[node.id])||'';
    const ph=lang==='en'?'Write your note for this node. Future analysis/proposal agents read node summaries first.':lang==='ja'?'このノードに対するメモを書いてください。次回の分析/提案で優先的に読まれます。':'为该节点添加你的摘要；下次 analyze/propose 会优先读取这些节点摘要。';
    return `<div class="section-h">${t('nodeSummary')}</div>
      <textarea class="summary-box" id="node-summary-text" placeholder="${escapeHtml(ph)}">${escapeHtml(val)}</textarea>
      <div class="pop-actions"><button class="btn primary" id="node-summary-save">${t('saveSummary')}</button><button class="btn" id="node-summary-download">${t('downloadSummary')}</button></div>`;
  }
  function renderDetail(){
    const panel=document.getElementById('detail-panel');
    document.querySelector('.page').classList.toggle('detail-open',!!state.selected);
    if(!state.selected){panel.innerHTML=`<div class="empty-detail">${t('empty')}</div>`;return}
    panel.innerHTML=renderDetailContent();
  }
  function addCompare(id){
    if(!id)return;
    state.compare=state.compare.filter(x=>x!==id);
    state.compare.push(id);
    let limited=false;
    if(state.compare.length>4){state.compare.shift();limited=true;}
    state.compareVisualIndex=0;
    state.compareWeights=[];
    renderToolbar();renderSvg();
    toast(limited?t('compareLimit'):(state.compare.length<2?t('compareAdded'):t('compareReady')));
  }
  function setZoom(next){
    state.zoom=Math.min(1.6,Math.max(.65,next));
    renderSvg();
  }
  function commonVisuals(a,b){
    const av=a?.visuals||[], bv=b?.visuals||[];
    const labels=av.map(v=>v.label).filter(l=>bv.some(x=>x.label===l));
    return labels.length?labels:[...(new Set([...av.map(v=>v.label),...bv.map(v=>v.label)]))];
  }
  function visualByLabel(node,label){return (node?.visuals||[]).find(v=>v.label===label)}
  function metricBlock(node){
    const m=node?.meta||{}; const keys=['acc','loss','top5','best_epoch'];
    return keys.filter(k=>m[k]).map(k=>`${k}: ${m[k]}`).join('\n') || (node?.metric||'');
  }
  function compactMetrics(node){
    const m=node?.meta||{};
    return [
      ['acc',  m.acc  ||'--'],
      ['loss', m.loss ||'--'],
      ['top5', m.top5 ||'--'],
    ];
  }
  function compareCard(node,label,weight){
    const v=visualByLabel(node,label);
    return `<div class="compare-card" style="flex:${weight} 1 0">
      <h3>${escapeHtml(node?.name||'未选择')}</h3>
      <p class="pop-detail">${escapeHtml(node?.detail||'')}</p>
      ${metricBlock(node)?`<div class="pop-meta">${escapeHtml(metricBlock(node))}</div>`:''}
      <div class="visual-frame">${v?`<img src="${escapeHtml(v.href)}" alt="${escapeHtml(v.label)}">`:`<div class="compare-empty">该节点没有 ${escapeHtml(label)} 图</div>`}</div>
      ${node?.notes?`<div class="pop-notes">${escapeHtml(String(node.notes).slice(0,420))}</div>`:''}
    </div>`;
  }
  function miniCard(node,index){
    const v=(node.visuals||[])[0];
    const metrics=compactMetrics(node).map(([k,v])=>`<div class="mini-metric"><span>${escapeHtml(k)}</span>${escapeHtml(v)}</div>`).join('');
    return `<article class="mini-card">
      <div class="mini-card-head">
          <span class="mini-index">${index+1}</span>
          <h3>${escapeHtml(node.name)}</h3>
          <span class="pop-status ${node.status||'info'}" style="margin:0"><i class="pulse"></i>${t(node.status||'info')}</span>
          <button class="icon-btn" data-remove-compare="${escapeHtml(node.id)}" aria-label="remove">×</button>
      </div>
      <div class="mini-metrics">${metrics}</div>
      <div class="mini-card-main">
        <div class="mini-card-img">${v?`<img src="${escapeHtml(v.href)}" alt="${escapeHtml(v.label)}">`:'<span class="pop-detail">No visual</span>'}</div>
        <div>
        ${node.notes?`<div class="mini-card-note">${escapeHtml(String(node.notes).slice(0,220))}</div>`:''}
        </div>
      </div>
    </article>`;
  }
  function renderCompareDock(){
    const dock=document.getElementById('compare-dock');
    if(!dock)return;
    const nodes=state.compare.map(id=>findNode(id)).filter(Boolean);
    dock.classList.add('show');
    if(!nodes.length){
      dock.innerHTML=`<div class="compare-dock-head">
        <div><div class="compare-dock-title">${t('compare')} (0)</div><div class="compare-dock-sub">拖拽树图节点到这里，或在节点详情中点击“加入对比”。</div></div>
        <div class="compare-dock-actions"><button class="btn primary" id="compare-open-dock" disabled>${t('openCompare')}</button><button class="btn" id="compare-clear-dock" disabled>${t('clearCompare')}</button></div>
      </div><div class="compare-drop-empty">Drop nodes here for comparison</div>`;
      return;
    }
    dock.innerHTML=`<div class="compare-dock-head">
      <div><div class="compare-dock-title">${t('compare')} (${nodes.length})</div><div class="compare-dock-sub">${t('compareSubtitle')}</div></div>
      <div class="compare-dock-actions"><button class="btn primary" id="compare-open-dock" ${nodes.length<2?'disabled':''}>${t('openCompare')}</button><button class="btn" id="compare-clear-dock">${t('clearCompare')}</button></div>
    </div><div class="compare-dock-grid">${nodes.map((node,i)=>miniCard(node,i)).join('')}</div>`;
  }
  function startNodeDrag(id,x,y){
    const node=findNode(id); if(!node)return;
    state.drag={id,startX:x,startY:y,active:false,ghost:null};
  }
  function updateNodeDrag(x,y){
    if(!state.drag)return;
    const dx=x-state.drag.startX, dy=y-state.drag.startY;
    if(!state.drag.active&&Math.hypot(dx,dy)>6){
      state.drag.active=true;
      const ghost=document.createElement('div');
      ghost.className='drag-ghost';
      ghost.textContent=findNode(state.drag.id)?.name||state.drag.id;
      document.body.appendChild(ghost);
      state.drag.ghost=ghost;
    }
    if(state.drag.active&&state.drag.ghost){
      state.drag.ghost.style.left=(x+12)+'px';
      state.drag.ghost.style.top=(y+12)+'px';
      const dock=document.getElementById('compare-dock');
      if(dock){
        const r=dock.getBoundingClientRect();
        const inside=x>=r.left&&x<=r.right&&y>=r.top&&y<=r.bottom;
        dock.classList.toggle('drop-target',inside);
      }
    }
  }
  function finishNodeDrag(x,y){
    if(!state.drag)return false;
    const wasActive=state.drag.active;
    const id=state.drag.id;
    if(state.drag.ghost)state.drag.ghost.remove();
    const dock=document.getElementById('compare-dock');
    let dropped=false;
    if(dock){
      const r=dock.getBoundingClientRect();
      dropped=wasActive&&x>=r.left&&x<=r.right&&y>=r.top&&y<=r.bottom;
      dock.classList.remove('drop-target');
    }
    state.drag=null;
    if(dropped)addCompare(id);
    return wasActive;
  }
  function normalizeCompareWeights(n){
    if(!n)return [];
    if(!Array.isArray(state.compareWeights)||state.compareWeights.length!==n){
      state.compareWeights=Array(n).fill(100/n);
      return state.compareWeights;
    }
    const sum=state.compareWeights.reduce((a,b)=>a+b,0)||100;
    state.compareWeights=state.compareWeights.map(w=>w*100/sum);
    return state.compareWeights;
  }
  function renderCompare(){
    const overlay=document.getElementById('compare-overlay');
    overlay.classList.toggle('show',state.compareOpen);
    if(!state.compareOpen)return;
    const content=document.getElementById('compare-content');
    if(state.compare.length<2){content.innerHTML=`<div class="compare-empty">${t('compareEmpty')}</div>`;return}
    const selected=state.compare.map(id=>findNode(id)).filter(Boolean);
    let labels=selected.reduce((acc,n,i)=>i===0?(n.visuals||[]).map(v=>v.label):acc.filter(l=>(n.visuals||[]).some(v=>v.label===l)),[]);
    if(!labels.length) selected.forEach(n=>(n.visuals||[]).forEach(v=>{if(!labels.includes(v.label))labels.push(v.label)}));
    if(state.compareVisualIndex>=labels.length)state.compareVisualIndex=0;
    const label=labels[state.compareVisualIndex]||'visual';
    const tabs=labels.map((l,i)=>`<button class="visual-tab ${i===state.compareVisualIndex?'active':''}" data-compare-visual-index="${i}">${escapeHtml(l)}</button>`).join('');
    const weights=normalizeCompareWeights(selected.length);
    const cards=selected.map((n,i)=>{
      const card=compareCard(n,label,weights[i]);
      const handle=i<selected.length-1?`<div class="compare-resizer" data-resizer-index="${i}" title="拖拽调整相邻两栏宽度"></div>`:'';
      return card+handle;
    }).join('');
    content.innerHTML=`<div style="padding:12px 18px 10px"><div class="visual-tabs">${tabs}</div></div>
      <div class="compare-body" id="compare-body">${cards}</div>`;
    const range=document.getElementById('compare-width-range');
    if(range){
      range.value=String(Math.round(weights[0]||50));
      range.style.display=selected.length===2?'':'none';
    }
    installCompareResizer();
  }
  function installCompareResizer(){
    const body=document.getElementById('compare-body');
    if(!body)return;
    const handles=Array.from(body.querySelectorAll('[data-resizer-index]'));
    handles.forEach(handle=>{
      let dragging=false;
      const idx=Number(handle.getAttribute('data-resizer-index')||0);
      const setWidth=(clientX)=>{
      const r=body.getBoundingClientRect();
        const cards=Array.from(body.querySelectorAll('.compare-card'));
        const left=cards[idx], right=cards[idx+1];
        if(!left||!right)return;
        const lr=left.getBoundingClientRect(), rr=right.getBoundingClientRect();
        const pairLeft=lr.left, pairRight=rr.right;
        const pairWidth=pairRight-pairLeft;
        if(pairWidth<=0)return;
        const pairWeight=state.compareWeights[idx]+state.compareWeights[idx+1];
        const leftPct=Math.min(88,Math.max(12,((clientX-pairLeft)/pairWidth)*100));
        state.compareWeights[idx]=pairWeight*leftPct/100;
        state.compareWeights[idx+1]=pairWeight-state.compareWeights[idx];
        cards.forEach((card,i)=>{card.style.flex=`${state.compareWeights[i]} 1 0`});
        if(state.compareWeights.length===2) state.compareWidth=state.compareWeights[0];
      const range=document.getElementById('compare-width-range');
        if(range&&state.compareWeights.length===2) range.value=String(Math.round(state.compareWeights[0]));
      };
      handle.addEventListener('pointerdown',e=>{dragging=true;handle.setPointerCapture(e.pointerId);e.preventDefault()});
      handle.addEventListener('pointermove',e=>{if(dragging)setWidth(e.clientX)});
      handle.addEventListener('pointerup',e=>{dragging=false;try{handle.releasePointerCapture(e.pointerId)}catch(_){}})
    });
  }
  function renderSummaryPanel(){
    state.selected='task';renderSvg();
    const panel=document.getElementById('detail-panel');
    document.querySelector('.page').classList.add('detail-open');
    panel.innerHTML=`<div class="detail-head"><div><span class="pop-status info"><i class="pulse"></i>${t('nodeSummary')}</span><h3 class="pop-name">${lang==='en'?'Global dashboard note':lang==='ja'?'グローバルメモ':'全局摘要'}</h3><p class="pop-detail">${lang==='en'?'Node notes are usually more useful; this legacy note is still supported.':lang==='ja'?'通常はノード別メモの方が便利ですが、従来の全局メモも保存できます。':'现在更推荐在具体节点中保存摘要；这里仍保留全局摘要。'}</p></div><button class="icon-btn" id="b-close-detail" aria-label="关闭">×</button></div>
      <textarea class="summary-box" id="summary-text" placeholder="写下你希望下一轮优先考虑的判断、约束、对比结论或下一步建议。">${escapeHtml(data.user_summary||'')}</textarea>
      <div class="pop-actions"><button class="btn primary" id="summary-save">${t('saveSummary')}</button><button class="btn" id="summary-download">${t('downloadSummary')}</button></div>
      <div class="pop-notes">提示：直接双击 HTML 打开时，浏览器不能写入磁盘；用 scripts/serve_dashboard.py 打开时可以直接保存。</div>`;
  }
  async function saveSummary(){
    const text=document.getElementById('summary-text')?.value||'';
    data.user_summary=text;
    try{
      const res=await fetch('/api/user-summary',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
      if(!res.ok)throw new Error('server rejected save');
      toast(t('saved'));
    }catch(_){
      downloadSummary(text);
      toast(t('downloaded'));
    }
  }
  function downloadSummary(text){
    const blob=new Blob([text],{type:'text/markdown;charset=utf-8'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download='user_summary.md';a.click();
    setTimeout(()=>URL.revokeObjectURL(url),500);
  }
  async function saveNodeSummary(){
    const node=findNode(state.selected); if(!node)return;
    const text=document.getElementById('node-summary-text')?.value||'';
    data.user_summaries=data.user_summaries||{}; data.user_summaries[node.id]=text;
    try{
      const res=await fetch('/api/node-summary',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({node_id:node.id,node_name:node.name,text})});
      if(!res.ok)throw new Error('server rejected save');
      toast(t('saved'));
    }catch(_){
      downloadSummary(`<!-- node:${node.id} -->\n## ${node.name}\n\n${text}\n`);
      toast(t('downloaded'));
    }
  }
  function toast(msg){const t=document.getElementById('toast');t.textContent=msg;t.classList.add('show');clearTimeout(t._timer);t._timer=setTimeout(()=>t.classList.remove('show'),2000)}
  document.addEventListener('click',function(e){
    if(state._suppressClick){state._suppressClick=false;return}
    const svgNode=e.target.closest('[data-id]');
    if(svgNode){state.selected=svgNode.getAttribute('data-id');state.visualIndex=0;renderSvg();renderDetail();return}
    const visual=e.target.closest('[data-visual-index]');
    if(visual){state.visualIndex=Number(visual.getAttribute('data-visual-index')||0);renderDetail();return}
    const cmpVisual=e.target.closest('[data-compare-visual-index]');
    if(cmpVisual){state.compareVisualIndex=Number(cmpVisual.getAttribute('data-compare-visual-index')||0);renderCompare();return}
    if(e.target.id==='b-close-detail'){state.selected=null;renderSvg();renderDetail();return}
    if(e.target.id==='b-add-compare'){addCompare(state.selected);return}
    if(e.target.id==='compare-open'){state.compareOpen=true;renderCompare();return}
    if(e.target.id==='compare-open-dock'){state.compareOpen=true;renderCompare();return}
    if(e.target.id==='compare-close'){state.compareOpen=false;renderCompare();return}
    if(e.target.id==='compare-clear'){state.compare=[];state.compareWeights=[];state.compareOpen=false;renderToolbar();renderSvg();renderCompare();return}
    if(e.target.id==='compare-clear-dock'){state.compare=[];state.compareWeights=[];state.compareOpen=false;renderToolbar();renderSvg();renderCompare();return}
    const removeCompare=e.target.closest('[data-remove-compare]');
    if(removeCompare){state.compare=state.compare.filter(id=>id!==removeCompare.getAttribute('data-remove-compare'));state.compareWeights=[];renderToolbar();renderSvg();renderCompare();return}
    if(e.target.id==='compare-width-range'){return}
    if(e.target.id==='export-html'){window.print();return}
    if(e.target.id==='zoom-in'){setZoom(state.zoom+.1);return}
    if(e.target.id==='zoom-out'){setZoom(state.zoom-.1);return}
    if(e.target.id==='fit-view'){state.zoom=1;renderSvg();document.getElementById('tree-body')?.scrollTo({left:0,top:0,behavior:'smooth'});return}
    if(e.target.id==='fullscreen-tree'){document.getElementById('canvas')?.requestFullscreen?.();return}
    if(e.target.id==='summary-open'){renderSummaryPanel();return}
    if(e.target.id==='summary-save'){saveSummary();return}
    if(e.target.id==='summary-download'){downloadSummary(document.getElementById('summary-text')?.value||'');return}
    if(e.target.id==='node-summary-save'){saveNodeSummary();return}
    if(e.target.id==='node-summary-download'){const n=findNode(state.selected);downloadSummary(`<!-- node:${n?.id||'node'} -->\n## ${n?.name||'Node'}\n\n${document.getElementById('node-summary-text')?.value||''}\n`);return}
    if(e.target.id==='b-copy'){const n=findNode(state.selected);navigator.clipboard&&navigator.clipboard.writeText(`${n.name} ${n.metric||''} ${n.detail||''}`);toast('已复制摘要');return}
  });
  document.addEventListener('input',function(e){
    if(e.target.id==='compare-width-range'){
      state.compareWidth=Number(e.target.value||50);
      if(state.compare.length===2){
        state.compareWeights=[state.compareWidth,100-state.compareWidth];
        const cards=Array.from(document.querySelectorAll('#compare-body .compare-card'));
        cards.forEach((card,i)=>{card.style.flex=`${state.compareWeights[i]} 1 0`});
      }
    }
    if(e.target.id==='search'){
      state.search=e.target.value||'';
      renderSvg();
    }
    if(e.target.id==='tree-filter'){
      state.filter=e.target.value||'all';
      renderSvg();
    }
  });
  document.addEventListener('pointerdown',function(e){
    const svgNode=e.target.closest('[data-id]');
    if(svgNode)startNodeDrag(svgNode.getAttribute('data-id'),e.clientX,e.clientY);
  });
  document.addEventListener('pointermove',function(e){updateNodeDrag(e.clientX,e.clientY)});
  document.addEventListener('pointerup',function(e){
    const dragged=finishNodeDrag(e.clientX,e.clientY);
    if(dragged)state._suppressClick=true;
  });
  document.getElementById('lang-select').addEventListener('change',e=>{lang=e.target.value;localStorage.setItem('dashboard_lang',lang);applyLanguage();renderToolbar();renderDetail();renderCompare()});
  document.getElementById('tree-filter').addEventListener('change',e=>{state.filter=e.target.value||'all';renderSvg()});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){if(state.compareOpen){state.compareOpen=false;renderCompare()}else if(state.selected){state.selected=null;renderSvg();renderDetail()}}});
  applyLanguage();renderToolbar();renderSvg();
})();
</script>
</body>
</html>
"""


def write_html(tree: dict[str, Any], out_path: Path) -> None:
    payload = json.dumps(tree, ensure_ascii=False)
    html_text = HTML_TEMPLATE.replace("__DATA_JSON__", html.escape(payload, quote=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")


def _iter_nodes(tree: dict[str, Any]):
    yield tree["task"]
    for group in tree.get("nodes", []):
        yield group
        for child in group.get("subs", []):
            yield child


def _resolve_href(out_path: Path, href: str) -> Path | None:
    if not href or re.match(r"^[a-zA-Z]+://", href):
        return None
    direct = (out_path.parent / href).resolve()
    if direct.exists():
        return direct
    stripped = href
    while stripped.startswith("../"):
        stripped = stripped[3:]
    root_relative = (ROOT / stripped).resolve()
    if root_relative.exists():
        return root_relative
    return direct


def bundle_assets(tree: dict[str, Any], out_path: Path, *,
                  exclude_csv: bool = True, write_manifest: bool = False,
                  embed_images: bool = True) -> Path:
    """Copy all linked artifacts into a folder next to the HTML and rewrite hrefs.

    Optimizations:
      • Content-hash dedup — two source files with identical bytes share ONE
        copy in the bundle. On large iter sets, this can save tens of MB.
      • Visible images are embedded as data: URIs by default, so a single
        index.html is self-contained (handy when you want to email or upload
        the dashboard somewhere). Pass embed_images=False to keep them as
        external asset files (smaller HTML, asset/ dir required to deploy).
      • CSVs (per_class.csv, per_class_delta*.csv) are skipped by default —
        they're data dumps, not web assets, and are still accessible from the
        repo if someone really wants them.
      • The unused manifest.json is no longer written.
    """
    asset_dir = out_path.parent / ("assets" if out_path.name == "index.html" else f"{out_path.stem}_assets")
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}      # cache key → bundled relpath
    by_hash: dict[str, str] = {}     # content-hash → bundled relpath

    def copy_one(href: str, node_id: str, *, embed: bool = False) -> str:
        """Resolve href → bundled relpath. Reads the source bytes from the
        working tree if present, otherwise falls back to autoresearch/iter-NNN
        branch via `git show` (because git_iter_commit.sh only restores the
        most-recent iter's outputs to main's working tree)."""
        # CSV bundling is opt-in to keep dashboard size bounded (the demo
        # ships per_class.csv but on large iter sets these add up).
        if exclude_csv and href.lower().endswith(".csv"):
            return href

        src_path = _resolve_href(out_path, href)
        src_bytes: bytes | None = None
        src_name = ""
        cache_key = ""

        if src_path is not None and src_path.exists() and src_path.is_file():
            try:
                src_bytes = src_path.read_bytes()
                src_name = src_path.name
                cache_key = str(src_path)
            except OSError:
                src_bytes = None

        if src_bytes is None:
            # Working-tree miss — try the iter branch. Path patterns we know
            # are committed on autoresearch/iter-NNN: figs/iter_NNN/* and
            # logs/iteration_NNN.md.
            m = re.search(r"(figs/iter_(\d+)/[^?#]+|logs/iteration_(\d+)\.md)", href)
            if m and src_path is not None:
                rel_in_repo = m.group(1)
                iter_pad = m.group(2) or m.group(3)
                blob = _git_show_blob(f"autoresearch/iter-{iter_pad}", rel_in_repo)
                if blob is not None:
                    src_bytes = blob
                    src_name = Path(rel_in_repo).name
                    cache_key = f"git:autoresearch/iter-{iter_pad}:{rel_in_repo}"

        if src_bytes is None:
            return href  # genuinely missing; leave broken href so the bug is visible

        # Embed-mode: return a data: URI directly instead of writing to assets/.
        # Only used for "visuals" (the inline <img>s on each node card), not
        # for "links" (the download buttons), since data: URIs aren't great
        # for "right-click → save as".
        is_image = Path(src_name).suffix.lower() in IMAGE_EXTS
        if embed and embed_images and is_image:
            mime = mimetypes.guess_type(src_name)[0] or "application/octet-stream"
            encoded = base64.b64encode(src_bytes).decode("ascii")
            return f"data:{mime};base64,{encoded}"

        if cache_key in copied:
            return copied[cache_key]

        digest = hashlib.md5(src_bytes).hexdigest()
        if digest in by_hash:
            copied[cache_key] = by_hash[digest]
            return by_hash[digest]

        safe_node = re.sub(r"[^A-Za-z0-9_.-]+", "_", node_id)
        node_dir = asset_dir / safe_node
        node_dir.mkdir(parents=True, exist_ok=True)
        dst = node_dir / src_name
        dst.write_bytes(src_bytes)
        rel = dst.relative_to(out_path.parent).as_posix()
        copied[cache_key] = rel
        by_hash[digest] = rel
        return rel

    # Drop link-list entries that point at unbundled CSV files (avoid broken buttons)
    def keep_or_drop(item: dict, new_href: str) -> bool:
        if exclude_csv and new_href.endswith(".csv"):
            return False                # filter from list
        item["href"] = new_href
        return True

    def dedup_by_href(items: list[dict]) -> list[dict]:
        """Drop entries whose href was already seen — keeps the FIRST label
        encountered. This collapses redundant pairs like
        attention/attention_maps or t-SNE/attr-t-SNE that the source repo
        stores as byte-identical duplicates."""
        seen: set[str] = set()
        out: list[dict] = []
        for it in items:
            h = it.get("href", "")
            if h in seen:
                continue
            seen.add(h)
            out.append(it)
        return out

    for node in _iter_nodes(tree):
        node_id = str(node.get("id", "node"))
        new_links = []
        for item in node.get("links", []):
            # Links (download buttons) — keep as external files for right-click-save.
            new_href = copy_one(str(item.get("href", "")), node_id, embed=False)
            if keep_or_drop(item, new_href):
                new_links.append(item)
        node["links"] = dedup_by_href(new_links)
        new_visuals = []
        for item in node.get("visuals", []):
            # Visuals (inline previews) — embed as data: URIs when possible
            # so the HTML is self-contained (only matters when embed_images=True).
            new_href = copy_one(str(item.get("href", "")), node_id, embed=True)
            if keep_or_drop(item, new_href):
                new_visuals.append(item)
        node["visuals"] = dedup_by_href(new_visuals)

    if write_manifest:
        manifest = {
            "generated_for": out_path.name,
            "asset_count": len({v for v in copied.values()}),
            "assets": sorted(set(copied.values())),
        }
        (asset_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return asset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=str(ROOT / "state" / "iterations.tsv"), help="Path to iterations.tsv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output HTML path")
    parser.add_argument("--no-bundle-assets", action="store_true", help="Do not copy linked resources next to the HTML")
    parser.add_argument("--include-csv", action="store_true", help="Include per_class*.csv data dumps in the bundle (default: skip — saves space)")
    parser.add_argument("--write-manifest", action="store_true", help="Write a manifest.json next to the bundle (default: skip — currently unused by the page)")
    parser.add_argument("--external-images", action="store_true", help="Keep image previews as external asset files instead of embedding them as data: URIs into the HTML (default: embed — produces a self-contained index.html)")
    args = parser.parse_args()

    rows = read_state(Path(args.state))
    tree = build_tree(rows)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    asset_dir = None
    if not args.no_bundle_assets:
        asset_dir = bundle_assets(tree, out_path,
                                  exclude_csv=not args.include_csv,
                                  write_manifest=args.write_manifest,
                                  embed_images=not args.external_images)
    write_html(tree, out_path)
    print(f"Wrote experiment tree webpage: {out_path}")
    if asset_dir is not None:
        print(f"Bundled assets: {asset_dir}")


if __name__ == "__main__":
    main()
