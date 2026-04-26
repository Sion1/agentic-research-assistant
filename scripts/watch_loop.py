#!/usr/bin/env python3
"""watch_loop.py — live monitor for the autoresearch loop.

Refreshes a single-screen status board every N seconds. Shows: wrapper
status, sentinel/lock state, ledger summary, running iters, GPU usage,
recent driver-log activity, and pending consensus / dashboard regen jobs.

Usage:
  python3 scripts/watch_loop.py                  # 5 s refresh
  python3 scripts/watch_loop.py --interval 2     # 2 s refresh
  python3 scripts/watch_loop.py --once           # one snapshot, no loop

Read-only — never touches state.tsv, never spawns workers, never blocks
loop ticks. Safe to run anytime.
"""
from __future__ import annotations
import argparse, csv, datetime, os, pathlib, re, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
TSV  = ROOT / "state" / "iterations.tsv"
LOG  = ROOT / "logs" / "driver.log"


# ------------------------------------------------------------------ ANSI
def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb")
USE_COLOR = supports_color()
def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text
RED, GRN, YLW, BLU, GRY, BLD = "31", "32", "33", "34", "90", "1"


# ------------------------------------------------------------------ helpers
def run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def read_tsv() -> list[dict]:
    if not TSV.exists():
        return []
    with TSV.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r for r in rows if (r.get("iter") or "").strip().isdigit()]


def parse_iso(ts: str) -> datetime.datetime | None:
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def humanize(seconds: float) -> str:
    if seconds < 60:    return f"{int(seconds)}s"
    if seconds < 3600:  return f"{int(seconds/60)}m{int(seconds%60):02d}s"
    return f"{int(seconds/3600)}h{int((seconds%3600)/60):02d}m"


# ------------------------------------------------------------------ panels
def panel_header(width: int) -> str:
    title = "  AutoResearch Loop · live monitor  "
    bar = "─" * ((width - len(title)) // 2)
    return c(BLD, f"{bar}{title}{bar}") + f"   {datetime.datetime.now().strftime('%H:%M:%S')}"


def panel_wrapper() -> str:
    out = [c(BLD, "wrapper")]
    procs = run(["pgrep", "-af", "while true.*loop.sh"])
    procs = [p for p in procs.splitlines() if "claude-code" not in p and "watch_loop" not in p]
    if not procs:
        out.append("  " + c(RED, "✗ wrapper not running") + "  (start with `tmux new-session -d -s autores ...`)")
    else:
        line = procs[0]
        m = re.match(r"(\d+)\s+(.*)", line)
        if m:
            pid, cmd = m.group(1), m.group(2)
            out.append(f"  {c(GRN, '✓ alive')}  PID={c(BLD, pid)}  cmd={c(GRY, cmd[:90])}")
    # tick PID (mid-execution?)
    ticks = run(["pgrep", "-af", "bash loop.sh"])
    ticks = [t for t in ticks.splitlines() if "while true" not in t and "watch_loop" not in t]
    if ticks:
        out.append(f"  {c(YLW, 'tick in progress')}: {ticks[0][:100]}")
    return "\n".join(out)


def panel_sentinel_lock() -> str:
    out = [c(BLD, "sentinel · lock")]
    state_dir = ROOT / "state"
    sentinels = sorted(state_dir.glob(".loop.enabled.*")) if state_dir.exists() else []
    if not sentinels:
        out.append("  " + c(RED, "✗ no sentinel") + f"  (touch {state_dir}/.loop.enabled.<host_tag>)")
    else:
        for s in sentinels:
            out.append(f"  {c(GRN, '✓ sentinel')}  {s.name}")
    # fd holders on /tmp/autores.*.lock
    fd_check = run(["bash", "-c",
                    "ls -la /proc/*/fd/ 2>/dev/null | grep -oE '/tmp/autores\\.[a-z0-9]+\\.lock' | sort -u"])
    if fd_check:
        out.append(f"  {c(YLW, 'lock fd open')}  {fd_check}")
    return "\n".join(out)


def panel_ledger(rows: list[dict]) -> str:
    out = [c(BLD, f"state/iterations.tsv  ·  {len(rows)} iters total")]
    if not rows:
        out.append("  " + c(GRY, "(empty — no experiments launched yet)"))
        return "\n".join(out)
    counts: dict[str, int] = {}
    for r in rows:
        s = r.get("status", "")
        counts[s] = counts.get(s, 0) + 1
    summary = []
    for s, color in [("running", BLU), ("completed", YLW), ("analyzed", GRN), ("failed", RED)]:
        if counts.get(s):
            summary.append(c(color, f"{s}={counts[s]}"))
    other = sum(v for k, v in counts.items() if k not in {"running", "completed", "analyzed", "failed"})
    if other:
        summary.append(c(GRY, f"other={other}"))
    out.append("  " + "  ".join(summary))
    # best metric so far (column 9)
    metric_col = "best_metric" if "best_metric" in (rows[0].keys() if rows else []) else "best_h"
    nums = []
    for r in rows:
        v = r.get(metric_col, "")
        try:
            nums.append(float(v))
        except (ValueError, TypeError):
            pass
    if nums:
        out.append(f"  best so far ({metric_col}): {c(GRN, f'{max(nums):.4f}')}  · count with metric: {len(nums)}/{len(rows)}")
    # latest verdicts
    verdicts = [r.get("verdict", "") for r in rows[-5:] if r.get("verdict")]
    if verdicts:
        out.append(f"  last 5 verdicts: {' · '.join(verdicts)}")
    return "\n".join(out)


def panel_running(rows: list[dict]) -> str:
    out = [c(BLD, "running iters")]
    running = [r for r in rows if r.get("status") == "running"]
    if not running:
        out.append("  " + c(GRY, "(none)"))
        return "\n".join(out)
    now = datetime.datetime.now(datetime.timezone.utc)
    for r in running:
        started = parse_iso(r.get("started_at", ""))
        elapsed = humanize((now - started).total_seconds()) if started else "?"
        pid = r.get("pid", "?")
        alive = pathlib.Path(f"/proc/{pid}").exists() if pid.isdigit() else None
        alive_tag = c(GRN, "alive") if alive else c(YLW, "remote/unknown")
        out.append(f"  iter {r.get('iter','?'):>3}  "
                   f"exp={c(BLU, (r.get('exp_name','')[:40]))}  "
                   f"gpu={r.get('gpu','?')}  pid={pid} ({alive_tag})  "
                   f"elapsed={c(GRY, elapsed)}")
    return "\n".join(out)


def panel_gpu() -> str:
    out = [c(BLD, "GPU")]
    csv_text = run(["nvidia-smi",
                    "--query-gpu=index,memory.free,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"])
    if not csv_text:
        out.append("  " + c(GRY, "(no GPUs / nvidia-smi unavailable)"))
        return "\n".join(out)
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5: continue
        idx, free_mb, used_mb, total_mb, util = parts
        free_gb  = int(free_mb)  / 1024
        total_gb = int(total_mb) / 1024
        bar_w = 20
        used_frac = (int(used_mb) / max(int(total_mb), 1))
        filled = int(bar_w * used_frac)
        bar = "█" * filled + "░" * (bar_w - filled)
        bar_color = RED if used_frac > 0.85 else (YLW if used_frac > 0.5 else GRN)
        out.append(f"  GPU {idx}  [{c(bar_color, bar)}]  "
                   f"free {free_gb:5.1f}/{total_gb:.1f} GB  · util {util:>3}%")
    return "\n".join(out)


def panel_log() -> str:
    out = [c(BLD, "driver.log · last 8 informative lines")]
    if not LOG.exists():
        out.append("  " + c(GRY, "(no driver.log yet)"))
        return "\n".join(out)
    lines = LOG.read_text(errors="replace").splitlines()
    # filter spammy lines
    drop = ("loop tick refused", "loop tick skipped", "Already ", "No GPU with free")
    keep = [ln for ln in lines if not any(d in ln for d in drop)]
    for ln in keep[-8:]:
        # color-code
        if "ERROR" in ln or "FAIL" in ln:
            out.append("  " + c(RED, ln[:160]))
        elif "STOP:" in ln or "Bug" in ln:
            out.append("  " + c(YLW, ln[:160]))
        elif "spawning consensus" in ln or "Analyzing" in ln or "Proposing" in ln:
            out.append("  " + c(BLU, ln[:160]))
        else:
            out.append("  " + ln[:160])
    return "\n".join(out)


def panel_consensus() -> str:
    out = [c(BLD, "consensus jobs in flight")]
    procs = run(["pgrep", "-af", "consensus_iter"])
    procs = [p for p in procs.splitlines() if "watch_loop" not in p]
    if not procs:
        out.append("  " + c(GRY, "(none)"))
        return "\n".join(out)
    for line in procs[:5]:
        m = re.match(r"(\d+)\s+(.*?(?:consensus_iter\.sh\s+)(\d+))", line)
        if m:
            pid, _, iter_num = m.group(1), m.group(2), m.group(3)
            out.append(f"  iter {iter_num:>3}  pid={pid}")
        else:
            out.append(f"  {line[:120]}")
    return "\n".join(out)


# ------------------------------------------------------------------ main
def render_screen() -> str:
    rows = read_tsv()
    width = shutil.get_terminal_size((120, 30)).columns
    blocks = [
        panel_header(width),
        "",
        panel_wrapper(),
        "",
        panel_sentinel_lock(),
        "",
        panel_ledger(rows),
        "",
        panel_running(rows),
        "",
        panel_gpu(),
        "",
        panel_consensus(),
        "",
        panel_log(),
    ]
    return "\n".join(blocks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=5.0, help="refresh seconds (default 5)")
    ap.add_argument("--once", action="store_true", help="one snapshot then exit")
    args = ap.parse_args()

    if args.once:
        print(render_screen())
        return

    try:
        while True:
            sys.stdout.write("\033[H\033[2J" if USE_COLOR else "")  # clear screen
            sys.stdout.write(render_screen() + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
