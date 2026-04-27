#!/usr/bin/env bash
# Driver — called on a schedule (tmux: `while true; do bash loop.sh; sleep 300; done`)
# Each tick does at most ONE unit of work, in this order:
#   1. Reap finished trainings (update state via FINISH-line + mtime heartbeat)
#   2. If any "completed" but "not analyzed" → analyze ONE via claude -p (BLOCKS new
#      proposals; ensures every experiment is analyzed before its successor is proposed)
#   3. Stopping criteria (configurable via AUTORES_MAX_ITERATIONS,
#      AUTORES_RECENT_FAIL_WINDOW/LIMIT, AUTORES_TARGET_METRIC/DIRECTION)
#   4. Concurrent-training cap (MAX_CONCURRENT)
#   5. Else, if GPU free → propose+launch next experiment via claude -p
#   6. Else: exit

set -u
cd "$(dirname "$0")"

# Optional per-tick env overrides — edit state/.consensus.env (or
# legacy .crossval.env) to change AUTORES_CONSENSUS_* / AUTORES_CROSSVAL_*
# without restarting the tmux loop. Picks up on the NEXT tick.
# Sourced BEFORE any logic so even sentinel/lock paths can be overridden.
if [ -f state/.consensus.env ]; then
    set -a
    # shellcheck disable=SC1091
    . state/.consensus.env
    set +a
fi
if [ -f state/.crossval.env ]; then
    set -a
    # shellcheck disable=SC1091
    . state/.crossval.env
    set +a
fi
# Onboarding choices (PYTHON, AUTORES_DATA_ROOT, AUTORES_SKIP_GPUS,
# WANDB_PROJECT, AUTORES_GIT_AUTOPUSH) — written by scripts/first_launch_setup.sh.
if [ -f state/.env ]; then
    set -a
    # shellcheck disable=SC1091
    . state/.env
    set +a
fi

# Host allowlist — the autoresearch directory lives on CIFS shared across
# multiple hosts. If more than one host has a while-true driver ticking,
# they race: non-owning hosts cannot see the owner's /proc/<pid>, so they
# falsely mark `running` rows as `completed` and trigger premature analysis.
# To prevent this, a tick proceeds ONLY if a sentinel file matching this
# host's tag exists.
#
# REQUIRED: export AUTORES_HOST_TAG=<unique-string> before running. The
# value is arbitrary — pick whatever distinguishes this host on this CIFS
# (e.g., "server1", "gpu4node", "tokyo-a6000"). The loop refuses to run
# without an explicit tag, since auto-detected hostname can collide across
# sibling containers.
#
# Setup on each host:
#   export AUTORES_HOST_TAG=<pick-a-name>
#   touch state/.loop.enabled.$AUTORES_HOST_TAG
#   # then run the while-true loop (env var MUST be exported in tmux too)
# Disable this host: `rm state/.loop.enabled.$AUTORES_HOST_TAG`
# Disable all hosts: `rm state/.loop.enabled.*`
if [ -z "${AUTORES_HOST_TAG:-}" ]; then
    printf '[%s] loop tick refused — AUTORES_HOST_TAG not set (export a unique tag per host)\n' \
        "$(date '+%F %T')" >> logs/driver.log
    exit 0
fi
_SENTINEL="state/.loop.enabled.${AUTORES_HOST_TAG}"
if [ ! -f "$_SENTINEL" ]; then
    printf '[%s] loop tick skipped on tag %s — no sentinel at %s\n' \
        "$(date '+%F %T')" "$AUTORES_HOST_TAG" "$_SENTINEL" >> logs/driver.log
    exit 0
fi

# Resolve claude CLI location — on servers where the VS Code Claude Code extension
# is installed, the binary lives under a version-pinned extension dir rather than
# /usr/local/bin. Pick the newest version by semver-sort so extension upgrades
# don't break the loop.
if ! command -v claude >/dev/null 2>&1; then
    _CLAUDE_DIR=$(ls -d /root/.vscode-server/extensions/anthropic.claude-code-*-linux-x64/resources/native-binary 2>/dev/null | sort -V | tail -1)
    if [ -n "$_CLAUDE_DIR" ] && [ -x "$_CLAUDE_DIR/claude" ]; then
        export PATH="$_CLAUDE_DIR:$PATH"
    fi
fi

# Serialize ticks within a single host — prevents two concurrent invocations
# from both computing NEXT_ITER from the same LAUNCHED count and both launching
# on the same GPU (what duplicated iter003 on 2026-04-24 and OOM-killed both).
#
# Lockfile lives on LOCAL /tmp, not on CIFS — CIFS often fails flock() with
# EACCES depending on mount opts / server support, and cross-host serialization
# is not the job of this lock anyway (that's what the sentinel guard above
# handles). One-file-per-autoresearch-tree key ensures independence when the
# same container services multiple projects.
_LOCK_KEY=$(echo -n "$(pwd -P)" | md5sum | awk '{print $1}' | cut -c1-12)
LOCK="/tmp/autores.${_LOCK_KEY}.lock"
exec 9>"$LOCK" 2>/dev/null || {
    printf '[%s] loop tick failed to open lockfile %s\n' \
        "$(date '+%F %T')" "$LOCK" >> logs/driver.log
    exit 0
}
if ! flock -n 9 2>/dev/null; then
    printf '[%s] loop tick skipped — another tick is already running\n' \
        "$(date '+%F %T')" >> logs/driver.log
    exit 0
fi

LOG=logs/driver.log
touch "$LOG"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# -------------------------------------------------
# Helper · regenerate the experiment-tree dashboard in the background.
#
# Triggered after every analyze tick (state.tsv just gained a fresh
# `analyzed` row + new figs/) and before every STOP exit (final snapshot).
# Safe to call mid-tick: the generator is read-only on the loop's working
# state and writes only to docs/autoresearch_dashboard/. The loop tick
# returns immediately; the regen runs detached.
#
# 9>&- closes the inherited flock fd in the child — without it, a 30-60 s
# regen would keep the per-tick lock held and block all subsequent ticks
# (same hazard as the consensus spawn; see operational lessons #10).
#
# Disable globally:    export AUTORES_DASHBOARD_ENABLED=0
# Skip when missing:   if the generator script vanished, this is a no-op
# -------------------------------------------------
_regen_dashboard_bg() {
    if [ "${AUTORES_DASHBOARD_ENABLED:-1}" != "1" ]; then
        return 0
    fi
    if [ ! -x scripts/generate_experiment_tree_web.sh ]; then
        return 0
    fi
    log "spawning dashboard regen in background (reason: ${1:-unspecified})"
    setsid nohup bash scripts/generate_experiment_tree_web.sh \
        </dev/null >>"$LOG" 2>&1 9>&- &
    disown 2>/dev/null || true
}

# -------------------------------------------------
# Step 0: sanity
# -------------------------------------------------
if ! command -v claude >/dev/null 2>&1; then
    log "ERROR: claude CLI not found in PATH"; exit 10
fi
test -f program.md || { log "ERROR: program.md missing"; exit 11; }
test -f state/iterations.tsv || { log "ERROR: state/iterations.tsv missing"; exit 12; }

# -------------------------------------------------
# Step 0.5: first-launch onboarding gate
#
# Fires ONLY on a truly fresh repo: state/iterations.tsv has just the header
# (no data rows) AND no state/.onboarding_done sentinel. In that case the loop
# halts with clear instructions pointing the user at scripts/first_launch_setup.sh,
# which interactively asks about Python env, data root, GPU skip-list, wandb,
# and git/GitHub remote (all strongly recommended) and writes the sentinel
# when finished.
#
# Once the sentinel exists OR any iteration row exists, this gate is a no-op
# — already-running loops with state are completely unaffected.
# -------------------------------------------------
_HAS_ITER_ROW=$(awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/ {c++} END{print c+0}' state/iterations.tsv)
if [ ! -f state/.onboarding_done ] && [ "$_HAS_ITER_ROW" -eq 0 ]; then
    log "first-launch detected — onboarding required before loop can tick"
    log "  run interactively:  bash scripts/first_launch_setup.sh"
    log "  it will ask about Python env / data root / GPU skip / wandb / git"
    log "  (all strongly recommended; you may decline any individual one)"
    exit 0
fi

# -------------------------------------------------
# Step 1: reap — mark trainings as completed/failed when they actually end.
#
# CROSS-HOST SAFE: do NOT rely on /proc/<pid> alone, because on a CIFS-shared
# autoresearch dir the row's pid may belong to a different host. Instead,
# use the training log as a shared heartbeat that is visible to any host:
#
#   (a) If the log file contains a "[iter NNN] FINISH ... rc=<N>" line,
#       the training has definitively ended — mark completed (rc==0) or
#       failed (rc!=0). run_experiment.sh's bg block writes this line
#       unconditionally right after python3 exits.
#   (b) Else if the log mtime is stale beyond REAP_HEARTBEAT_TIMEOUT seconds
#       (default 900 = 15 min, well beyond the ~10 min loop period and any
#       reasonable eval-only stall), assume the training died without
#       finalizing — mark completed with the current timestamp.
#   (c) Else, the log is fresh — leave the row as `running`. (Also, as a
#       fast path, if /proc/<pid> is alive locally we skip (b) and (c).)
# -------------------------------------------------
python3 - <<'PY'
import os, pathlib, re, time, glob
tsv = pathlib.Path('state/iterations.tsv')
lines = tsv.read_text().splitlines()
if len(lines) <= 1:
    raise SystemExit(0)
header, rows = lines[0], [l.split('\t') for l in lines[1:]]

REAP_HEARTBEAT_TIMEOUT = int(os.environ.get('REAP_HEARTBEAT_TIMEOUT', '900'))
FINISH_RE = re.compile(r'\[iter\s+(\d+)\]\s+FINISH\s+([\d\-: ]+)\s+rc=(-?\d+)')
now = time.time()
changed = False

def log_path_for(iter_num, exp_name):
    iter_pad = f'{int(iter_num):03d}'
    pattern = f'logs/exp_{iter_pad}_*.log'
    matches = glob.glob(pattern)
    if not matches:
        # Very old layout fallback — try without iter prefix
        alt = glob.glob(f'logs/exp_*{exp_name}*.log')
        return alt[0] if alt else None
    # If multiple, prefer one matching exp_name exactly
    for p in matches:
        if exp_name in p:
            return p
    return matches[0]

for r in rows:
    if len(r) < 10:
        r += [''] * (10 - len(r))
    if r[1] != 'running':
        continue

    pid, exp_name, iter_num = r[5], r[2], r[0]

    # Fast path: pid is alive in THIS host's /proc → definitely still running.
    if pid.isdigit() and pathlib.Path(f'/proc/{pid}').exists():
        continue

    log_file = log_path_for(iter_num, exp_name)
    if log_file is None or not os.path.exists(log_file):
        # No log file at all — could be a very fresh launch that hasn't
        # written yet; be conservative and leave as running.
        continue

    # CIFS caching: os.stat / os.path.getmtime alone can return a STALE
    # mtime (verified empirically — cold getmtime was ~50 min behind the
    # actual last-write). An explicit read forces the CIFS client to
    # refresh its metadata cache. So we combine the tail read (for FINISH
    # detection) AND mtime fetch in the same open(), which guarantees the
    # mtime we see is fresh.
    finish_rc = None
    finish_ts = None
    mtime = None
    try:
        with open(log_file, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 16384))
            tail = f.read().decode('utf-8', errors='replace')
            # fstat on the open fd also forces a fresh server round-trip.
            mtime = os.fstat(f.fileno()).st_mtime
        for line in reversed(tail.splitlines()):
            m = FINISH_RE.search(line)
            if m:
                finish_ts, finish_rc = m.group(2).strip(), int(m.group(3))
                break
    except OSError:
        # Can't read the log (CIFS orphan-inode bug — happens when
        # working-tree files get severed by stash/checkout while train.py
        # holds an open fd). Fallback: if the run's final.pth exists in
        # runs/, training has finished — mark completed
        # without a finish_ts (analysis will read final.pth directly).
        # If neither pid nor final.pth → still running, will retry next tick.
        # Note: `runs/<exp>/final.pth` (no subdir) is the demo's layout;
        # `runs/<exp>/*/final.pth` covers projects that nest by run-id.
        ckpt_glob = glob.glob(f'runs/{exp_name}/final.pth') + glob.glob(f'runs/{exp_name}/*/final.pth')
        if ckpt_glob:
            r[1] = 'completed'
            if r[7] == '':
                r[7] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            changed = True
        continue

    if finish_rc is not None:
        # Authoritative end signal in log.
        r[1] = 'completed' if finish_rc == 0 else 'failed'
        if r[7] == '':
            r[7] = finish_ts
        changed = True
        continue

    # No FINISH line. Use (freshly refreshed) log mtime as heartbeat.
    if mtime is None:
        continue
    age = now - mtime
    if age > REAP_HEARTBEAT_TIMEOUT:
        # Log has not been written in > timeout → training is dead.
        # Mark completed (not failed) because we don't know the exit code;
        # analysis tick will pick it up and investigate.
        if r[7] == '':
            r[7] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        r[1] = 'completed'
        changed = True

if changed:
    tsv.write_text(header + '\n' + '\n'.join('\t'.join(r) for r in rows) + '\n')
PY

# -------------------------------------------------
# Step 2: check if analysis pending (highest priority — runs BEFORE the
# stop-criteria check so that pending analyses still drain after the loop
# hits its launch cap. Without this ordering, a `completed` row that
# finishes after the launch cap is hit would never get analyzed:
# the STOP gate would exit before reaching the analysis branch.
# Furthermore, analysis BLOCKS new proposals — Step 5 (propose+launch)
# only runs when no `completed` row is waiting. This enforces the
# program-level invariant that every experiment is analyzed before its
# successor is proposed, so each new hypothesis can build on the prior
# iteration's documented findings rather than racing past them.)
#
# "completed" status = finished training but not yet analyzed
# "analyzed" status = Claude wrote iteration_NNN.md and marked it
# -------------------------------------------------
PENDING_ANALYSIS=$(awk -F'\t' 'NR>1 && $2 == "completed" {print $1; exit}' state/iterations.tsv)

if [ -n "$PENDING_ANALYSIS" ]; then
    log "Analyzing iteration $PENDING_ANALYSIS"
    ITER_PAD=$(printf '%03d' "$PENDING_ANALYSIS")
    PROMPT_FILE=$(mktemp)
    cat > "$PROMPT_FILE" <<EOF
You are in autoresearch loop mode. Iteration $ITER_PAD just finished training.

YOUR TASK: Analyze this single experiment per program.md §4 (mandatory visualizations) and §5 (log format).

CRITICAL ORDERING (write skeleton FIRST so timeout doesn't lose all work):
1. Read state/user_summaries.md and state/user_summary.md FIRST if they exist, then program.md, CLAUDE.md, last 3 logs/iteration_*.md, state/iterations.tsv. Treat user summaries as the user's highest-priority steering notes unless they conflict with HARD CONSTRAINTS.
2. Parse metrics from runs/<exp_name>/final.pth (or runs/<exp_name>/<subdir>/final.pth if your project nests by run-id) — torch load, ckpt['metrics'].
3. **WRITE logs/iteration_$ITER_PAD.md NOW** with §1-§4 + §6 verdict + §7 + §8 filled
   (use metrics from step 2; per_class.csv content can be sketched from prior knowledge or
   marked "TBD pending viz"). This is the SKELETON — must exist on disk after step 3.
4. **Update state/iterations.tsv NOW** (set \$2=analyzed, \$9=best_h, \$10=verdict). The
   defensive guard checks for the .md file's existence; once steps 3+4 are done, even if
   the rest times out, this iter is preserved.
5. Generate the 4 mandatory visualizations into figs/iter_$ITER_PAD/ — per_class.csv first
   (most informative for verdict), then gamma_sweep.png, then tsne.png, then attn.png last
   (heaviest, can be skipped if running short on time).
   IMPORTANT — VIZ GPU SELECTION: viz scripts auto-pick the least-loaded GPU when
   CUDA_VISIBLE_DEVICES is unset (skipping GPU 0 by default). Do NOT hard-code CUDA_VISIBLE_DEVICES.
   Copy per_class_delta_iter*.py from an existing one to inherit the auto-pick block.
6. Re-edit logs/iteration_$ITER_PAD.md §5 to fold in actual viz takeaways (replacing TBD).
7. Append a new "### Iteration $ITER_PAD" subsection to CLAUDE.md's "Documented findings" with a 1-paragraph lesson.
8. Exit (don't propose next experiment; the next loop tick handles that).

HARD CONSTRAINTS from program.md §HARD CONSTRAINTS still apply. No git push. Do not touch projects outside this repo's working tree.
EOF
    # timeout wraps claude -p so a hung / runaway subagent (previously seen
    # taking 20+ min and blocking the entire while-true loop) gets SIGTERM'd
    # after 20 min. Analysis ticks are usually 5-10 min; 20 min is generous.
    # Per-iter logfile (in addition to driver.log tee): captures FULL claude -p
    # transcript so a silent failure can be diagnosed (the 2026-04-25 iter039
    # incident produced ZERO transcript in driver.log because the | tee broke;
    # writing to a dedicated file via redirect is more robust on CIFS).
    ITER_TRANSCRIPT="logs/_analyze_${ITER_PAD}.transcript.log"
    timeout --signal=TERM 1800 claude -p "$(cat "$PROMPT_FILE")" \
        --model "${AUTORES_CLAUDE_MODEL:-claude-opus-4-7}" \
        --allowedTools "Bash,Read,Edit,Write,Glob,Grep" \
        --permission-mode acceptEdits \
        --max-turns 150 \
        > >(tee -a "$LOG" "$ITER_TRANSCRIPT") 2>&1
    _RC=$?
    rm -f "$PROMPT_FILE"
    if [ "$_RC" -eq 124 ]; then
        log "Analysis timeout (iter $ITER_PAD) — claude -p killed after 30 min (transcript: $ITER_TRANSCRIPT)"
    else
        log "Analysis dispatch for iter $ITER_PAD complete (rc=$_RC)"
    fi
    # Defensive (covers BOTH timeout AND clean-exit-no-report cases):
    # if claude -p didn't produce iteration_NNN.md, mark the row
    # analyzed | -1 | Bug to prevent infinite re-fire. The earlier patch
    # only handled rc=0 silent-failure, but iter042 (2026-04-25 13:20)
    # showed timeout (rc=124) hits the same loop-stuck pattern → guard
    # MUST run for both. Skip git_iter_commit + crossval if Bug-marked.
    if [ ! -f "logs/iteration_${ITER_PAD}.md" ]; then
        log "WARN: iteration_${ITER_PAD}.md NOT found after analyze (rc=$_RC) — marking Bug to prevent re-fire"
        python3 - "$PENDING_ANALYSIS" <<'PYEOF'
import csv, sys
iter_num = sys.argv[1]
rows = []
with open('state/iterations.tsv') as f:
    rows = list(csv.reader(f, delimiter='\t'))
for r in rows[1:]:
    if r[0] == iter_num:
        while len(r) < 10: r.append('')
        if r[1] == 'completed':
            r[1] = 'analyzed'; r[8] = '-1'; r[9] = 'Bug'
            print(f'  marked iter{iter_num} as Bug (silent or timeout claude -p failure)')
with open('state/iterations.tsv', 'w') as f:
    csv.writer(f, delimiter='\t', lineterminator='\n').writerows(rows)
PYEOF
        exit 0
    fi
    if [ "$_RC" -ne 0 ]; then
        log "WARN: claude rc=$_RC but iteration_${ITER_PAD}.md exists — proceeding to commit anyway"
    fi
    # Auto-commit the iteration's outputs to a per-iter branch.
    # Branch: autoresearch/iter-NNN (off main, NOT merged automatically).
    # User reviews + merges manually. See scripts/git_iter_commit.sh and
    # program.md HARD CONSTRAINT 7 for the policy.
    if [ -x scripts/git_iter_commit.sh ]; then
        bash scripts/git_iter_commit.sh "$PENDING_ANALYSIS" 2>&1 | tee -a "$LOG"
    fi
        # Fire-and-forget 5-cycle CONSENSUS workflow (replaces the old
        # parallel crossval, which was advisory-only). The consensus
        # workflow is a HARD GATE: propose phase BLOCKS until consensus.final.md
        # is written for this iter (see the propose-gate check below in step 5).
        # Sequential: main analyze (already done) → eval R1 → main revise →
        # eval R2 → main final. After 5 cycles, main's choice wins per user spec.
        # Enable with AUTORES_CONSENSUS_ENABLED=1; pick eval pool via
        # AUTORES_CONSENSUS_EVAL_AGENTS="claude,codex,gemini".
        # CRITICAL: 9>&- closes the inherited flock fd in the child. Without
        # it, consensus_iter.sh inherits fd 9 → keeps the loop tick lock held
        # for its full 13-min runtime → ALL subsequent loop ticks "skipped —
        # another tick is already running" → propose phase never fires →
        # GPUs sit idle. Discovered 2026-04-25 after iter47 second-round
        # consensus blocked propose for ~25 min. Same fix needed for legacy
        # crossval branch below.
        if [ "${AUTORES_CONSENSUS_ENABLED:-0}" = "1" ] && [ -x scripts/consensus_iter.sh ]; then
            log "spawning consensus workflow (eval agents: ${AUTORES_CONSENSUS_EVAL_AGENTS:-claude,codex,gemini}) in background"
            setsid nohup bash scripts/consensus_iter.sh "$PENDING_ANALYSIS" \
                </dev/null >>"$LOG" 2>&1 9>&- &
            disown 2>/dev/null || true
        elif [ "${AUTORES_CROSSVAL_ENABLED:-0}" = "1" ] && [ -x scripts/cross_validate_iter.sh ]; then
            # Legacy fallback: old parallel crossval (advisory only, no gate)
            log "spawning legacy crossval (agent=${AUTORES_CROSSVAL_AGENTS:-claude}) in background"
            setsid nohup bash scripts/cross_validate_iter.sh "$PENDING_ANALYSIS" \
                </dev/null >>"$LOG" 2>&1 9>&- &
            disown 2>/dev/null || true
        fi
    # Regenerate the experiment-tree dashboard so it reflects this iter's
    # fresh analyzed row + figs/. Background spawn so we don't delay the
    # exit; the next tick proceeds normally.
    _regen_dashboard_bg "post-analyze iter ${ITER_PAD}"
    exit 0
fi

# -------------------------------------------------
# Step 3: stopping criteria — only checked AFTER the analysis branch above,
# so completed-but-not-analyzed rows still drain to disk before we honor
# the launch cap. Without this ordering, a row that completes training
# after the cap is hit would sit in `completed` forever (the loop would
# exit at this gate every tick before reaching Step 2 analysis).
# -------------------------------------------------
# Configurable STOP thresholds (override per project via env vars):
#   AUTORES_MAX_ITERATIONS         iteration budget (default 80)
#   AUTORES_RECENT_FAIL_WINDOW     window length (default 5)
#   AUTORES_RECENT_FAIL_LIMIT      consecutive failures within window (default 3)
#   AUTORES_TARGET_METRIC          target value of state.tsv column 9
#                                  (default empty = disabled)
#   AUTORES_TARGET_DIRECTION       "max" (higher is better) | "min" (lower is better)
#                                  (default "max")
MAX_ITER="${AUTORES_MAX_ITERATIONS:-80}"
WIN="${AUTORES_RECENT_FAIL_WINDOW:-5}"
FAIL_LIMIT="${AUTORES_RECENT_FAIL_LIMIT:-3}"
TARGET="${AUTORES_TARGET_METRIC:-}"
DIRECTION="${AUTORES_TARGET_DIRECTION:-max}"

LAUNCHED=$(awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/ {c++} END{print c+0}' state/iterations.tsv)
FAILED_RECENT=$(awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/' state/iterations.tsv \
    | sort -k1n | tail -"$WIN" | awk -F'\t' '$10 == "Failure"' | wc -l)
# state.tsv column 9 is the iter's primary metric (named best_h / best_metric /
# best_acc / etc. depending on your launcher). Direction-aware best-so-far:
if [ "$DIRECTION" = "min" ]; then
    BEST_METRIC=$(awk -F'\t' 'NR>1 && $9 != "" {print $9}' state/iterations.tsv | sort -n  | head -1)
else
    BEST_METRIC=$(awk -F'\t' 'NR>1 && $9 != "" {print $9}' state/iterations.tsv | sort -rn | head -1)
fi

if [ "$LAUNCHED" -ge "$MAX_ITER" ]; then
    log "STOP: $MAX_ITER iterations launched."
    _regen_dashboard_bg "stop · launched cap"
    exit 0
fi
if [ "$FAILED_RECENT" -ge "$FAIL_LIMIT" ] && [ "$LAUNCHED" -ge "$WIN" ]; then
    log "STOP: $FAIL_LIMIT of last $WIN iterations failed."
    _regen_dashboard_bg "stop · failure rate"
    exit 0
fi
if [ -n "$TARGET" ] && [ -n "${BEST_METRIC:-}" ]; then
    HIT=$(awk -v b="$BEST_METRIC" -v t="$TARGET" -v d="$DIRECTION" \
        'BEGIN{ if (d=="min") print (b+0 <= t+0); else print (b+0 >= t+0); }')
    if [ "$HIT" = "1" ]; then
        log "STOP: target metric reached (best=$BEST_METRIC, target=$TARGET, direction=$DIRECTION)."
        _regen_dashboard_bg "stop · target reached"
        exit 0
    fi
fi

# -------------------------------------------------
# Step 4: concurrent-training cap (allow parallel across GPUs AND on same GPU)
# -------------------------------------------------
MAX_CONCURRENT="${MAX_CONCURRENT:-5}"   # override by exporting MAX_CONCURRENT
RUNNING=$(awk -F'\t' 'NR>1 && $2 == "running" {c++} END{print c+0}' state/iterations.tsv)
if [ "$RUNNING" -ge "$MAX_CONCURRENT" ]; then
    log "Already $RUNNING training(s) running (cap=$MAX_CONCURRENT). Waiting for next tick."
    exit 0
fi

# -------------------------------------------------
# Propose-gate: if consensus workflow is enabled, BLOCK propose until the
# most recent analyzed iter has its consensus.final.md written. This
# implements the user's hard requirement: only execute next iter when
# main+eval agents have reached agreement (or main has overridden after
# 5 cycles per consensus.final.md status).
# -------------------------------------------------
if [ "${AUTORES_CONSENSUS_ENABLED:-0}" = "1" ]; then
    # Find latest analyzed iter EXCLUDING Bug-verdict rows. Bug rows are
    # placeholders for analyze failures; consensus workflow can't run on
    # them (no iteration_NNN.md exists), so they should not block propose.
    LATEST_ANALYZED=$(awk -F'\t' 'NR>1 && $2 == "analyzed" && $1 ~ /^[0-9]+$/ && $10 != "Bug" {n=$1} END{print n+0}' state/iterations.tsv)
    if [ "$LATEST_ANALYZED" -gt 0 ]; then
        LA_PAD=$(printf '%03d' "$LATEST_ANALYZED")
        CFINAL="logs/iteration_${LA_PAD}.consensus.final.md"
        if [ ! -f "$CFINAL" ]; then
            log "Propose blocked — consensus.final.md not yet written for iter $LA_PAD (workflow may still be running; wait for next tick)"
            exit 0
        fi
        # Honor parse_consensus.py's documented contract: STATUS=PARSE_FAIL
        # means propose phase BLOCKS pending user intervention. Without this
        # check, a corrupt final.md (R5 + §8 both unparseable) would still
        # release the gate AND its placeholder NEXT_STEP would be fed to
        # propose-claude as BINDING — exactly the failure mode parse_consensus.py
        # was designed to flag. (Aligns gate with the docstring at lines 7-11.)
        CFINAL_STATUS=$(awk -F'=' '/^STATUS=/{print $2; exit}' "$CFINAL")
        if [ "$CFINAL_STATUS" = "PARSE_FAIL" ]; then
            log "Propose blocked — iter $LA_PAD consensus.final.md has STATUS=PARSE_FAIL (R5 + §8 both unparseable); user must inspect $CFINAL and either (a) fix STATUS to OVERRIDE_BY_MAIN with a usable next-step or (b) set AUTORES_CONSENSUS_ENABLED=0 in state/.consensus.env"
            exit 0
        fi
    fi
fi

# -------------------------------------------------
# Step 5: propose + launch next experiment
# -------------------------------------------------
# GPU selection: free-memory-only policy (no utilization check) — per user
# request we run in parallel as long as VRAM suffices, regardless of util.
# AUTORES_SKIP_GPUS: comma-separated container-local GPU indices to skip.
#   Default: "0" (GPU 0 has ~21GB neighbor on this host); matches run_experiment.sh.
# AUTORES_MIN_FREE_MB: minimum free VRAM on a GPU to consider it launchable
#   (default 24000 = 24 GB; matches run_experiment.sh after iter026-034 cascade fix).
# OOM-guard defaults (must match run_experiment.sh defaults):
#   SKIP_GPUS=0           — GPU 0 has a 21 GB persistent neighbor on this host
#   MIN_FREE_MEM_MB=24000 — covers mid-training peak with margin
# Without these defaults, propose phase would happily target GPU 0 even when
# run_experiment.sh would later reject it — wasting a propose tick on a config
# that can't launch. Setting both to align with run_experiment.sh.
SKIP_GPUS="${AUTORES_SKIP_GPUS-0}"  # OOM-guard: skip GPU 0 (21GB neighbor); matches run_experiment.sh
MIN_FREE_MEM_MB="${AUTORES_MIN_FREE_MB:-24000}"  # OOM-guard: bumped after iter026-034 cascade
# Pick the GPU with the MOST free memory (natural load-balancing across GPUs).
# Sort by memory.free DESC, then pick first eligible (not in skip list, passes threshold).
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free \
    --format=csv,noheader,nounits \
    | sort -t',' -k2 -rn \
    | awk -F', ' -v skip="$SKIP_GPUS" -v thresh="$MIN_FREE_MEM_MB" '
        BEGIN { n = split(skip, s, ","); for (i=1; i<=n; i++) if (s[i] != "") skipm[s[i]] = 1 }
        $2+0 >= thresh && !($1 in skipm) { print $1; exit }')
if [ -z "$FREE_GPU" ]; then
    log "No GPU with free mem >= ${MIN_FREE_MEM_MB} MB (skipped GPUs: ${SKIP_GPUS:-none}). Waiting for next tick."
    exit 0
fi

NEXT_ITER=$((LAUNCHED + 1))
NEXT_PAD=$(printf '%03d' "$NEXT_ITER")
log "Proposing iteration $NEXT_PAD (free GPU: $FREE_GPU)"

PROMPT_FILE=$(mktemp)
# If consensus workflow is enabled, the most recent analyzed iter has a
# consensus.final.md with the NEXT_STEP that this propose call MUST honor.
CONSENSUS_HINT=""
if [ "${AUTORES_CONSENSUS_ENABLED:-0}" = "1" ]; then
    PREV_ITER=$(awk -F'\t' 'NR>1 && $2 == "analyzed" && $1 ~ /^[0-9]+$/ && $10 != "Bug" {n=$1} END{print n+0}' state/iterations.tsv)
    if [ "$PREV_ITER" -gt 0 ]; then
        PREV_PAD=$(printf '%03d' "$PREV_ITER")
        CONSENSUS_FILE="logs/iteration_${PREV_PAD}.consensus.final.md"
        if [ -f "$CONSENSUS_FILE" ]; then
            # Read STATUS to calibrate hint authority. Gate above already blocks
            # PARSE_FAIL, so here we expect CONSENSUS or OVERRIDE_BY_MAIN.
            CHINT_STATUS=$(awk -F'=' '/^STATUS=/{print $2; exit}' "$CONSENSUS_FILE")
            case "$CHINT_STATUS" in
                CONSENSUS)
                    CONSENSUS_HINT="

CONSENSUS GUIDANCE (BINDING — do NOT pick a different hypothesis):
All eval agents agreed on the next-step in the previous iter's consensus.final.md
($CONSENSUS_FILE). Read that file FIRST and use its '## Final next-step' section
to drive your config + hypothesis. Only deviate if the consensus next-step is
impossible (e.g., requires a config field that doesn't exist) — in that case log
why and pick the closest viable variant.
"
                    ;;
                OVERRIDE_BY_MAIN)
                    CONSENSUS_HINT="

CONSENSUS GUIDANCE (STRONG — but with noted dissent):
The previous iter's consensus.final.md ($CONSENSUS_FILE) has STATUS=OVERRIDE_BY_MAIN,
meaning eval agents disagreed but main's '## Final next-step' won per project policy.
Read that file FIRST (including the dissent recorded in the R2 tally) and use the
'## Final next-step' section. You MAY surface the dissent in your hypothesis writeup
(§1) if it materially affects the falsification criterion, but you must still execute
the prescribed next-step.
"
                    ;;
                *)
                    # Unrecognized STATUS — proceed with weak hint, do not mark BINDING
                    CONSENSUS_HINT="

CONSENSUS GUIDANCE (WEAK — STATUS=$CHINT_STATUS unrecognized):
The previous iter's consensus.final.md ($CONSENSUS_FILE) reports an unrecognized
STATUS field. Read its '## Final next-step' section as a hint but exercise judgment;
if the content looks malformed, fall back to the standard frontier-pick prompt.
"
                    ;;
            esac
        fi
    fi
fi

cat > "$PROMPT_FILE" <<EOF
You are in autoresearch loop mode. It's time to propose iteration $NEXT_PAD.
$CONSENSUS_HINT
YOUR TASK: Propose ONE new experiment targeting SUN H improvement, create its config, and launch it via run_experiment.sh.

STEPS (in order):
1. Read state/user_summaries.md and state/user_summary.md FIRST if they exist, then program.md completely, then CLAUDE.md, then all files in logs/iteration_*.md (may be none on first iter). Treat user summaries as the user's highest-priority steering notes unless they conflict with HARD CONSTRAINTS or binding consensus guidance.
2. Review state/iterations.tsv for what's already been tried and their verdicts. IMPORTANT: rows with status="running" are currently training on other GPUs — treat them as "already taken" and do NOT duplicate their hypothesis. Pick something different.
3. Look at the "Exploration frontier" in program.md and the "Documented findings" in CLAUDE.md. Pick ONE hypothesis to try. Prefer high-priority items that haven't been tested yet and aren't currently running. You may propose something new if well-justified.
4. Create a new config at configs/ablation/iter${NEXT_PAD}_<short_name>.yaml. Its output_root MUST be under runs/. Base it on configs/cifar10_resnet34.yaml (or whatever your project's baseline config is) unless your hypothesis requires otherwise. Change ONE thing (or a small coherent set of things belonging to one mechanism).
5. If your change requires modifying code (src/<your_pkg>/*.py or train.py), do it now. Make the change targeted, not sweeping.
6. Launch via: bash run_experiment.sh configs/ablation/iter${NEXT_PAD}_<short_name>.yaml $NEXT_ITER
7. After the launch script returns, STOP. Do NOT wait for training to finish. Do NOT analyze. Those happen in subsequent loop ticks.

HARD CONSTRAINTS from program.md §HARD CONSTRAINTS still apply.
EOF
# Proposal is typically 3-5 min; 15 min timeout kills hung subagents that
# would otherwise block the entire while-true loop (seen on 2026-04-24 where
# one claude -p lingered 17+ min after iter008 was already launched).
timeout --signal=TERM 900 claude -p "$(cat "$PROMPT_FILE")" \
    --model "${AUTORES_CLAUDE_MODEL:-claude-opus-4-7}" \
    --allowedTools "Bash,Read,Edit,Write,Glob,Grep" \
    --permission-mode acceptEdits \
    --max-turns 60 \
    2>&1 | tee -a "$LOG"
_RC=$?
rm -f "$PROMPT_FILE"
if [ "$_RC" -eq 124 ]; then
    log "Proposal timeout (iter $NEXT_PAD) — claude -p killed after 15 min"
else
    log "Proposal dispatch for iter $NEXT_PAD complete (rc=$_RC)"
fi
exit 0
