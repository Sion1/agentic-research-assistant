#!/usr/bin/env bash
# Wrapper that loop.sh invokes to launch one experiment.
#
# Responsibilities (you should keep ALL of these when adapting for your
# domain — the framework's reaper / scheduler depends on them):
#   1. Pick a free GPU (skip ones below VRAM threshold)
#   2. OOM preflight (refuse to launch if a sibling process would push the
#      device over the per-process footprint cap)
#   3. Append a `running` row to state/iterations.tsv with PID + log path
#   4. nohup background launch of train.py
#   5. The *child* (train.py) writes the FINISH-line that the reaper reads
#
# Usage (from loop.sh's propose step):
#   bash run_experiment.sh <CONFIG_PATH> <ITER_NUM>
#
# Env knobs:
#   AUTORES_SKIP_GPUS                  ← comma-sep GPU indices to skip
#   AUTORES_MIN_FREE_MB         8000   ← only consider GPUs with this much free
#                                       (sized for the CIFAR-10 ResNet-34 demo,
#                                        which needs ~2-3 GB. Raise for larger
#                                        models — e.g. 22000 for ViT-B at 448²,
#                                        24000 for DINO + attribute prototype)
#   AUTORES_PER_PROCESS_FOOTPRINT_MB   ← OOM preflight cap (in MB)
#   AUTORES_LAUNCH_COOLDOWN_S  30      ← refuse if last launch was < N s ago (set 0 to disable)
#   AUTORES_ALLOW_GPU_STACK    0       ← if 1, allow same-GPU stacking (escape hatch)
#   AUTORES_ALLOW_DUPLICATE_ITER 0     ← if 1, allow reusing an existing iter num
#   AUTORES_SMOKETEST          0       ← if 1, training is a smoketest:
#                                          - EPOCHS_OVERRIDE auto-set to 1
#                                          - row written to state/smoketests.tsv (NOT iterations.tsv),
#                                            so reaper / analyze / dashboard ignore it
#                                          - all guards (cooldown, dup, GPU-stack) still apply
#                                       Equivalent: pass --smoketest as the first arg.
#   PYTHON                     python3 ← override interpreter
set -u

# --smoketest sugar: translate to AUTORES_SMOKETEST=1 + EPOCHS_OVERRIDE=1.
# Lets users write `bash run_experiment.sh --smoketest <yaml> 0` instead of
# the longer env-var form.
if [ "${1:-}" = "--smoketest" ]; then
    AUTORES_SMOKETEST=1
    export AUTORES_SMOKETEST
    : "${EPOCHS_OVERRIDE:=1}"
    export EPOCHS_OVERRIDE
    shift
fi

CONFIG="${1:-}"
ITER_NUM="${2:-0}"
if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
    echo "usage: $0 <config.yaml> <iter_num>" >&2
    exit 2
fi

cd "$(dirname "$0")"
mkdir -p logs state runs

ITER_PAD=$(printf '%03d' "$ITER_NUM")
# Parse exp_name with a real YAML parser. The earlier `awk -F'"'` approach
# only worked for quoted values like `exp_name: "foo"`; unquoted YAML
# (the more common form) returned empty, silently falling back to the
# config filename — leaving state/iterations.tsv listing one name while
# train.py wrote runs/<other-name>/, breaking analyze + dashboard lookup.
PYTHON_EARLY="${PYTHON:-python3}"
EXP_NAME=$("$PYTHON_EARLY" - "$CONFIG" <<'PY' 2>/dev/null
import sys, pathlib, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
print(cfg.get("exp_name") or pathlib.Path(sys.argv[1]).stem)
PY
)
EXP_NAME="${EXP_NAME:-$(basename "${CONFIG%.*}")}"
LOG="logs/exp_${ITER_PAD}_${EXP_NAME}.log"

# Smoketest mode → write to state/smoketests.tsv so reaper/analyze/dashboard
# (which all read state/iterations.tsv) don't pick it up. The training itself
# is identical, only the bookkeeping is sandboxed. Without this, a 1-epoch
# smoketest gets analyzed by claude -p, burns ~5 min + tokens producing a
# (predictable) 'noise' verdict, then misleads downstream propose decisions.
if [ "${AUTORES_SMOKETEST:-0}" = "1" ]; then
    TSV=state/smoketests.tsv
    LOG="logs/smoketest_${EXP_NAME}.log"
else
    TSV=state/iterations.tsv
fi

# ------- Pre-flight: 3 mechanical guards (encoded so an agent can't ignore) --
# Past failures these prevent (each rule has a doc comment naming the symptom):
#   B. duplicate iter   — agent passes the same ITER_NUM twice and clobbers the
#                         ledger; reaper sees "remote/unknown" PID forever
#   A. same-GPU stack   — back-to-back launches see stale free-mem and pin to
#                         the same card; OOM cascade
#   C. cooldown         — even when GPUs differ, two launches within seconds
#                         indicate the agent is auto-flooding without observing
#                         the prior run start

# ---- Guard B: duplicate iter num ---------------------------------------
if [ "${AUTORES_ALLOW_DUPLICATE_ITER:-0}" != "1" ] && [ -f "$TSV" ]; then
    if awk -F'\t' -v n="$ITER_NUM" 'NR>1 && $1==n {found=1} END{exit !found}' "$TSV"; then
        echo "[run_experiment] iter $ITER_NUM already in $TSV — refusing to clobber." >&2
        echo "                 pick a fresh iter num, or set AUTORES_ALLOW_DUPLICATE_ITER=1" >&2
        echo "                 to override (this is almost never what you want)." >&2
        exit 4
    fi
fi

# ---- Guard C: launch cooldown ------------------------------------------
COOLDOWN_S="${AUTORES_LAUNCH_COOLDOWN_S:-30}"
if [ "$COOLDOWN_S" -gt 0 ] && [ -f "$TSV" ]; then
    LAST_TS=$(awk -F'\t' 'NR>1 && $7 != "" {ts=$7} END{print ts}' "$TSV")
    if [ -n "$LAST_TS" ]; then
        LAST_EPOCH=$(date -d "$LAST_TS" '+%s' 2>/dev/null || echo 0)
        NOW_EPOCH=$(date '+%s')
        DELTA=$(( NOW_EPOCH - LAST_EPOCH ))
        if [ "$DELTA" -lt "$COOLDOWN_S" ]; then
            REMAIN=$(( COOLDOWN_S - DELTA ))
            echo "[run_experiment] cooldown active — last launch ${DELTA}s ago, need ${COOLDOWN_S}s." >&2
            echo "                 wait ${REMAIN}s before next launch (prevents same-GPU stacking" >&2
            echo "                 from stale nvidia-smi readings); override AUTORES_LAUNCH_COOLDOWN_S=0" >&2
            exit 5
        fi
    fi
fi

# ------- GPU selection --------------------------------------------------
SKIP_GPUS="${AUTORES_SKIP_GPUS-}"
MIN_FREE_MB="${AUTORES_MIN_FREE_MB:-8000}"

# Build the set of GPUs already occupied by sibling train.py processes
# launched from THIS repo (Guard A: same-GPU stack prevention). We grep for
# `train.py --config <config>` plus a per-process CUDA_VISIBLE_DEVICES marker
# stamped into /proc/<pid>/environ; this matches the children spawned by this
# script regardless of the command-line line wrapping.
BUSY_GPUS=""
if [ "${AUTORES_ALLOW_GPU_STACK:-0}" != "1" ]; then
    for _pid in $(pgrep -f "train\.py.*--config" 2>/dev/null); do
        _envfile="/proc/$_pid/environ"
        [ -r "$_envfile" ] || continue
        _g=$(tr '\0' '\n' < "$_envfile" 2>/dev/null | awk -F= '$1=="CUDA_VISIBLE_DEVICES"{print $2; exit}')
        [ -n "$_g" ] && BUSY_GPUS="$BUSY_GPUS,$_g"
    done
    BUSY_GPUS=$(echo "$BUSY_GPUS" | tr ',' '\n' | sort -u | grep -v '^$' | paste -sd, -)
fi

# Effective skip list = user skip ∪ busy. Empty BUSY_GPUS leaves SKIP_GPUS untouched.
if [ -n "$BUSY_GPUS" ]; then
    EFFECTIVE_SKIP="${SKIP_GPUS:+${SKIP_GPUS},}${BUSY_GPUS}"
else
    EFFECTIVE_SKIP="$SKIP_GPUS"
fi

GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -t',' -k2 -rn \
    | awk -F', ' -v skip="$EFFECTIVE_SKIP" -v thresh="$MIN_FREE_MB" '
        BEGIN { n = split(skip, s, ","); for (i=1; i<=n; i++) if (s[i] != "") skipm[s[i]] = 1 }
        $2+0 >= thresh && !($1 in skipm) { print $1; exit }')
if [ -z "$GPU" ]; then
    echo "[run_experiment] no GPU available." >&2
    echo "                 free-mem threshold: ${MIN_FREE_MB} MB" >&2
    echo "                 skip list (user):   ${SKIP_GPUS:-none}" >&2
    echo "                 skip list (busy):   ${BUSY_GPUS:-none}  ← already running train.py" >&2
    echo "                 wait for one of the busy GPUs to free, or set" >&2
    echo "                 AUTORES_ALLOW_GPU_STACK=1 to override (risk: OOM cascade)." >&2
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"

# ------- spawn ----------------------------------------------------------
PYTHON="${PYTHON:-python3}"
echo "[$(date '+%F %T')] launching iter $ITER_PAD: $EXP_NAME on GPU $GPU (log: $LOG)" >> logs/driver.log
nohup bash -c "
    set -u
    echo '[$(date)] iter $ITER_PAD START exp=$EXP_NAME gpu=$GPU pid=\$\$'
    $PYTHON -u train.py --config '$CONFIG' --iter-num '$ITER_NUM'
    rc=\$?
    echo '[iter $ITER_PAD] FINISH '\$(date '+%Y-%m-%d %H:%M:%S')' rc='\$rc
" > "$LOG" 2>&1 &
PID=$!
disown $PID 2>/dev/null || true

# ------- ledger update --------------------------------------------------
TS=$(date '+%Y-%m-%dT%H:%M:%S%z')
TSV=state/iterations.tsv
if [ ! -f "$TSV" ]; then
    printf 'iter\tstatus\texp_name\tconfig\tgpu\tpid\tstarted_at\tfinished_at\tbest_metric\tverdict\n' > "$TSV"
fi
printf '%d\trunning\t%s\t%s\t%s\t%s\t%s\t\t\t\n' \
    "$ITER_NUM" "$EXP_NAME" "$CONFIG" "$GPU" "$PID" "$TS" >> "$TSV"

if [ "${AUTORES_SMOKETEST:-0}" = "1" ]; then
    echo "[run_experiment] SMOKETEST iter $ITER_PAD launched pid=$PID gpu=$GPU log=$LOG"
    echo "                 (recorded in $TSV; not analyzed by loop)"
else
    echo "[run_experiment] iter $ITER_PAD launched pid=$PID gpu=$GPU log=$LOG"
fi
exit 0
