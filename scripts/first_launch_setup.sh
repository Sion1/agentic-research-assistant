#!/usr/bin/env bash
# Interactive onboarding for a fresh agentic-research-assistant checkout.
#
# The autoresearch loop runs many experiments unattended and a fresh
# environment usually has 4 small surprises waiting for the user (or for an
# autonomous agent that gets handed the repo): which Python env to use, where
# to put the dataset, which GPUs to skip, whether to enable wandb / git push.
# Doing this once interactively at the start saves a lot of guess-and-fix
# wandering later.
#
# All four integrations are STRONGLY RECOMMENDED. You may decline any of
# them; the loop runs correctness-equivalently without them.
#
# Writes:
#   state/.env               — sourced by loop.sh / run_experiment.sh
#   state/.onboarding_done   — sentinel that unblocks Step 0.5 in loop.sh

set -u
cd "$(dirname "$0")/.."

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
BOLD='\033[1m'
NC='\033[0m'

prompt_yn() {
    local q="$1" def="${2:-y}" hint reply
    if [ "$def" = "y" ]; then hint="[Y/n]"; else hint="[y/N]"; fi
    while true; do
        printf "%b%s%b %s " "$BOLD" "$q" "$NC" "$hint"
        read -r reply
        reply="${reply:-$def}"
        case "$reply" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo])     return 1 ;;
        esac
    done
}

prompt_str() {
    local q="$1" def="${2:-}" reply
    if [ -n "$def" ]; then
        printf "%b%s%b [default: %s]: " "$BOLD" "$q" "$NC" "$def"
    else
        printf "%b%s%b: " "$BOLD" "$q" "$NC"
    fi
    read -r reply
    printf '%s' "${reply:-$def}"
}

# ----------------------------------------------------------------------------
# Non-interactive mode
#
# Agents (and CI) cannot drive a TTY-only Q&A. Pass --non-interactive plus the
# choices as flags / env vars; the script then skips all prompts and writes
# state/.env + state/.onboarding_done with the supplied values. Defaults below
# match the interactive path's defaults.
#
# Flags (non-interactive only):
#   --non-interactive            enable this mode
#   --python <path>              Python interpreter (must import torch + torchvision)
#   --data-root <dir>            dataset root (default ./data)
#   --skip-gpus <csv>            GPU indices to force-skip (default empty)
#   --wandb                      enable wandb (will REQUIRE WANDB_API_KEY env)
#   --no-wandb                   disable wandb (default)
#   --wandb-project <name>       wandb project name (default agentic-research-cifar10-demo)
#   --git                        enable per-iter git (default)
#   --no-git                     disable per-iter git
#   --push                       enable git auto-push (REQUIRES --remote-url or existing origin)
#   --no-push                    disable git auto-push (default)
#   --remote-url <url>           GitHub remote URL (only used with --push)
# ----------------------------------------------------------------------------
NONINT=0
NI_PYTHON="${PYTHON:-$(command -v python3 || echo python3)}"
NI_DATA_ROOT="./data"
NI_SKIP_GPUS=""
NI_WANDB=0
NI_WANDB_PROJECT="agentic-research-cifar10-demo"
NI_GIT=1
NI_PUSH=0
NI_REMOTE_URL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --non-interactive)  NONINT=1; shift ;;
        --python)           NI_PYTHON="$2"; shift 2 ;;
        --data-root)        NI_DATA_ROOT="$2"; shift 2 ;;
        --skip-gpus)        NI_SKIP_GPUS="$2"; shift 2 ;;
        --wandb)            NI_WANDB=1; shift ;;
        --no-wandb)         NI_WANDB=0; shift ;;
        --wandb-project)    NI_WANDB_PROJECT="$2"; shift 2 ;;
        --git)              NI_GIT=1; shift ;;
        --no-git)           NI_GIT=0; shift ;;
        --push)             NI_PUSH=1; shift ;;
        --no-push)          NI_PUSH=0; shift ;;
        --remote-url)       NI_REMOTE_URL="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,/^# ---/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ "$NONINT" = 0 ] && [ ! -t 0 ]; then
    echo "ERROR: this script must run interactively (a TTY is required)." >&2
    echo "       For agents/CI: pass --non-interactive plus --python / --data-root /" >&2
    echo "                      --skip-gpus / --wandb|--no-wandb / --git|--no-git /" >&2
    echo "                      --push|--no-push (see header for full flag list)."  >&2
    exit 2
fi

mkdir -p state
ENV_FILE=state/.env
: > "$ENV_FILE"

# ---- Non-interactive fast path ---------------------------------------------
if [ "$NONINT" = 1 ]; then
    echo "[setup] non-interactive mode"
    if ! "$NI_PYTHON" -c 'import torch, torchvision' >/dev/null 2>&1; then
        echo "ERROR: '$NI_PYTHON' cannot import torch + torchvision" >&2
        exit 3
    fi
    TORCH_VER=$("$NI_PYTHON" -c 'import torch; print(torch.__version__)')
    CUDA_OK=$("$NI_PYTHON" -c 'import torch; print(torch.cuda.is_available())')
    mkdir -p "$NI_DATA_ROOT" || { echo "ERROR: cannot create $NI_DATA_ROOT" >&2; exit 4; }

    {
        echo "export PYTHON=$NI_PYTHON"
        echo "export AUTORES_DATA_ROOT=$NI_DATA_ROOT"
        [ -n "$NI_SKIP_GPUS" ] && echo "export AUTORES_SKIP_GPUS=$NI_SKIP_GPUS"
        if [ "$NI_WANDB" = 1 ]; then
            if ! "$NI_PYTHON" -c 'import wandb' >/dev/null 2>&1; then
                echo "ERROR: --wandb requested but wandb not importable from $NI_PYTHON" >&2
                exit 5
            fi
            if [ -z "${WANDB_API_KEY:-}" ]; then
                echo "ERROR: --wandb requested but WANDB_API_KEY env not set" >&2
                exit 6
            fi
            echo "export WANDB_PROJECT=$NI_WANDB_PROJECT"
        fi
        echo "export AUTORES_GIT_AUTOPUSH=$NI_PUSH"
    } > "$ENV_FILE"

    if [ "$NI_GIT" = 1 ]; then
        if [ ! -d .git ] && [ ! -f .git ]; then
            git init -b main 2>/dev/null || git init
        fi
        git config user.name  >/dev/null 2>&1 || git config user.name  "autoresearch"
        git config user.email >/dev/null 2>&1 || git config user.email "autoresearch@localhost"
        if [ "$NI_PUSH" = 1 ] && [ -n "$NI_REMOTE_URL" ]; then
            git remote get-url origin >/dev/null 2>&1 || git remote add origin "$NI_REMOTE_URL"
        fi
    fi

    {
        echo "onboarding_done_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "mode=non-interactive"
        echo "python=$NI_PYTHON"
        echo "torch=$TORCH_VER"
        echo "cuda_available=$CUDA_OK"
        echo "data_root=$NI_DATA_ROOT"
        echo "wandb_enabled=$NI_WANDB"
        echo "git_enabled=$NI_GIT"
        echo "git_autopush_enabled=$NI_PUSH"
    } > state/.onboarding_done
    echo "[setup] state/.env + state/.onboarding_done written. Done."
    exit 0
fi

echo
printf "%b== agentic-research-assistant first-launch setup ==%b\n" "$BOLD" "$NC"
echo
echo "Four small choices, all strongly recommended. You may decline any."
echo

# ---- 1. Python environment -------------------------------------------------
echo
printf "%b[1/4] Python environment%b\n" "$BOLD" "$NC"
PYTHON_CMD=$(prompt_str "Path or command for python with torch + torchvision installed" "$(command -v python3 || echo python3)")
if ! "$PYTHON_CMD" -c 'import torch, torchvision' >/dev/null 2>&1; then
    printf "%b'%s' cannot import torch + torchvision.%b\n" "$RED" "$PYTHON_CMD" "$NC"
    echo "  Either:"
    echo "    a) activate a conda env with torch first, then re-run this script;"
    echo "    b) install:  pip install torch torchvision pyyaml scikit-learn matplotlib wandb"
    echo "  Then re-run this script."
    exit 3
fi
TORCH_VER=$("$PYTHON_CMD" -c 'import torch; print(torch.__version__)')
CUDA_OK=$("$PYTHON_CMD" -c 'import torch; print(torch.cuda.is_available())')
printf "%bpython=%s torch=%s cuda=%s%b\n" "$GREEN" "$PYTHON_CMD" "$TORCH_VER" "$CUDA_OK" "$NC"
echo "export PYTHON=$PYTHON_CMD" >> "$ENV_FILE"

# ---- 2. Dataset root --------------------------------------------------------
echo
printf "%b[2/4] Dataset root%b\n" "$BOLD" "$NC"
echo "  Where should CIFAR-10 (and other datasets) live?"
echo "  Default ./data/ stays inside the repo (safe, ~170 MB for CIFAR-10)."
echo "  Use an absolute path if you want to share a dataset cache across runs."
DATA_ROOT=$(prompt_str "Dataset root directory" "./data")
mkdir -p "$DATA_ROOT" 2>/dev/null || {
    printf "%bcannot create %s — pick another path%b\n" "$RED" "$DATA_ROOT" "$NC"
    exit 4
}
printf "%bdataset root: %s%b\n" "$GREEN" "$DATA_ROOT" "$NC"
echo "export AUTORES_DATA_ROOT=$DATA_ROOT" >> "$ENV_FILE"

# ---- 3. GPU skip-list -------------------------------------------------------
echo
printf "%b[3/4] GPU selection%b\n" "$BOLD" "$NC"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null \
        | sed 's/^/    /'
    echo
    echo "  GPUs with < AUTORES_MIN_FREE_MB (default 15 GB) free are auto-skipped"
    echo "  by run_experiment.sh. Add indices here only if you also want to reserve"
    echo "  full ones for other work (e.g. a shared GPU 0 used by your IDE)."
    SKIP=$(prompt_str "Comma-separated GPU indices to FORCE-skip (blank = none)" "")
    if [ -n "$SKIP" ]; then
        echo "export AUTORES_SKIP_GPUS=$SKIP" >> "$ENV_FILE"
        printf "%bskip list: %s%b\n" "$GREEN" "$SKIP" "$NC"
    else
        printf "%bno forced skip%b\n" "$GREEN" "$NC"
    fi
else
    printf "%bnvidia-smi not found — assuming CPU-only%b\n" "$YELLOW" "$NC"
fi

# ---- 4. wandb ---------------------------------------------------------------
echo
printf "%b[4a/4] Weights & Biases (wandb)%b\n" "$BOLD" "$NC"
echo "  Tracks per-epoch metrics + hyperparameters with a free academic tier."
echo "  Without it: only local runs/<exp>/history.json available."
WANDB_OK=0
if prompt_yn "Enable wandb tracking? (strongly recommended)" "y"; then
    if ! "$PYTHON_CMD" -c 'import wandb' >/dev/null 2>&1; then
        printf "%bwandb not in this Python env.%b Install: pip install wandb\n" "$RED" "$NC"
        echo "  Then re-run this script."
        exit 5
    fi
    if "$PYTHON_CMD" -m wandb status 2>/dev/null | grep -qi "logged in"; then
        printf "%balready logged in%b\n" "$GREEN" "$NC"
    else
        echo "Launching 'wandb login' — paste API key from https://wandb.ai/authorize:"
        if ! "$PYTHON_CMD" -m wandb login; then
            printf "%bwandb login failed.%b You can re-run this script after fixing it.\n" "$RED" "$NC"
            exit 6
        fi
    fi
    PROJECT=$(prompt_str "wandb project name" "agentic-research-cifar10-demo")
    echo "export WANDB_PROJECT=$PROJECT" >> "$ENV_FILE"
    WANDB_OK=1
else
    printf "%bwandb disabled.%b Re-run this script anytime to enable.\n" "$YELLOW" "$NC"
fi

# ---- 4b. git / GitHub -------------------------------------------------------
echo
printf "%b[4b/4] git + (optional) GitHub remote%b\n" "$BOLD" "$NC"
echo "  Each analyzed iteration becomes a branch autoresearch/iter-NNN with"
echo "  config diff, log, viz outputs and consensus verdict. Pushable to GitHub"
echo "  for browsable history + multi-host sync."
GIT_OK=0; PUSH_OK=0
if prompt_yn "Enable per-iter git commits? (strongly recommended)" "y"; then
    if ! command -v git >/dev/null 2>&1; then
        printf "%bgit not installed.%b Install it and re-run.\n" "$RED" "$NC"
        exit 7
    fi
    if [ ! -d .git ] && [ ! -f .git ]; then
        echo "No git repo here yet. Initializing..."
        git init -b main 2>/dev/null || git init
    fi
    git config user.name  >/dev/null 2>&1 || git config user.name  "autoresearch"
    git config user.email >/dev/null 2>&1 || git config user.email "autoresearch@localhost"
    GIT_OK=1
    printf "%bgit ready%b\n" "$GREEN" "$NC"

    echo
    if prompt_yn "Also push each iter branch to a GitHub remote? (recommended)" "y"; then
        EXISTING_REMOTE=$(git remote get-url origin 2>/dev/null || true)
        if [ -n "$EXISTING_REMOTE" ]; then
            printf "  remote 'origin' already set to: %s\n" "$EXISTING_REMOTE"
        else
            REMOTE_URL=$(prompt_str "GitHub remote URL (blank = skip)" "")
            if [ -n "$REMOTE_URL" ]; then
                git remote add origin "$REMOTE_URL" || true
                EXISTING_REMOTE="$REMOTE_URL"
            fi
        fi
        if [ -n "${EXISTING_REMOTE:-}" ]; then
            echo "export AUTORES_GIT_AUTOPUSH=1" >> "$ENV_FILE"
            PUSH_OK=1
            printf "%bauto-push enabled%b\n" "$GREEN" "$NC"
        else
            echo "export AUTORES_GIT_AUTOPUSH=0" >> "$ENV_FILE"
            printf "%bno remote — auto-push disabled%b\n" "$YELLOW" "$NC"
        fi
    else
        echo "export AUTORES_GIT_AUTOPUSH=0" >> "$ENV_FILE"
        printf "%bauto-push disabled%b\n" "$YELLOW" "$NC"
    fi
else
    printf "%bgit commits disabled.%b Iter outputs will only live in figs/ + logs/.\n" "$YELLOW" "$NC"
fi

# ---- sentinel ---------------------------------------------------------------
{
    echo "onboarding_done_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "python=$PYTHON_CMD"
    echo "torch=$TORCH_VER"
    echo "cuda_available=$CUDA_OK"
    echo "data_root=$DATA_ROOT"
    echo "wandb_enabled=$WANDB_OK"
    echo "git_enabled=$GIT_OK"
    echo "git_autopush_enabled=$PUSH_OK"
} > state/.onboarding_done

echo
printf "%bSetup complete.%b Choices written to state/.env, sentinel at state/.onboarding_done.\n" "$GREEN" "$NC"
echo
echo "Quick smoketest (1 epoch on the GPU you just configured):"
echo "  source state/.env"
echo "  bash run_experiment.sh configs/cifar10_resnet34.yaml --smoketest"
echo
echo "Full run (60 epochs):"
echo "  bash run_experiment.sh configs/cifar10_resnet34.yaml"
echo
