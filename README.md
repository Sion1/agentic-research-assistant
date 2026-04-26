# Agentic Research Assistant

> An agent-assisted research-engineering framework.
>
> **You bring**: research idea, template code, initial environment, prompt files.
> **The framework brings**: agent-driven experiment execution, multi-agent
> consensus review, auto-VCS workflow, interactive HTML dashboard.
>
> The goal is not to replace researchers or automatically generate papers —
> it's to reduce engineering time so you can focus on thinking.

---

## What's inside

```
agentic-research-assistant/
├── README.md                ← this file
├── program.md               ← TEMPLATE: research rules + ablation matrix
├── CLAUDE.md                ← TEMPLATE: long-term working memory
├── loop.sh                  ← framework: scheduler state machine
├── run_experiment.sh        ← TEMPLATE: launcher wrapper
├── train.py / test.py       ← demo: CIFAR-10 + ResNet-34 entry points
├── src/cifar_demo/          ← demo: model / data / trainer
├── configs/                 ← demo: baseline + 3 ablation configs
├── scripts/
│   ├── git_iter_commit.sh           ← framework
│   ├── consensus_iter.sh            ← framework
│   ├── parse_consensus.py           ← framework
│   ├── generate_experiment_tree_web.{sh,py}  ← framework (adapt parser)
│   ├── serve_dashboard.py           ← framework (optional editor)
│   ├── visualize_tsne.py            ← demo: feature t-SNE
│   ├── visualize_cam.py             ← demo: Grad-CAM
│   └── watch_loop.py                ← framework: live loop monitor (read-only)
├── docs/
│   ├── autoresearch_general_by_claude/  ← framework architecture (read first)
│   └── codex_flowcharts/                ← independent re-derivation (5 SVGs)
└── .claude/skills/
    ├── README.md                            ← what to read / what to fill in
    ├── experiment-analysis/                 ← shipped: methodology skill (default)
    └── _template_task_background/           ← TEMPLATE: your domain knowledge
```

The shipped demo trains **ResNet-34 on CIFAR-10** and ablates over
augmentation × optimizer × scheduler. It exists to show how the framework
binds to a concrete project; replace `src/cifar_demo/`, `train.py`,
`configs/`, and the `program.md` / `CLAUDE.md` content with your own and
the same loop will drive *your* research.

---

## Quick start (CIFAR-10 demo)

### 1. Install dependencies

**Required (steps 2–4 — single-experiment + visualization)**:

```bash
pip install torch torchvision pyyaml scikit-learn matplotlib
```

**Required for step 5 (the autonomous loop)**:

You need at least one LLM CLI installed and authenticated. The loop's
`analyze`, `consensus`, and `propose` phases each spawn `claude -p` /
`codex exec` / `gemini -p` subprocesses; without one of them the loop
will refuse to tick (the `claude` binary is checked first; you can
remove that hard requirement by editing `loop.sh:104-106`).

| LLM | Install / auth | Used for |
|---|---|---|
| **Claude Code** (recommended primary) | <https://docs.claude.com/en/docs/claude-code/setup> · login: `claude /login` | analyze · propose · main of consensus |
| **Codex CLI** (recommended secondary) | `npm install -g @openai/codex` · login: `codex auth login` | consensus eval round |
| **Gemini CLI** (optional 3rd reviewer) | `npm install -g @google/gemini-cli` · login: `gemini auth` | consensus eval round |

Verify:
```bash
claude --version       # must succeed for loop.sh
codex --version        # optional, used in consensus_iter.sh
gemini --version       # optional
```

**Optional — Weights & Biases tracking**:

`train.py` will log per-epoch train/test loss & acc to W&B if (a) `wandb`
is importable, AND (b) `WANDB_PROJECT` is set OR `wandb.project` is in
the YAML config. Without both, training silently falls back to local
`history.json`.

```bash
pip install wandb
wandb login                              # paste your API key from wandb.ai/authorize
export WANDB_PROJECT=agentic-research-cifar10-demo
# or in YAML: wandb: { project: my-project, tags: [demo] }
```

### 2. First-launch onboarding (interactive, required once)

On a fresh checkout, run the interactive setup:

```bash
bash scripts/first_launch_setup.sh
```

It asks four small questions (all **strongly recommended**, each declinable):

1. Which Python (with torch + torchvision installed)?
2. Where should datasets live? (default: `./data/`)
3. Which GPUs to force-skip? (shows `nvidia-smi` so you can see)
4. Enable wandb tracking + per-iter git commits / GitHub push?

Choices are written to `state/.env` (sourced by `loop.sh` and
`run_experiment.sh`) and the sentinel `state/.onboarding_done` is created
— that unblocks the Step 0.5 gate in `loop.sh` so it can start ticking.
The gate fires only on a fresh repo (no iter rows yet); existing
checkouts with state are unaffected.

You can re-run the script anytime to change your choices.

**For agents (no TTY available):** pass `--non-interactive` plus the
choices as flags:

```bash
bash scripts/first_launch_setup.sh --non-interactive \
    --python "$(which python3)" \
    --data-root ./data \
    --skip-gpus 0,2 \
    --no-wandb --git --no-push
```

A ready-to-paste agent prompt that drives the whole reproduction is in
[`docs/AGENT_REPRO_PROMPT.md`](docs/AGENT_REPRO_PROMPT.md).

### 3. Smoketest with a single epoch

After onboarding, verify the pipeline end-to-end on a single epoch
(~30 s, downloads CIFAR-10 if absent):

```bash
source state/.env                                              # picks up PYTHON / data root / GPU skip list
EPOCHS_OVERRIDE=1 bash run_experiment.sh configs/cifar10_resnet34.yaml 0
tail -f logs/exp_000_*.log
```

You should see `[ep 0/1]` then a `FINISH ... rc=0` line and a populated
`runs/cifar10_baseline/`. If that works, the full 60-epoch run is the
same command without `EPOCHS_OVERRIDE=1`.

### 4. Run a single experiment manually

```bash
# train one config end-to-end (downloads CIFAR-10 if absent)
bash run_experiment.sh configs/ablation/no_aug.yaml 1

# tail the training log
tail -f logs/exp_001_*.log
```

### 5. Visualize after training

```bash
CKPT=runs/cifar10_iter001_no_aug/best.pth

python3 scripts/visualize_tsne.py --ckpt $CKPT --out figs/iter_001/tsne.png
python3 scripts/visualize_cam.py  --ckpt $CKPT --out figs/iter_001/cam.png
```

### 6. Generate the interactive dashboard

```bash
bash scripts/generate_experiment_tree_web.sh
# → docs/autoresearch_dashboard/index.html (open in any browser)
```

### 7. Hand the wheel to the loop

> **Prerequisite:** at least `claude` (and ideally also `codex` /
> `gemini`) installed and authenticated — see step 1. The loop *will not
> start* without `claude` on PATH.

```bash
# one-time host setup
mkdir -p state && touch state/.loop.enabled.$(hostname)

# (optional) preconfigure consensus mode
cat > state/.consensus.env <<'EOF'
AUTORES_CONSENSUS_ENABLED=1
AUTORES_CONSENSUS_EVAL_AGENTS=claude,codex,gemini
AUTORES_CONSENSUS_TIMEOUT=900
EOF

# launch the supervisor in tmux
tmux new-session -d -s autores "
    export AUTORES_HOST_TAG=$(hostname)
    export MAX_CONCURRENT=2
    export AUTORES_MAX_ITERATIONS=30
    export AUTORES_TARGET_METRIC=0.96
    export AUTORES_TARGET_DIRECTION=max
    while true; do bash loop.sh; sleep 300; done
"
```

The loop ticks every 5 min: reaps finished trainings, calls an LLM to analyze
each one, runs a 5-cycle multi-agent consensus on the next-step, then proposes
and launches the next experiment.

### 8. Watch the loop in real time

While the loop runs, this shows a single-screen status board (wrapper
status, sentinel/lock state, ledger summary, running iters, GPU load,
recent driver-log activity, consensus jobs in flight):

```bash
python3 scripts/watch_loop.py                  # 5 s refresh (default)
python3 scripts/watch_loop.py --interval 2     # 2 s refresh
python3 scripts/watch_loop.py --once           # snapshot, no live refresh
```

The script is **read-only** — it never touches `state/`, never spawns
subprocesses that hold the tick lock, never blocks loop ticks. Safe to
keep open in a side tmux pane indefinitely.

If you set up wandb in step 1, you can also watch live curves at
[wandb.ai](https://wandb.ai) under your project — each iter becomes its
own run with the iter's `exp_name`.

---

## GitHub integration (optional, recommended)

The loop runs correctness-equivalently on a local-only repo, but a
GitHub remote turns each iteration into an auditable artifact. **Recommended
but never required.**

What you get with a remote configured:

- Every analyzed iter becomes a branch `autoresearch/iter-NNN` with the
  config diff, log, viz outputs, and consensus verdict, browsable on
  GitHub.
- `git_iter_commit.sh` auto-pushes the branch and opens a PR via `gh api`,
  so you review iterations as PRs instead of digging through `logs/`.
- The dashboard (`docs/autoresearch_dashboard/`) and the design flowcharts
  (`docs/`) render directly on GitHub Pages or in the file viewer.
- Multi-host runs converge on a single shared remote.

### Setup (one-time)

If you skipped git/push during step 2's onboarding, you can enable it
later by either re-running the setup script or doing it manually.

**Manual setup — all commands MUST run from the repo root**:

```bash
cd /path/to/agent-test            # ← REQUIRED. Without this, `git init` /
                                  #   `git remote add` will fail with
                                  #   "not a git repository".
```

#### Case A — you started from a fresh `git clone` of this framework

This is the common case (you cloned `Sion1/agentic-research-assistant`
to follow the demo, and now you want your own fork to host iter PRs).
The `origin` remote already exists and points at the upstream framework.
You have two choices:

```bash
# Option 1: REPLACE origin to point at your new repo (severs upstream link)
git remote set-url origin https://github.com/<you>/<your-new-repo>.git

# Option 2: KEEP upstream as `upstream`, point `origin` at your fork
#           (recommended — lets you `git pull upstream main` for future updates)
git remote rename origin upstream
git remote add origin https://github.com/<you>/<your-new-repo>.git
```

Then push:

```bash
git push -u origin main
```

#### Case B — your repo isn't a git repo yet

```bash
# 1. init repo. `-b main` requires git ≥ 2.28; older git use the fallback.
git init -b main 2>/dev/null || { git init && git symbolic-ref HEAD refs/heads/main; }

# 2. add your remote.
#    Pick ONE URL format — do NOT mix `git@` and `https://`:
#      SSH:    git@github.com:<you>/<repo>.git        (needs SSH key configured)
#      HTTPS:  https://github.com/<you>/<repo>.git    (needs `gh auth login` or token)
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

#### After either case

```bash
# 3. flip the auto-push flag in state/.env (loop.sh sources this)
echo "export AUTORES_GIT_AUTOPUSH=1" >> state/.env

# 4. authenticate gh CLI (lets autoresearch-bot create PRs)
gh auth login

# 5. wire git itself to use gh's credentials (REQUIRED — gh auth login
#    alone does NOT configure git. Without this, `git push` over HTTPS
#    will prompt for a password, GitHub rejects passwords since 2021-08,
#    and you'll see "Password authentication is not supported".)
gh auth setup-git
```

**Common pitfalls when doing this manually:**

- `state/.env: No such file or directory` → run `scripts/first_launch_setup.sh`
  first (it creates `state/`), or just `mkdir -p state` before the `echo` line.
- `gh: command not found` → see the **No-gh fallback** below.
- `git push` over HTTPS prompts for a password → you skipped step 5
  (`gh auth setup-git`). Run it and retry. (Or use the PAT path in the fallback.)
- `remote: Repository not found` on push → the GitHub repo doesn't
  exist yet. Create it with `gh repo create Sion1/<name> --public`
  (or `--private`) before pushing — or via the GitHub web UI.
- Typos in the repo name go silently into `git remote add` and only
  surface at push time. Fix with `git remote set-url origin <correct-url>`,
  no need to remove and re-add.

#### No-gh fallback (if you can't / don't want to install gh)

```bash
# 1. Create the empty repo via web UI:
#      https://github.com/new   →  name it, Public/Private, do NOT init README

# 2. Create a Personal Access Token via web UI:
#      https://github.com/settings/tokens/new
#      scope: tick [x] repo
#      copy the token (shown ONCE)

# 3. Cache the credentials so you only paste the PAT once:
git config --global credential.helper store

# 4. Push (Username = your GitHub login; Password = the PAT, NOT your account password):
cd /path/to/agent-test
git remote set-url origin https://github.com/<you>/<repo>.git
git push -u origin main
```

The autoresearch loop's auto-PR feature uses `gh api`, so without gh you
get auto-pushes but not auto-PRs. You can still review iter branches on
GitHub; just open PRs manually when you want to merge.

#### Installing gh (one-time, recommended)

```bash
# Debian/Ubuntu — official source
type curl || apt install -y curl
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  > /etc/apt/sources.list.d/github-cli.list
apt update && apt install -y gh

# macOS:    brew install gh
# Conda:    conda install -c conda-forge gh
```

Re-running the interactive setup is equivalent and updates `state/.env`
for you:

```bash
bash scripts/first_launch_setup.sh        # answer Y to git + push
```

Or non-interactively (for agents):

```bash
bash scripts/first_launch_setup.sh --non-interactive \
    --python "$(which python3)" --data-root ./data \
    --git --push --remote-url git@github.com:<you>/<repo>.git \
    --no-wandb
```

### Toggling auto-push

```bash
export AUTORES_GIT_AUTOPUSH=1   # default after onboarding with --push
export AUTORES_GIT_AUTOPUSH=0   # commit per-iter branches locally only, no push
```

The `loop.sh` and `git_iter_commit.sh` paths read this on every tick, so
you can change your mind without restarting anything — toggle the env
var (or edit `state/.env`) and the next iter picks it up.

### When NOT to enable push

- The repo is private and you don't have a GitHub account ready
- You're benchmarking on data that can't leave the local network
- You want to review/squash commits before they leave the machine

In any of those cases, leave `AUTORES_GIT_AUTOPUSH=0` (or pass `--no-push`
during setup). Per-iter branches still get committed locally — you can
push them yourself later with `git push origin autoresearch/iter-NNN`.

---

## Adapting to YOUR research

These five files are where you customize:

| File | What you must change |
|---|---|
| **`README.md`** | Project intro (replace this content) |
| **`program.md`** | Goal / module catalog / ablation matrix / verdict thresholds. **Keep the section structure** — agents parse it. |
| **`CLAUDE.md`** | Baseline numbers + the matrix table; keep "Documented findings" empty for the agent to fill. |
| **`run_experiment.sh`** | Point it at your `train.py`. Keep GPU selection + FINISH-line convention. |
| **`scripts/generate_experiment_tree_web.py`** | `parse_metric_table()` + `group_for()` — replace the demo's `acc / per_class_acc` parser with your metrics, replace `aug / opt / sched` grouping keywords with your ablation axes. |
| **`.claude/skills/_template_task_background/`** | Required. Copy → rename → fill in your field's domain knowledge (failure modes, named conventions, reading checklist). The agent re-reads this every analyze/propose tick. See `.claude/skills/README.md`. |

These you **provide** (replace the demo's contents):

| Path | What |
|---|---|
| `train.py` / `test.py` | Your training/eval entry points. |
| `src/<your_pkg>/` | Your model / data / trainer code. |
| `configs/` | Your YAML configs (baseline + ablation variants). |

These are **task-agnostic framework code**, no edit needed:

| File | Role |
|---|---|
| `loop.sh` | Scheduler state machine (reap → analyze → STOP → cap → gate → propose). Configure via env vars only. |
| `scripts/git_iter_commit.sh` | Per-iter VCS branch + auto-push + auto-PR. |
| `scripts/consensus_iter.sh` | 5-cycle multi-agent consensus on each iter's next-step. |
| `scripts/parse_consensus.py` | Aggregator with 3-layer fallback for the propose-gate. |
| `scripts/serve_dashboard.py` | Optional inline-editor server for user notes. |
| `.claude/skills/experiment-analysis/` | The default analysis protocol skill — `hypothesis × evidence × verdict` rubric. Methodology is task-agnostic; worked examples use the bundled CIFAR-10 demo. |

---

## Framework architecture

The framework's design is documented as flowcharts:

| Diagram | What it shows |
|---|---|
| [docs/autoresearch_general_by_claude/](docs/autoresearch_general_by_claude/) | **Read first.** Architecture-level view of the framework. |
| [docs/codex_flowcharts/](docs/codex_flowcharts/) | An independent re-derivation of the design (5 SVGs) by a second LLM. |

Key invariants you should know before modifying `loop.sh`:

1. **Single-tick single-responsibility** — one tick does at most one of
   {reap, analyze, propose}.
2. **File-as-state-machine** — all cross-tick state lives in
   `state/iterations.tsv` + `logs/iteration_NNN.md` + git branches; the
   scheduler is stateless.
3. **Step ordering** — reap → analyze → STOP → cap → gate → propose, must
   not reorder. Analyze must precede STOP so completed rows always drain.
4. **Multi-host safety** — `AUTORES_HOST_TAG` + per-host sentinel +
   local-FS tick lock.
5. **Defensive Bug-mark** — silent / timeout LLM failures mark the row
   `Bug` to prevent infinite re-fire and token burn.
6. **Async consensus with fd hygiene** — `setsid + nohup ... 9>&-` closes
   the inherited tick-lock fd so consensus doesn't block subsequent ticks.

---

## Configuration via env vars (no code edit needed)

```bash
# Required:
export AUTORES_HOST_TAG=$(hostname)            # multi-host disambiguation

# Resource caps:
export MAX_CONCURRENT=4                         # parallel experiments
export AUTORES_MIN_FREE_MB=24000                # GPU VRAM threshold
export AUTORES_SKIP_GPUS=0,1                    # comma list to skip

# Multi-agent consensus:
export AUTORES_CONSENSUS_ENABLED=1
export AUTORES_CONSENSUS_EVAL_AGENTS=claude,codex,gemini
export AUTORES_CONSENSUS_TIMEOUT=900            # seconds per agent

# Auto-PR:
export AUTORES_GIT_AUTOPUSH=1                   # default 1
export AUTORES_GIT_REMOTE=origin

# Dashboard auto-regen:
export AUTORES_DASHBOARD_ENABLED=1              # default 1
```

The two STOP thresholds in `loop.sh:380-388` (iteration budget + target
metric) are still hard-coded — update them when you adapt to your research.

---

## Citing / acknowledging

If this framework helps your research, a link back to the repo is
appreciated. The shipped CIFAR-10 demo is intentionally minimal — it's a
template, not a benchmark.

---

## License

[Your choice — fill in when you publish.]
