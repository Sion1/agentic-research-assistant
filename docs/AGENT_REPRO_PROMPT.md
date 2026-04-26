# Agent reproduction prompt

Paste the block below into a fresh Claude Code / Codex / Gemini chat to
reproduce this repo end-to-end. Edit only the **TARGET_PATH**, **PYTHON**,
and **SKIP_GPUS** lines to match your machine.

The prompt encodes the discipline (smoketest first, no GPU stacking,
non-interactive setup) so an autonomous agent doesn't reinvent it. It
also makes the agent **explicitly ask the user** about wandb tracking
and git push BEFORE running setup, instead of silently defaulting to
"off".

---

```
You will reproduce the agentic-research-assistant framework's CIFAR-10
demo end-to-end. Read these instructions in full, then execute step by
step. Do NOT skip ahead. Stop and report at every checkpoint marked STOP.

# Target
- repo:        https://github.com/Sion1/agentic-research-assistant
- TARGET_PATH: /radish/xl/agent-test
- PYTHON:      $(which python3)        # must import torch + torchvision
- SKIP_GPUS:   <comma-list, e.g. 0,2>  # GPUs you must NOT use (busy / shared)

# Step 1 — clone (skip if already cloned)
- git clone <repo> <TARGET_PATH>
- cd <TARGET_PATH>

# Step 2 — verify Python env
- Run: $PYTHON -c 'import torch, torchvision, yaml, sklearn, matplotlib; print(torch.__version__, torch.cuda.is_available())'
- If it fails: do NOT auto-install into the system python. Either (a) point
  PYTHON at a conda env that has these, or (b) ask the user.

# Step 3 — ASK THE USER before onboarding (do NOT skip)
Both wandb and git/GitHub-push are STRONGLY RECOMMENDED but require
information you don't have, so ask the user explicitly. Do NOT default
silently to "off" — surface the question. Use this exact wording:

    Before I run scripts/first_launch_setup.sh, two strongly-recommended
    integrations need your decision (each is declinable):

    (1) wandb (Weights & Biases) — tracks per-epoch loss/accuracy and
        hyperparameters with a free academic tier. Without it, only
        local runs/<exp>/history.json is available.
        → Enable? [y/N]
        If yes: I need WANDB_API_KEY in env (you can paste it now), and a
                project name (default: agentic-research-cifar10-demo).

    (2) git + GitHub remote — every analyzed iter becomes a branch
        autoresearch/iter-NNN with config diff + log + viz + verdict.
        Pushing to GitHub gives a browsable audit trail and multi-host
        sync. Without it, iter outputs only live in figs/ + logs/.
        → Init local git repo? [Y/n]
        → Also push to a GitHub remote? [Y/n]
        If yes-to-push: paste the remote URL
                        (e.g. git@github.com:<you>/agentic-research-assistant-fork.git)

Wait for the user's answers before proceeding to Step 4. Do NOT pick
defaults on the user's behalf.

# Step 4 — non-interactive onboarding (replaces the TTY-only setup script)
Build the flag list from Step 3's answers, then run:

    bash scripts/first_launch_setup.sh --non-interactive \
        --python "$PYTHON" \
        --data-root ./data \
        --skip-gpus "$SKIP_GPUS" \
        <wandb-flags> <git-flags> <push-flags>

Flag mapping:
- wandb yes:  --wandb --wandb-project "<project>"   (and export WANDB_API_KEY=... before running)
- wandb no:   --no-wandb
- git yes:    --git
- git no:     --no-git
- push yes:   --push --remote-url "<url>"
- push no:    --no-push

This writes state/.env and state/.onboarding_done. Verify both exist
and confirm to the user which integrations are now active.

# Step 5 — SMOKETEST FIRST (non-negotiable)
Run exactly:
    source state/.env
    EPOCHS_OVERRIDE=1 bash run_experiment.sh configs/cifar10_resnet34.yaml 0
Then:
    until grep -q 'FINISH' logs/exp_000_cifar10_resnet34.log; do sleep 5; done
    tail -3 logs/exp_000_cifar10_resnet34.log
You should see "FINISH ... rc=0" and a populated runs/cifar10_baseline/.
STOP HERE. Report: train_acc, test_acc, wall time, rc.
Do not launch anything else until the user confirms next step.

# Step 6 — only on user's explicit go-ahead
The user will tell you what to run next. Possible asks:
  (a) full 60-epoch baseline:
        bash run_experiment.sh configs/cifar10_resnet34.yaml 1
  (b) one ablation:
        bash run_experiment.sh configs/ablation/no_aug.yaml 2
  (c) hand the wheel to the loop (long-running):
        export AUTORES_HOST_TAG=$(hostname)
        touch state/.loop.enabled.$AUTORES_HOST_TAG
        tmux new-session -d -s autores 'while true; do bash loop.sh; sleep 300; done'

# HARD RULES (re-read before every launch)
1. Smoketest must succeed before any longer run. No exceptions.
2. NEVER launch two `run_experiment.sh` calls within 30 seconds of each
   other — GPU selection sees stale free-mem and stacks them on the same
   card. Wait until the previous training appears in nvidia-smi (or its
   log shows ep 0 starting) before launching the next.
3. NEVER hand-write state/.onboarding_done to bypass the setup script.
   If --non-interactive truly fails, surface the error and ask the user.
4. NEVER set AUTORES_GIT_AUTOPUSH=1 without explicit user confirmation
   of the remote URL.
5. Do not pip-install into a system python without first asking.

# Reporting
- After every step, output one line: "[step N] <what you did> → <result>".
- After Step 3 (the question), STOP and wait for the user's answers.
- After Step 5 (smoketest), STOP and ask before any longer run.
- If you hit any error, stop immediately, paste the last 20 log lines,
  and ask. Do not auto-retry.
```

---

## What the prompt protects against

The four failure modes seen in past reproductions, all encoded as hard
rules above:

| Past failure | Encoded rule |
|---|---|
| Agent bypassed `first_launch_setup.sh` by hand-writing `state/.onboarding_done` because the script required a TTY | Step 4 uses `--non-interactive`; rule #3 forbids hand-write |
| Agent launched 3 trainings in 4 seconds, all on GPU 3 | Rule #2: 30 s cooldown + visible-in-nvidia-smi check |
| Agent skipped smoketest and went straight to a 20-epoch run | Step 5 is non-negotiable; rule #1 |
| Agent enabled `AUTORES_GIT_AUTOPUSH` to a remote it guessed | Rule #4 + Step 3 forces explicit user confirmation |
| Agent silently picked `--no-wandb --no-push` defaults without telling the user, so wandb tracking and GitHub history were never set up | Step 3 makes the agent **ask** before running setup, with strong-recommendation framing |
