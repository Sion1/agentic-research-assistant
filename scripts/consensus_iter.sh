#!/usr/bin/env bash
# consensus_iter.sh — 5-cycle main+eval consensus workflow for one iteration.
#
# Replaces parallel cross_validate_iter.sh. Sequential, not parallel:
#   Cycle 1: main analyze (already done by loop's analyze step before this fires)
#   Cycle 2: eval R1 — each eval agent reviews iter_NNN.md + raw data
#   Cycle 3: main revise — main reads 3 eval reports, holds or revises §8 next-step
#   Cycle 4: eval R2 — each eval re-reviews the revised §8
#   Cycle 5: main final — main reads R2 evals, declares CONSENSUS or OVERRIDE
#
# Outcome: writes logs/iteration_NNN.consensus.final.md with the next-step that
# the propose phase MUST use (whether consensus or main-override).
#
# After cycle 5: if not consensus, main's choice wins (per user spec).
#
# Triggered by loop.sh in background (non-blocking) when AUTORES_CONSENSUS_ENABLED=1.
#
# Usage: bash scripts/consensus_iter.sh <ITER_NUM>
#
# Env (sourced from state/.consensus.env at loop tick start):
#   AUTORES_CONSENSUS_ENABLED=1            — required to be triggered
#   AUTORES_CONSENSUS_EVAL_AGENTS="claude,codex,gemini"
#                                          — comma-list, sequential per cycle
#   AUTORES_CONSENSUS_TIMEOUT=900          — per-agent SIGTERM (seconds)
#
# Output (all under logs/):
#   iteration_NNN.consensus.<agent>.r1.md  — eval round 1 reports
#   iteration_NNN.consensus.main.r3.md     — main revise (cycle 3) note
#   iteration_NNN.consensus.<agent>.r2.md  — eval round 2 (just AGREE/DISAGREE)
#   iteration_NNN.consensus.main.r5.md     — main final (cycle 5) verdict
#   iteration_NNN.consensus.final.md       — comparator output: CONSENSUS or OVERRIDE
#
# Exit codes:
#   0 = consensus.final.md written (with status CONSENSUS or OVERRIDE)
#   1 = bad usage / missing primary report
set -u
cd "$(dirname "$0")/.."

ITER="${1:-}"
if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "[consensus] usage: $0 <ITER_NUM>" >&2
    exit 1
fi
ITER_PAD=$(printf '%03d' "$ITER")
LOG=logs/driver.log
log() { printf '[%s] [consensus iter%s] %s\n' "$(date '+%F %T')" "$ITER_PAD" "$*" | tee -a "$LOG"; }

PRIMARY="logs/iteration_${ITER_PAD}.md"
if [ ! -f "$PRIMARY" ]; then
    log "ERROR: primary report $PRIMARY not found; cannot run consensus"
    exit 1
fi

EXP_NAME=$(awk -F'\t' -v i="$ITER" '$1 == i {print $3; exit}' state/iterations.tsv)
if [ -z "$EXP_NAME" ]; then
    log "ERROR: no row in state/iterations.tsv for iter $ITER"
    exit 1
fi

IFS=',' read -ra EVAL_AGENTS <<< "${AUTORES_CONSENSUS_EVAL_AGENTS:-claude,codex,gemini}"
TIMEOUT_SEC="${AUTORES_CONSENSUS_TIMEOUT:-900}"

# Per-agent model overrides — let users pin to whatever's available on their
# account/tier without editing this file. Defaults to the latest stable
# flagships at time of writing; CLI flags accept any model the agent supports.
AUTORES_CLAUDE_MODEL="${AUTORES_CLAUDE_MODEL:-claude-opus-4-7}"
AUTORES_CODEX_MODEL="${AUTORES_CODEX_MODEL:-gpt-5.5}"
AUTORES_GEMINI_MODEL="${AUTORES_GEMINI_MODEL:-gemini-2.5-pro}"

log "starting 5-cycle consensus (eval agents: ${EVAL_AGENTS[*]}, per-agent timeout ${TIMEOUT_SEC}s)"

# ---------------------------------------------------------------------------
# Helper: dispatch one agent with a prompt file. Output goes to driver.log.
# ---------------------------------------------------------------------------
dispatch_agent() {
    local agent="$1"
    local prompt_file="$2"
    case "$agent" in
        claude)
            timeout --signal=TERM --kill-after=30 "$TIMEOUT_SEC" claude -p "$(cat "$prompt_file")" \
                --model "$AUTORES_CLAUDE_MODEL" \
                --allowedTools "Bash,Read,Glob,Grep,Write" \
                --permission-mode acceptEdits \
                --max-turns 40 \
                >> "$LOG" 2>&1
            ;;
        codex)
            timeout --signal=TERM --kill-after=30 "$TIMEOUT_SEC" codex exec \
                --model "$AUTORES_CODEX_MODEL" \
                --sandbox danger-full-access \
                "$(cat "$prompt_file")" \
                >> "$LOG" 2>&1
            ;;
        gemini)
            timeout --signal=TERM --kill-after=30 "$TIMEOUT_SEC" gemini \
                --model "$AUTORES_GEMINI_MODEL" \
                --skip-trust --yolo \
                -p "$(cat "$prompt_file")" \
                >> "$LOG" 2>&1
            ;;
        *)
            log "ERROR: unknown agent '$agent' (supported: claude codex gemini)"
            return 1
            ;;
    esac
    return $?
}

# ===========================================================================
# CYCLE 2 — eval round 1: each agent reviews main's analysis + writes critique
# ===========================================================================
log "cycle 2/5 — eval R1 (sequential across ${#EVAL_AGENTS[@]} agents)"
for agent in "${EVAL_AGENTS[@]}"; do
    OUT="logs/iteration_${ITER_PAD}.consensus.${agent}.r1.md"
    PROMPT=$(mktemp)
    cat > "$PROMPT" <<EOF
You are an EVAL agent reviewing a primary research analysis. Your role is
critique, NOT independent re-derivation. The main agent already analyzed
iteration ${ITER_PAD} (experiment: ${EXP_NAME}); your job is to validate
its verdict (§6) and especially its proposed NEXT step (§8).

## TASK

Read these files first:
1. ${PRIMARY} — main's full analysis (verdict in §6, next-step in §8)
2. program.md — research rules + Success/Partial/Failure criteria
3. CLAUDE.md — current best, prior findings, the running ablation matrix
4. state/iterations.tsv — overall loop state

Then verify the main's claims by spot-checking:
- Load metrics: runs/${EXP_NAME}/final.pth — does the primary report's headline metric match the checkpoint's `metrics` dict?
- Read figs/iter_${ITER_PAD}/per_class.csv if present — do per-class signals match the report's claims?
- Read any other artifacts the project emits (e.g. figs/iter_${ITER_PAD}/*.json, figs/iter_${ITER_PAD}/*.png) and confirm they are consistent with the verdict.

## OUTPUT — write to ${OUT} using THIS EXACT format

\`\`\`markdown
# Iteration ${ITER_PAD} consensus eval R1 by ${agent}
Date: $(date '+%Y-%m-%d %H:%M %Z')

## Verdict review (main said: <quote main's §6 word>)
**AGREE** | **DISAGREE**
<1 sentence: if DISAGREE, what do you think the verdict should be and why>

## Next-step review (main's §8 said: <quote main's §8 first sentence>)
**AGREE** | **DISAGREE** | **COUNTER-PROPOSE**
<1 paragraph: if DISAGREE/COUNTER-PROPOSE, your alternative next-step with reasoning. Be SPECIFIC — name the config delta, the hypothesis, and the falsification criterion. Cite which prior iter's finding motivates it.>

## Issues identified (the things main missed or got wrong)
- <bullet 1, or "none">
- <bullet 2>

## Confidence in this critique
**high** | **medium** | **low** — <1 sentence reason>
\`\`\`

## HARD CONSTRAINTS
- DO NOT modify state/iterations.tsv, CLAUDE.md, configs/, src/, train.py, or any checkpoint.
- DO NOT launch training / GPU jobs.
- DO NOT commit / push / open PRs.
- The main agent will read your critique and may revise. Your critique should be
  ACTIONABLE: if you DISAGREE, give main something concrete to consider, not just
  "I don't like this".

Be terse — under 40 turns. Quality over verbosity.
EOF
    log "  R1 dispatching ${agent} → $(basename $OUT)"
    dispatch_agent "$agent" "$PROMPT"
    rm -f "$PROMPT"
    log "  R1 ${agent} done (file exists: $([ -f "$OUT" ] && echo yes || echo NO))"
done

# ===========================================================================
# CYCLE 3 — main revise: read 3 eval reports, hold or revise §8
# ===========================================================================
log "cycle 3/5 — main revise"
R3_OUT="logs/iteration_${ITER_PAD}.consensus.main.r3.md"
PROMPT=$(mktemp)
cat > "$PROMPT" <<EOF
You wrote the primary analysis for iteration ${ITER_PAD} (${EXP_NAME}). Three
eval agents have just reviewed it. Your job in this cycle 3/5 is to read
their critiques and decide whether to HOLD your §8 next-step or REVISE it.

## INPUT

1. Your original analysis: ${PRIMARY}
2. Eval critiques (read all):
$(for a in "${EVAL_AGENTS[@]}"; do echo "   - logs/iteration_${ITER_PAD}.consensus.${a}.r1.md"; done)

## TASK

Decide per HOLD-vs-REVISE policy:
- If 0 evals DISAGREE on next-step → HOLD (no revision needed; you'll likely get consensus in R2)
- If 1+ evals DISAGREE/COUNTER-PROPOSE on next-step:
  - Evaluate their alternatives objectively
  - REVISE if their critique is technically valid and you can integrate it
  - HOLD with explicit defense if you believe their critique is wrong

## OUTPUT — TWO things

(a) Update ${PRIMARY} §8 if revising (overwrite with the revised next-step;
    keep the rest of the file intact).

(b) Write ${R3_OUT} using THIS EXACT format:

\`\`\`markdown
# Iteration ${ITER_PAD} consensus main R3 (revise step)
Date: $(date '+%Y-%m-%d %H:%M %Z')

## Decision
**HOLD** | **REVISE**

## Eval critique summary (1 line per eval agent)
- claude (R1): AGREE/DISAGREE/COUNTER on next-step — <key point>
- codex  (R1): AGREE/DISAGREE/COUNTER on next-step — <key point>
- gemini (R1): AGREE/DISAGREE/COUNTER on next-step — <key point>

## My response (1-2 paragraphs)
<If HOLD: defend why your original §8 next-step is still correct despite eval critique.
If REVISE: explain what you changed in §8 and why eval critique was technically valid.
Either way, be specific — name the config delta, hypothesis, falsification
criterion. Cite the prior iter findings that motivate the choice.>

## Revised next-step (only if REVISE; else "unchanged")
<config delta + hypothesis + falsification — 1 paragraph>
\`\`\`

## HARD CONSTRAINTS
- You may EDIT ${PRIMARY} (only the §8 section if REVISE).
- DO NOT modify state.tsv, CLAUDE.md (yet — that comes after consensus.final).
- DO NOT launch training. DO NOT commit / push.

Under 30 turns.
EOF
log "  R3 dispatching main claude"
dispatch_agent "claude" "$PROMPT"
rm -f "$PROMPT"
log "  R3 main claude done (file exists: $([ -f "$R3_OUT" ] && echo yes || echo NO))"

# ===========================================================================
# CYCLE 4 — eval round 2: each agent re-reviews the revised §8 (terse AGREE/DISAGREE)
# ===========================================================================
log "cycle 4/5 — eval R2 (just AGREE/DISAGREE on revised §8)"
for agent in "${EVAL_AGENTS[@]}"; do
    OUT="logs/iteration_${ITER_PAD}.consensus.${agent}.r2.md"
    PROMPT=$(mktemp)
    cat > "$PROMPT" <<EOF
You evaluated iteration ${ITER_PAD} in R1 (your file:
logs/iteration_${ITER_PAD}.consensus.${agent}.r1.md). Main has now responded
in cycle 3 (logs/iteration_${ITER_PAD}.consensus.main.r3.md) and either
held or revised the §8 next-step. Your job in R2 is a SHORT verdict on
the (possibly-revised) §8 next-step in ${PRIMARY}.

## TASK

1. Re-read ${PRIMARY} §8 (the revised next-step — may be unchanged from R1).
2. Read main's R3 reasoning at logs/iteration_${ITER_PAD}.consensus.main.r3.md.
3. Decide: AGREE or DISAGREE on the now-final §8.

## OUTPUT — write to ${OUT} using THIS EXACT format

\`\`\`markdown
# Iteration ${ITER_PAD} consensus eval R2 by ${agent}
Date: $(date '+%Y-%m-%d %H:%M %Z')

## R2 verdict
**AGREE** | **DISAGREE**

## Reasoning (1-2 sentences max)
<If AGREE: 1 line acknowledging main's defense or accepting the revision.
If DISAGREE: 1-2 lines on what's still wrong with the §8 next-step.>
\`\`\`

## HARD CONSTRAINTS
- Read-only. Don't modify project files except your own R2 report.
- This is the FINAL eval round. After this, main makes a final call in cycle 5.
- Be terse — under 15 turns.
EOF
    log "  R2 dispatching ${agent} → $(basename $OUT)"
    dispatch_agent "$agent" "$PROMPT"
    rm -f "$PROMPT"
    log "  R2 ${agent} done (file exists: $([ -f "$OUT" ] && echo yes || echo NO))"
done

# ===========================================================================
# CYCLE 5 — main final: declare CONSENSUS or OVERRIDE based on R2 evals
# ===========================================================================
log "cycle 5/5 — main final"
R5_OUT="logs/iteration_${ITER_PAD}.consensus.main.r5.md"
PROMPT=$(mktemp)
cat > "$PROMPT" <<EOF
This is cycle 5/5 — the FINAL step of the consensus workflow for iteration
${ITER_PAD}. Read the R2 eval verdicts and declare either CONSENSUS or
OVERRIDE.

## INPUT

- ${PRIMARY} §8 — the now-final next-step
- Eval R2 reports:
$(for a in "${EVAL_AGENTS[@]}"; do echo "   - logs/iteration_${ITER_PAD}.consensus.${a}.r2.md"; done)
- Your own R3 reasoning: logs/iteration_${ITER_PAD}.consensus.main.r3.md

## DECISION RULE (per project policy, NO EXCEPTIONS)

- If ALL R2 evals report AGREE → status = **CONSENSUS**
- If ANY R2 eval reports DISAGREE → status = **OVERRIDE_BY_MAIN**
  (per user spec: "after 5 cycles, if no agreement, main's next-step wins").
  In OVERRIDE case, you must STILL go with the §8 next-step (the dissenting
  evals are noted but do not block).

## OUTPUT — write to ${R5_OUT} using THIS EXACT format

\`\`\`markdown
# Iteration ${ITER_PAD} consensus main R5 (final call)
Date: $(date '+%Y-%m-%d %H:%M %Z')

## Status
**CONSENSUS** | **OVERRIDE_BY_MAIN**

## Eval R2 tally
- claude: AGREE | DISAGREE
- codex:  AGREE | DISAGREE
- gemini: AGREE | DISAGREE

## Final next-step (the propose phase will use THIS)
<1-paragraph specification of the next iter:
- config delta (e.g., "attr_reg_weight: 0.2 → 0.3")
- hypothesis (1 sentence)
- falsification criterion (H thresholds for Success/Partial/Failure)
Should match ${PRIMARY} §8 exactly.>

## Outstanding dissent (only if OVERRIDE_BY_MAIN)
<1 sentence per dissenting agent: their objection + why you're proceeding anyway>
\`\`\`

## HARD CONSTRAINTS
- Read-only on project except writing ${R5_OUT}.
- Be terse — under 10 turns.
EOF
log "  R5 dispatching main claude"
dispatch_agent "claude" "$PROMPT"
rm -f "$PROMPT"
log "  R5 main claude done (file exists: $([ -f "$R5_OUT" ] && echo yes || echo NO))"

# ===========================================================================
# Final: write consensus.final.md — the propose phase reads this
# ===========================================================================
log "writing consensus.final.md via parse_consensus.py"
python3 scripts/parse_consensus.py "$ITER" >> "$LOG" 2>&1
RC=$?
if [ "$RC" -ne 0 ]; then
    log "ERROR: parse_consensus.py rc=$RC"
fi

# Sanity log
if [ -f "logs/iteration_${ITER_PAD}.consensus.final.md" ]; then
    STATUS=$(grep -oE 'STATUS=[A-Z_]+' "logs/iteration_${ITER_PAD}.consensus.final.md" | head -1 | cut -d= -f2)
    log "DONE — consensus.final.md status: ${STATUS:-?}"
else
    log "ERROR: consensus.final.md NOT written; propose phase will be BLOCKED for this iter"
fi
exit 0
