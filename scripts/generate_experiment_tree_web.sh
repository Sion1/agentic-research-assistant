#!/usr/bin/env bash
# One-command wrapper for the interactive autoresearch experiment tree.
#
# Usage:
#   bash scripts/generate_experiment_tree_web.sh                       # default output path
#   bash scripts/generate_experiment_tree_web.sh /tmp/out.html         # override OUT
#   bash scripts/generate_experiment_tree_web.sh --no-csv              # pass-through flag
#   bash scripts/generate_experiment_tree_web.sh /tmp/out.html --once  # both
set -euo pipefail
cd "$(dirname "$0")/.."

# Treat the first positional arg as OUT only if it's not a flag. This lets
# users pass --flags directly to the python script without accidentally
# binding them to OUT (which would make `bash …web.sh --no-csv` write the
# dashboard to a file literally named `--no-csv`).
OUT="${1:-docs/autoresearch_dashboard/index.html}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  shift
else
  OUT="docs/autoresearch_dashboard/index.html"
fi

python3 scripts/generate_experiment_tree_web.py --out "$OUT" "$@"

# Don't double-prepend pwd when OUT is already absolute — the previous
# version printed `/cwd//abs/path.html` which broke clickable links in
# some terminals.
if [[ "$OUT" = /* ]]; then
  OPEN_PATH="$OUT"
else
  OPEN_PATH="$(pwd)/$OUT"
fi
printf 'Open: %s\n' "$OPEN_PATH"
printf 'For editable user summaries, run: python3 scripts/serve_dashboard.py\n'
