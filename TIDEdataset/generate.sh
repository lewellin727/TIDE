#!/usr/bin/env bash
# One-command benchmark generation: Phase A (structure) + Phase B (NL queries), with all
# output tee'd into a single log/<dataset>.log. Run inside the project conda env, e.g.:
#   conda run -n tide bash generate.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1               # every print lands in the log in real time
DS=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['dataset'])")
LOG="log/${DS}.log"; mkdir -p log
{
  echo "################ GENERATION $(date '+%F %T')  dataset=${DS} ################"
  echo;  echo "################ PHASE A — structure + augment ################"
  python -u main.py
  echo;  echo "################ PHASE B — NL queries ################"
  python -u query_llm.py
} 2>&1 | stdbuf -oL -eL tee "$LOG"
echo "[generate] consolidated log -> $LOG"
