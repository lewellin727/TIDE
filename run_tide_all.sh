#!/usr/bin/env bash
# =============================================================================
# TIDE RERUN pass — re-run the recall=0 / empty-candidate queries of an EXISTING
# run, merge (later-overrides) into the canonical per-model result, re-eval.
#
# Usage:
#   ./run_tide_all.sh [TAG] [GPU] [ROUND] [K]
#     TAG   : model tag (default: deepseek). Operates on results/<ds>/tide-<TAG>.json.
#     GPU   : CUDA device index (default: 0).
#     ROUND : rerun round number (default: 1) — only names the rerun iter/log files.
#     K     : recall cutoff defining "recall=0" (default: 30). A query is a rerun
#             target iff it has EMPTY candidates OR zero GT hits in its top-K.
#
#   Recommended inside a detached screen so it survives disconnect:
#     screen -dmS tide_rerun bash -lic "cd $(pwd) && ./run_tide_all.sh deepseek 0 1"
#
# Per dataset it:
#   1. computes the recall=0 / empty idx list from results/<ds>/tide-<TAG>.json
#      (rerun_helper.py zeros). Skips the dataset if the list is empty.
#   2. re-runs ONLY those idxs -> results/<ds>/iter-<TAG>/<ds>_zeros<ROUND>_rerun.json
#   3. later-overrides merge that into results/<ds>/tide-<TAG>.json (rerun_helper.py merge)
#   4. re-evals -> eval/<ds>/tide-<TAG>.json
#
# This is idempotent across rounds: the merge target is always tide-<TAG>.json, so
# round 2 naturally targets whatever is STILL recall=0 after round 1. Bump ROUND so
# each round's rerun file/log doesn't clobber the previous one.
#
# NOTE: runs whatever LLM the CODE is configured for; TAG only names outputs.
# Logging is UNBUFFERED (PYTHONUNBUFFERED=1 + main.py line-buffered append log).
# =============================================================================
set -uo pipefail
cd /home/liangzhilin/workspace/TDAgent

TAG="${1:-deepseek}"
GPU="${2:-0}"
ROUND="${3:-1}"
K="${4:-30}"
DATASETS=(OpenWikiTable_tide_v4 WebTable OpenData_SG OpenData_USA GitTable)
DRIVER="logs/TIDE-${TAG}-rerun.log"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
mkdir -p "logs/TIDE-${TAG}"

echo "######## TIDE RERUN round=$ROUND START $(date '+%F %T') | TAG=$TAG GPU=$GPU recall<K=$K ########" | tee -a "$DRIVER"

for ds in "${DATASETS[@]}"; do
  TIDE="results/${ds}/tide-${TAG}.json"
  ITERDIR="results/${ds}/iter-${TAG}"
  RERUN_OUT="${ITERDIR}/${ds}_zeros${ROUND}_rerun.json"
  LOG="logs/TIDE-${TAG}/${ds}-${TAG}-rerun${ROUND}.log"
  mkdir -p "$ITERDIR"

  if [ ! -f "$TIDE" ]; then
    echo "######## [$ds] SKIP — no $TIDE ########" | tee -a "$DRIVER"; continue
  fi

  IDXS="$(conda run -n tide python rerun_helper.py zeros "$ds" "$TIDE" --k "$K")"
  if [ -z "$IDXS" ]; then
    echo "######## [$ds] SKIP — 0 recall=0 queries ########" | tee -a "$DRIVER"; continue
  fi
  ncnt=$(awk -F, '{print NF}' <<< "$IDXS")
  echo "######## [$ds] RERUN START $(date '+%F %T') | $ncnt queries: $IDXS ########" | tee -a "$DRIVER"

  # 1. re-run only the recall=0 / empty idxs
  conda run -n tide python main.py --dataset "$ds" --idxs "$IDXS" --out "$RERUN_OUT" --log "$LOG"
  rc=$?

  # 2. later-overrides merge into the canonical result, then 3. re-eval
  if [ "$rc" -eq 0 ] && [ -f "$RERUN_OUT" ]; then
    conda run -n tide python rerun_helper.py merge "$ds" "$TIDE" "$RERUN_OUT" 2>>"$DRIVER"
    conda run -n tide python eval.py --dataset "$ds" --method "tide-${TAG}" \
        2>&1 | grep -E "Success rate|saved" | tee -a "$DRIVER"
  fi
  echo "######## [$ds] RERUN DONE rc=$rc | $(date '+%F %T') ########" | tee -a "$DRIVER"
done

echo "######## TIDE RERUN round=$ROUND ALL DONE $(date '+%F %T') ########" | tee -a "$DRIVER"
