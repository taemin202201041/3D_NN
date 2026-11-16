#!/usr/bin/env bash
# Runs experiment files one after another. Each run gets a FRESH Python process
# (one long process slowed down ~2.6x over many runs). Finished runs are skipped via the ledger.
# usage: bash run_queue.sh config/e07_prune.json config/e02_epochs.json ...
cd "$(dirname "$0")"
PY=${PY:-python}   # 다른 파이썬을 쓰려면: PY=/경로/python bash run_queue.sh ...
mkdir -p logs
for exp in "$@"; do
  name=$(basename "$exp" .json)
  echo "=== $(date '+%F %T') start $name" | tee -a logs/queue.log
  while PYTHONUTF8=1 "$PY" run.py --exp "$exp" --dry-run 2>/dev/null | grep -q '^run '; do
    PYTHONUTF8=1 "$PY" run.py --exp "$exp" --max-runs 1 >> "logs/$name.log" 2>&1 || { echo "=== $(date '+%F %T') FAILED step in $name" | tee -a logs/queue.log; break; }
  done
  echo "=== $(date '+%F %T') end $name" | tee -a logs/queue.log
done
