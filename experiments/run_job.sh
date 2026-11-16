#!/bin/bash
# one job = one (config, variant, seed). first jobs (idx 0-2) start 100 s apart so the data loading peaks of RAM do not coincide.
cd /c/Users/rlaxo/OneDrive/Desktop/Deep_Learning_Project/experiments
idx=$1; cfg=$2; var=$3; seed=$4
if [ "$idx" -lt 9 ]; then sleep $((idx * 100)); fi
echo "START $cfg $var s$seed $(date +%H:%M:%S)"
python -X utf8 run.py --exp config/$cfg.json --only $var --seeds $seed 2>&1 | grep -E "done in|Traceback|Error|MemoryError"
echo "END $cfg $var s$seed $(date +%H:%M:%S)"
