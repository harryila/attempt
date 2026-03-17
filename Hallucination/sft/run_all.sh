#!/bin/bash
# Run all 3 conditions for the monofact-targeted upweighting experiment.
# Execute from the sft/ directory.
#
# Usage: bash run_all.sh
# Estimated time: ~4-6 hours total on A100 (T5-base)

set -e

DATA_PATH="../data/biography_data.csv"
MODEL="t5-base"
ALPHA=1  # primary setting (~27-32% monofact rate)

echo "============================================"
echo "Running BASELINE (no upweighting)"
echo "============================================"
python3 run_experiment.py \
    --condition baseline \
    --alpha $ALPHA \
    --data_path $DATA_PATH \
    --model_name $MODEL

echo ""
echo "============================================"
echo "Running RANDOM UPWEIGHT"
echo "============================================"
python3 run_experiment.py \
    --condition random_upweight \
    --alpha $ALPHA \
    --data_path $DATA_PATH \
    --model_name $MODEL

echo ""
echo "============================================"
echo "Running MONOFACT-TARGETED UPWEIGHT"
echo "============================================"
python3 run_experiment.py \
    --condition monofact_upweight \
    --alpha $ALPHA \
    --data_path $DATA_PATH \
    --model_name $MODEL

echo ""
echo "============================================"
echo "ALL RUNS COMPLETE"
echo "============================================"
echo "Results in sft/results/"
ls -la results/*_final.csv

# --- OPTIONAL: run alpha=1.5 as secondary ---
# Uncomment below if you have time:
#
# for COND in baseline random_upweight monofact_upweight; do
#     python3 run_experiment.py --condition $COND --alpha 1.5 --data_path $DATA_PATH --model_name $MODEL
# done
