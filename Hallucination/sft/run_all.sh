#!/bin/bash
# Run all 5 conditions for the monofact-targeted upweighting experiment.
# Execute from the sft/ directory.
#
# Usage: bash run_all.sh
# Estimated time: ~7-8 hours total on A100 SXM (T5-base)

set -e

DATA_PATH="../data/biography_data.csv"
MODEL="t5-base"
ALPHA=1

echo "============================================"
echo "1/5 — BASELINE (no upweighting)"
echo "============================================"
python3 run_experiment.py \
    --condition baseline \
    --alpha $ALPHA \
    --data_path $DATA_PATH \
    --model_name $MODEL

echo ""
echo "============================================"
echo "2/5 — RANDOM UPWEIGHT (10×)"
echo "============================================"
python3 run_experiment.py \
    --condition random_upweight \
    --alpha $ALPHA \
    --data_path $DATA_PATH \
    --model_name $MODEL

echo ""
echo "============================================"
echo "3/5 — MONOFACT UPWEIGHT (10×)"
echo "============================================"
python3 run_experiment.py \
    --condition monofact_upweight \
    --alpha $ALPHA \
    --data_path $DATA_PATH \
    --model_name $MODEL

echo ""
echo "============================================"
echo "4/5 — MONOFACT UPWEIGHT (2× dose-controlled)"
echo "============================================"
python3 run_experiment.py \
    --condition monofact_upweight \
    --alpha $ALPHA \
    --data_path $DATA_PATH \
    --model_name $MODEL \
    --duplications 2

echo ""
echo "============================================"
echo "5/5 — MIXED UPWEIGHT (50/50, 10×)"
echo "============================================"
python3 run_experiment.py \
    --condition mixed_upweight \
    --alpha $ALPHA \
    --data_path $DATA_PATH \
    --model_name $MODEL

echo ""
echo "============================================"
echo "ALL 5 RUNS COMPLETE"
echo "============================================"
echo "Results in sft/results/"
ls -la results/*_final.csv
