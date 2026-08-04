#!/usr/bin/env bash
# Week 2 training campaign for the Train-Short-Infer-Long paper.
# Runs the MTL headline + 3 single-task baselines back-to-back, single seed each.
# Total wall-clock: ~40 hours on RTX 4070 12GB. Run from project root.
#
# Usage:
#   chmod +x scripts/run_w2_campaign.sh
#   ./scripts/run_w2_campaign.sh
#   # (or to detach from terminal: nohup ./scripts/run_w2_campaign.sh &)
#
# Outputs:
#   outputs/w2_mtl_v3/seed_17/         — joint MTL (headline)
#   outputs/w2_baselines/sum_seed_17/  — single-task summarization baseline
#   outputs/w2_baselines/emo_seed_17/  — single-task emotion baseline
#   outputs/w2_baselines/top_seed_17/  — single-task topic baseline
#
# Author: Oliver Perrin
# Date: April 2026

set -euo pipefail
cd "$(dirname "$0")/.."

SEED=17

run() {
    local cfg=$1
    local out=$2
    mkdir -p "$out/checkpoints"
    echo
    echo "============================================================"
    echo "STARTING: $cfg -> $out (seed=$SEED) at $(date)"
    echo "============================================================"
    python scripts/train.py training="$cfg" seed=$SEED \
        checkpoint_out="$out/checkpoints/best.pt" \
        history_out="$out/training_history.json" \
        labels_out="$out/labels.json" \
        2>&1 | tee "$out/train.log"
}

run full                  outputs/w2_mtl_v3/seed_17
run single_summarization  outputs/w2_baselines/sum_seed_17
run single_emotion        outputs/w2_baselines/emo_seed_17
run single_topic          outputs/w2_baselines/top_seed_17

echo
echo "============================================================"
echo "ALL RUNS COMPLETE at $(date)"
echo "============================================================"
