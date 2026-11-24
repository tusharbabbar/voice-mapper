#!/usr/bin/env bash
# Everything on the GPU machine in one go: setup -> fine-tune -> export the latest checkpoint.
# Safe to re-run: training resumes from the last checkpoint.
#
#   nohup ./scripts/runpod_all.sh > train.log 2>&1 &
#   tail -f train.log
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_config.sh
[ -f "$DATASET_DIR/metadata.csv" ] || { echo "Dataset not found at $DATASET_DIR (set DATASET_DIR in config.env)"; exit 1; }
echo "[$(date)] setup";  ./scripts/setup_gpu.sh
echo "[$(date)] train";  ./scripts/train.sh
echo "[$(date)] export"; ./scripts/export.sh
echo "[$(date)] all done -> export/$VOICE_NAME.tar.gz"
