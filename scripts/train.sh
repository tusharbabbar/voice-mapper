#!/usr/bin/env bash
# Prepare the dataset and fine-tune F5-TTS v1 Base on it.
# Usage: ./scripts/train.sh            (settings from config.env)
#        EPOCHS=40 ./scripts/train.sh  (override any setting)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
source scripts/_config.sh

DATA_NAME="$VOICE_NAME"                     # F5-TTS reads data/${DATA_NAME}_pinyin
PREP_DIR="$F5_DIR/data/${DATA_NAME}_pinyin"

echo "== 1/2 Preparing dataset from $DATASET_DIR"
EXCLUDE_ARG=()
[ -f "$REPO/exclude.txt" ] && EXCLUDE_ARG=(--exclude "$REPO/exclude.txt")
python "$REPO/scripts/prepare_dataset.py" "$DATASET_DIR" "$REPO/data/${DATA_NAME}.csv" "${EXCLUDE_ARG[@]}"
# Fine-tune mode (no --pretrain flag): keeps the base model's vocabulary.
python "$F5_DIR/src/f5_tts/train/datasets/prepare_csv_wavs.py" "$REPO/data/${DATA_NAME}.csv" "$PREP_DIR"

echo "== 2/2 Fine-tuning (checkpoints -> $F5_DIR/ckpts/$DATA_NAME)"
cd "$F5_DIR"
accelerate launch --mixed_precision="$MIXED_PRECISION" src/f5_tts/train/finetune_cli.py \
  --exp_name F5TTS_v1_Base \
  --finetune \
  --dataset_name "$DATA_NAME" \
  --tokenizer pinyin \
  --learning_rate "$LEARNING_RATE" \
  --batch_size_type frame \
  --batch_size_per_gpu "$BATCH_FRAMES" \
  --max_samples 64 \
  --epochs "$EPOCHS" \
  --num_warmup_updates "$WARMUP_UPDATES" \
  --save_per_updates "$SAVE_EVERY" \
  --last_per_updates "$LAST_EVERY" \
  --keep_last_n_checkpoints "$KEEP_LAST" \
  --log_samples \
  --logger tensorboard

echo "Done. Listen to the samples in $F5_DIR/ckpts/$DATA_NAME/samples, then run ./scripts/export.sh"
