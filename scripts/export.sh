#!/usr/bin/env bash
# Package a checkpoint + vocab into export/<voice>.tar.gz to download to your Mac.
# Usage: ./scripts/export.sh                       (latest checkpoint, model_last.pt)
#        ./scripts/export.sh model_3000.pt         (a specific one you preferred)
#        WEIGHTS=ema ./scripts/export.sh           (EMA weights instead; see below)
#
# Why "online" weights by default: F5-TTS keeps an EMA (slow moving average) copy that updates
# every 10 steps with beta 0.9999, so after a few thousand fine-tuning steps it is still mostly
# the base model. The training samples in ckpts/<voice>/samples are made with the online weights,
# so exporting those gives you the voice you heard there. Use WEIGHTS=ema only for very long runs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_config.sh
WEIGHTS=${WEIGHTS:-online}
CKPT_DIR="$F5_DIR/ckpts/$VOICE_NAME"
CKPT="${1:-model_last.pt}"
[ -f "$CKPT_DIR/$CKPT" ] || { echo "Not found: $CKPT_DIR/$CKPT"; ls "$CKPT_DIR"; exit 1; }

OUT="export/$VOICE_NAME"
mkdir -p "$OUT"
# Save only inference weights (~1.3 GB instead of ~5 GB), in the layout f5-tts_infer-cli loads.
python - "$CKPT_DIR/$CKPT" "$OUT/model.pt" "$WEIGHTS" <<'PY'
import sys, torch
src, dst, which = sys.argv[1:4]
ck = torch.load(src, map_location="cpu", weights_only=True)
if which == "ema":
    sd = ck["ema_model_state_dict"]
else:
    sd = {"ema_model." + k: v for k, v in ck["model_state_dict"].items()}
torch.save({"ema_model_state_dict": sd}, dst)
print(f"saved {which} weights from {src} (update {ck.get('update')})")
PY
cp "$F5_DIR/data/${VOICE_NAME}_pinyin/vocab.txt" "$OUT/vocab.txt"
tar -czf "export/$VOICE_NAME.tar.gz" -C export "$VOICE_NAME"
ls -lh "export/$VOICE_NAME.tar.gz"
echo "Download export/$VOICE_NAME.tar.gz to your Mac (e.g. runpodctl send export/$VOICE_NAME.tar.gz)."
