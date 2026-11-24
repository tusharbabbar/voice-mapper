#!/usr/bin/env bash
# Package a checkpoint into a self-contained model folder, export/<voice>.tar.gz, that runs offline:
#   model.pt (fp16 inference weights), vocab.txt, vocos/ (vocoder), ref.wav + ref.txt (default reference)
#
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
rm -rf "$OUT"; mkdir -p "$OUT/vocos"
python - "$CKPT_DIR/$CKPT" "$OUT/model.pt" "$WEIGHTS" <<'PY'
import sys, torch
src, dst, which = sys.argv[1:4]
ck = torch.load(src, map_location="cpu", weights_only=True)
if which == "ema":
    sd = ck["ema_model_state_dict"]
else:
    sd = {"ema_model." + k: v for k, v in ck["model_state_dict"].items()}
sd = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in sd.items()}
torch.save({"ema_model_state_dict": sd}, dst)
print(f"saved {which} weights from {src} (update {ck.get('update')})")
PY
cp "$F5_DIR/data/${VOICE_NAME}_pinyin/vocab.txt" "$OUT/vocab.txt"

# Vocoder, so generation never needs the internet.
python - "$OUT/vocos" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
for f in ("config.yaml", "pytorch_model.bin"):
    shutil.copy(hf_hub_download("charactr/vocos-mel-24khz", f), f"{sys.argv[1]}/{f}")
PY

# Default reference clip: REF_ID from config.env, else the first 5-10 s clip in the dataset.
python - "$DATASET_DIR" "${REF_ID:-}" "$OUT" <<'PY'
import sys, shutil, wave
from pathlib import Path
ds, ref_id, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
rows = [l.split("|") for l in (ds / "metadata.csv").read_text(encoding="utf-8").splitlines() if "|" in l]
def secs(p):
    with wave.open(str(p)) as w: return w.getnframes() / w.getframerate()
pick = next((r for r in rows if r[0] == ref_id), None) if ref_id else None
if pick is None:
    pick = next(r for r in rows if 5 <= secs(ds / "wavs" / f"{r[0]}.wav") <= 10 and r[1].strip()[-1:] in ".?!")
shutil.copy(ds / "wavs" / f"{pick[0]}.wav", out / "ref.wav")
(out / "ref.txt").write_text(pick[1].strip() + "\n")
print(f"reference clip: {pick[0]}: {pick[1].strip()}")
PY

tar -czf "export/$VOICE_NAME.tar.gz" -C export "$VOICE_NAME"
ls -lh "export/$VOICE_NAME.tar.gz"
echo "Download it to your Mac:  runpodctl send export/$VOICE_NAME.tar.gz"
