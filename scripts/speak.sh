#!/usr/bin/env bash
# Generate speech in your voice on your Mac (or any machine) with the exported model.
#
# One-time setup on the Mac:
#   python3 -m venv ~/f5-venv && source ~/f5-venv/bin/activate && pip install f5-tts
#
# Usage:
#   ./scripts/speak.sh "Text to say."                      -> out/speech.wav
#   ./scripts/speak.sh -f script.txt -o out/reel.wav       (text from a file)
#
# Needs: models/<voice>/model.pt + vocab.txt (unpacked export), and a reference clip of you
# with its exact transcript (REF_WAV / REF_TEXT below; a clean 5-10 s clip from your dataset works).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_config.sh

MODEL_DIR=${MODEL_DIR:-models/$VOICE_NAME}
REF_WAV=${REF_WAV:-ref/ref.wav}
REF_TEXT=${REF_TEXT:-$(cat ref/ref.txt 2>/dev/null || true)}
OUT=out/speech.wav; TEXT=""; TEXT_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    -f) TEXT_FILE="$2"; shift 2;;
    -o) OUT="$2"; shift 2;;
    *)  TEXT="$1"; shift;;
  esac
done
[ -f "$MODEL_DIR/model.pt" ] || { echo "Model not found in $MODEL_DIR (unpack export/$VOICE_NAME.tar.gz into models/)"; exit 1; }
[ -f "$REF_WAV" ] && [ -n "$REF_TEXT" ] || { echo "Put a reference clip at ref/ref.wav and its exact words in ref/ref.txt"; exit 1; }

GEN=(--gen_text "$TEXT"); [ -n "$TEXT_FILE" ] && GEN=(--gen_file "$TEXT_FILE")
mkdir -p "$(dirname "$OUT")"
f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ckpt_file "$MODEL_DIR/model.pt" \
  --vocab_file "$MODEL_DIR/vocab.txt" \
  --ref_audio "$REF_WAV" \
  --ref_text "$REF_TEXT" \
  "${GEN[@]}" \
  --output_dir "$(dirname "$OUT")" \
  --output_file "$(basename "$OUT")"
echo "-> $OUT"
