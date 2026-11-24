#!/usr/bin/env bash
# One-time setup on the GPU machine (e.g. a RunPod PyTorch pod). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_config.sh

# F5-TTS version this repo was written against. Change only if you know why.
F5_COMMIT=${F5_COMMIT:-600e5fdfbf319f84735557aca80a7aaf60d487e6}

if ! command -v ffmpeg >/dev/null; then
  echo "Installing ffmpeg..."
  (apt-get update -qq && apt-get install -y -qq ffmpeg) || echo "Could not install ffmpeg; install it manually."
fi

if [ ! -d "$F5_DIR/.git" ]; then
  git clone https://github.com/SWivid/F5-TTS "$F5_DIR"
fi
git -C "$F5_DIR" fetch -q origin "$F5_COMMIT" 2>/dev/null || true
git -C "$F5_DIR" checkout -q "$F5_COMMIT"

# Editable install: F5-TTS looks for data/ and ckpts/ inside its own folder.
pip install -q -e "$F5_DIR" tensorboard

python - <<'PY'
import torch
assert torch.cuda.is_available(), "No CUDA GPU visible - training would be extremely slow."
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GB")
PY
echo "Setup done. Next: ./scripts/train.sh"
