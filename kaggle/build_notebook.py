"""Builds kaggle/voice_mapper_kaggle.ipynb from the cells below. Run: python kaggle/build_notebook.py"""
import json
from pathlib import Path

CELLS = []
def md(s): CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# voice-mapper: fine-tune F5-TTS on your voice (free, on Kaggle)

**Before running, set these in the right sidebar:**
1. **Accelerator:** GPU T4 x2.
2. **Internet:** On. This needs a phone-verified Kaggle account.
3. **Input:** add your private dataset containing `metadata.csv` and `wavs/`.

Then click **Save Version, then Save & Run All (Commit)**. It runs in the background, so you can close the tab.

**What you get:**
- **Output tab:** `<VOICE_NAME>.tar.gz` (the model, ready for `scripts/speak.py` on your Mac), `samples/` (the model speaking at each checkpoint) and `train.log`.
- **Time limit:** training stops itself before Kaggle's 12-hour limit, and the latest checkpoint is exported either way.
""")

code(r"""
# ---- Settings ----
VOICE_NAME       = "tushar"
REF_ID           = "sf_0327"   # dataset clip used as the default reference voice ("" = first clean 5-10 s clip)
EPOCHS           = 60          # passes over the data; training also stops at TIME_LIMIT_HOURS
BATCH_FRAMES     = 3200        # audio frames per GPU per batch (~94 per second of audio); lower if out of memory
LEARNING_RATE    = 1e-5
WARMUP_UPDATES   = 500
SAVE_EVERY       = 500         # checkpoint + audio sample every N updates
LAST_EVERY       = 100         # refresh model_last.pt every N updates (what gets exported if time runs out)
KEEP_LAST        = 2           # numbered checkpoints kept on disk (~5 GB each)
TIME_LIMIT_HOURS = 10.5        # stop training after this, leaving time to export within Kaggle's 12 h
NUM_GPUS         = None        # None = all visible GPUs (2 on T4 x2); set 1 if multi-GPU gives trouble
F5_COMMIT        = "600e5fdfbf319f84735557aca80a7aaf60d487e6"   # F5-TTS version this was written against

import os, time
WORK   = "/kaggle/working"
F5_DIR = "/tmp/F5-TTS"               # code + data + checkpoints (not saved to Output)
T0 = time.time()
""")

code(r"""
# ---- Check GPU and internet ----
import subprocess, torch, urllib.request
assert torch.cuda.is_available(), "No GPU. Set Accelerator to 'GPU T4 x2' in the sidebar."
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(f"GPU {i}: {p.name}, {p.total_memory/2**30:.0f} GB")
try:
    urllib.request.urlopen("https://huggingface.co", timeout=10); print("Internet: OK")
except Exception as e:
    raise SystemExit("No internet. Turn Internet on in the sidebar (needs a phone-verified account).") from e
if NUM_GPUS is None:
    NUM_GPUS = torch.cuda.device_count()
""")

code(r"""
# ---- Find the dataset ----
import glob
from pathlib import Path
hits = sorted(glob.glob("/kaggle/input/**/metadata.csv", recursive=True), key=len)
assert hits, "metadata.csv not found under /kaggle/input. Add your dataset with '+ Add Input'."
DATASET_DIR = Path(hits[0]).parent
assert (DATASET_DIR / "wavs").is_dir(), f"No wavs/ folder next to {hits[0]}"
n = sum(1 for l in open(hits[0], encoding="utf-8") if "|" in l)
print(f"Dataset: {DATASET_DIR} ({n} clips)")
""")

code(r"""
# ---- Install F5-TTS (pinned) ----
import subprocess, sys
def sh(cmd):
    print("$", cmd); subprocess.run(cmd, shell=True, check=True)
if not os.path.isdir(f"{F5_DIR}/.git"):
    sh(f"git clone -q https://github.com/SWivid/F5-TTS {F5_DIR}")
sh(f"git -C {F5_DIR} checkout -q {F5_COMMIT}")
sh(f"{sys.executable} -m pip install -q -e {F5_DIR}")
""")

code(r"""
# ---- Prepare the data (F5-TTS CSV -> arrow) ----
import csv, wave
def secs(p):
    with wave.open(str(p)) as w: return w.getnframes() / w.getframerate()
rows, total = [], 0.0
for line in open(DATASET_DIR / "metadata.csv", encoding="utf-8"):
    parts = line.rstrip("\n").split("|")
    if len(parts) < 2: continue
    cid, text = parts[0].strip(), (parts[2] if len(parts) > 2 else parts[1]).strip()   # normalized text
    wav = DATASET_DIR / "wavs" / f"{cid}.wav"
    if not text or not wav.exists(): continue
    s = secs(wav)
    if 1.0 <= s <= 30.0:
        rows.append((str(wav), text.replace("|", " "))); total += s
os.makedirs(f"{WORK}/prep", exist_ok=True)
CSV_PATH = f"{WORK}/prep/{VOICE_NAME}.csv"
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
    w.writerow(["audio_file", "text"]); w.writerows(rows)
print(f"{len(rows)} clips, {total/60:.1f} min")
PREP_DIR = f"{F5_DIR}/data/{VOICE_NAME}_pinyin"
sh(f"{sys.executable} {F5_DIR}/src/f5_tts/train/datasets/prepare_csv_wavs.py {CSV_PATH} {PREP_DIR}")
""")

code(r"""
# ---- Fine-tune (stops itself at TIME_LIMIT_HOURS) ----
import signal
CKPT_DIR = f"{F5_DIR}/ckpts/{VOICE_NAME}"
LOG = f"{WORK}/train.log"
launch = f"accelerate launch --mixed_precision fp16 --num_processes {NUM_GPUS}" + (" --multi_gpu" if NUM_GPUS > 1 else "")
cmd = (f"cd {F5_DIR} && {launch} src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --finetune "
       f"--dataset_name {VOICE_NAME} --tokenizer pinyin --learning_rate {LEARNING_RATE} "
       f"--batch_size_type frame --batch_size_per_gpu {BATCH_FRAMES} --max_samples 64 --epochs {EPOCHS} "
       f"--num_warmup_updates {WARMUP_UPDATES} --save_per_updates {SAVE_EVERY} --last_per_updates {LAST_EVERY} "
       f"--keep_last_n_checkpoints {KEEP_LAST} --log_samples")
print(cmd)
deadline = T0 + TIME_LIMIT_HOURS * 3600
with open(LOG, "a") as log:
    proc = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    last_print = 0
    while proc.poll() is None:
        time.sleep(30)
        if time.time() - last_print > 600:          # progress every 10 min
            last_print = time.time()
            tail = open(LOG, errors="ignore").read()[-400:].replace("\r", "\n").strip().splitlines()
            print(f"[{(time.time()-T0)/3600:.1f} h] {tail[-1][:200] if tail else ''}", flush=True)
        if time.time() > deadline:
            print("Time limit reached; stopping training (latest model_last.pt will be exported).")
            os.killpg(proc.pid, signal.SIGINT); time.sleep(60)
            if proc.poll() is None: os.killpg(proc.pid, signal.SIGKILL)
            break
print("training exit code:", proc.returncode)
print(open(LOG, errors="ignore").read()[-1500:])
""")

code(r"""
# ---- Copy audio samples to Output ----
import shutil
os.makedirs(f"{WORK}/samples", exist_ok=True)
for f in sorted(glob.glob(f"{CKPT_DIR}/samples/*.wav")):
    shutil.copy(f, f"{WORK}/samples/")
print(sorted(os.listdir(f"{WORK}/samples"))[-6:])
print("checkpoints:", sorted(os.listdir(CKPT_DIR)))
""")

code(r"""
# ---- Export a self-contained model folder -> Output/<VOICE_NAME>.tar.gz ----
# Uses the "online" weights: F5-TTS's EMA copy (beta 0.9999, every 10 steps) is still mostly the base
# model after a short fine-tune, and the training samples above were made with the online weights.
from huggingface_hub import hf_hub_download
EXPORT_CKPT = "model_last.pt"          # or e.g. "model_2000.pt" after listening to samples/
OUT = f"/tmp/export/{VOICE_NAME}"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(f"{OUT}/vocos")
ck = torch.load(f"{CKPT_DIR}/{EXPORT_CKPT}", map_location="cpu", weights_only=True)
sd = {"ema_model." + k: (v.half() if torch.is_floating_point(v) else v) for k, v in ck["model_state_dict"].items()}
torch.save({"ema_model_state_dict": sd}, f"{OUT}/model.pt"); del ck, sd
print("exported update", torch.load(f"{CKPT_DIR}/{EXPORT_CKPT}", map_location="cpu", weights_only=True).get("update"))
shutil.copy(f"{PREP_DIR}/vocab.txt", f"{OUT}/vocab.txt")
for f in ("config.yaml", "pytorch_model.bin"):
    shutil.copy(hf_hub_download("charactr/vocos-mel-24khz", f), f"{OUT}/vocos/{f}")
meta = [l.rstrip("\n").split("|") for l in open(DATASET_DIR / "metadata.csv", encoding="utf-8") if "|" in l]
pick = next((r for r in meta if r[0] == REF_ID), None) if REF_ID else None
if pick is None:
    pick = next(r for r in meta if 5 <= secs(DATASET_DIR / "wavs" / f"{r[0]}.wav") <= 10 and r[1].strip()[-1:] in ".?!")
shutil.copy(DATASET_DIR / "wavs" / f"{pick[0]}.wav", f"{OUT}/ref.wav")
open(f"{OUT}/ref.txt", "w").write(pick[1].strip() + "\n")
sh(f"tar -czf {WORK}/{VOICE_NAME}.tar.gz -C /tmp/export {VOICE_NAME}")
shutil.rmtree(f"{WORK}/prep", ignore_errors=True)
print(f"Done in {(time.time()-T0)/3600:.1f} h -> Output: {VOICE_NAME}.tar.gz, samples/, train.log")
""")

md(r"""
## Next
1. **Listen.** In the Output tab, play `samples/update_*_gen.wav` (the model) next to `update_*_ref.wav` (the real clip).
2. **Download** `<VOICE_NAME>.tar.gz` and unpack it into `voice-mapper/models/` on your Mac.
3. **Generate:** `python scripts/speak.py "Hello" -o out/hello.wav`.

If an earlier sample sounded better than the last one, set `EXPORT_CKPT` in the export cell and re-run that cell in an interactive session. Only the last `KEEP_LAST` numbered checkpoints are kept.
""")

nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": s} if t == "markdown" else
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})
        for t, s in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).with_name("voice_mapper_kaggle.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
