# voice-mapper

Fine-tune [F5-TTS](https://github.com/SWivid/F5-TTS) v1 Base on recordings of one voice, then generate speech in that voice locally.

- **Train** on a rented GPU (one 24 GB card such as an RTX 4090 is enough).
- **Generate** on your own machine, including a Mac; no GPU needed.

This repo holds code only. Voice data, checkpoints and generated audio are git-ignored on purpose.

## What you need

- A voice dataset in LJSpeech format: `metadata.csv` (`id|text|normalized_text`) plus `wavs/<id>.wav`. 30-90 minutes of clean speech from one speaker works well.
- A GPU machine for training, e.g. a [RunPod](https://www.runpod.io) pod with the PyTorch template, an RTX 4090 (24 GB) and a 50 GB volume at `/workspace`.

## Steps

### 1. (Optional) Leave out clips you don't want
Listen through the dataset and put the ids of bad clips in `exclude.txt`, one per line:
```
sf_0018
sf_0037
```

### 2. Get the code and data onto the GPU machine
On the pod:
```bash
cd /workspace
git clone https://github.com/tusharbabbar/voice-mapper.git   # private repo: use `gh auth login` or a token
```
Copy the dataset over. With RunPod's `runpodctl` (installed on pods; on a Mac: `brew install runpod/runpodctl/runpodctl`):
```bash
# on your Mac
cd ~/Movies/Systems__first && zip -qr voice_dataset.zip voice_dataset -x 'voice_dataset/dropped/*'
runpodctl send voice_dataset.zip            # prints a one-time code

# on the pod
cd /workspace && runpodctl receive <code> && unzip -q voice_dataset.zip
```

### 3. Train (one command)
```bash
cd /workspace/voice-mapper
nohup ./scripts/runpod_all.sh > train.log 2>&1 &
tail -f train.log
```
`runpod_all.sh` does three things:
- installs F5-TTS, pinned to the version this repo was tested with;
- prepares the data, downloads the base model and fine-tunes;
- exports the latest checkpoint.

Each step can also be run on its own: `setup_gpu.sh`, `train.sh`, `export.sh`. Re-running resumes from the last checkpoint.

- Checkpoints go to `/workspace/F5-TTS/ckpts/<VOICE_NAME>/`, with sample audio in `samples/` every `SAVE_EVERY` updates. `update_N_gen.wav` is the model speaking; `update_N_ref.wav` is the real clip.
- To watch progress: `tensorboard --logdir /workspace/F5-TTS/runs`.
- Rough cost: 1 hour of data at the defaults is about 5,000 updates; expect 1-3 hours on a 4090, i.e. a few dollars.
- Stop the pod when you're done; it bills by the second.

### 4. Pick a checkpoint and export it
Listen to `samples/`. Pick the earliest checkpoint that sounds like you and still sounds natural:
- If later checkpoints start sounding strained or garbled, use an earlier one.
- If none sound like you yet, train longer (raise `EPOCHS`).

The one-command run already exported the latest checkpoint. To export a different one:
```bash
./scripts/export.sh model_4000.pt
runpodctl send export/<VOICE_NAME>.tar.gz
```
The export is a self-contained folder: model weights (fp16), vocab, the vocoder, and a default reference clip with its transcript. Generating speech needs no internet.

### 5. Generate speech on your Mac
```bash
python3 -m venv ~/f5-venv && source ~/f5-venv/bin/activate && pip install f5-tts num2words
cd ~/Movies/Systems__first && git clone https://github.com/tusharbabbar/voice-mapper.git && cd voice-mapper
mkdir -p models && runpodctl receive <code> && tar -xzf <VOICE_NAME>.tar.gz -C models

python scripts/speak.py "Here is the catch though. Did we just move the single point of failure?" -o out/line.wav
python scripts/speak.py -f my_script.txt -o out/reel_vo.wav --speed 0.95
```
`speak.py` details:
- It spells out numbers the way the training text was written ("2016" becomes "twenty sixteen").
- It uses the Mac's GPU when available.
- It prints a seed, so you can regenerate the same take with `--seed`.

The reference clip sets the delivery: tone, pace and energy. Try others from your dataset with `--ref <clip.wav> --ref-text "<its exact words>"`.

## Settings (`config.env`)

| Setting | Default | Notes |
|---|---|---|
| `BATCH_FRAMES` | 6400 | Audio frames per batch (~94 per second of audio). Lower it if you run out of GPU memory; raise it on 48-80 GB cards. |
| `LEARNING_RATE` | 1e-5 | The F5-TTS fine-tuning default. |
| `EPOCHS` | 80 | Passes over the data. |
| `WARMUP_UPDATES` | 500 | F5-TTS's default (20,000) is meant for pre-training and too long for a small dataset. |
| `REF_ID` | sf_0327 | Dataset clip bundled as the default reference voice. |
| `SAVE_EVERY` / `KEEP_LAST` | 1000 / 5 | Checkpoint + samples every N updates; keeps the last 5 (~5 GB each). |

Any setting can be overridden per run: `EPOCHS=40 ./scripts/train.sh`.

## Notes

- **Which weights are exported.** `export.sh` exports the online weights by default, not F5-TTS's EMA copy. The EMA copy updates every 10 steps with β = 0.9999, so after a few thousand fine-tuning steps it is still mostly the base model. The training samples are generated with the online weights, so the export matches what you heard. Use `WEIGHTS=ema ./scripts/export.sh` only after very long runs.
- **Text.** Training uses the dataset's normalized text (numbers spelled out); `speak.py` applies the same normalization to what you give it.
- **Licence.** F5-TTS code is MIT, but the pre-trained base weights are CC-BY-NC 4.0 (non-commercial), and a fine-tune inherits that. Check before using the output commercially or in monetised content.
- **Consent.** Only train on voices you have the right to use.

## Layout

```
config.env                  settings
scripts/setup_gpu.sh        install F5-TTS on the GPU machine
scripts/prepare_dataset.py  LJSpeech -> F5-TTS CSV (absolute paths, optional exclude list)
scripts/runpod_all.sh       setup + train + export in one go
scripts/train.sh            prepare + fine-tune
scripts/export.sh           checkpoint -> self-contained model folder (.tar.gz)
scripts/speak.py            generate speech with the exported model (offline)
```
