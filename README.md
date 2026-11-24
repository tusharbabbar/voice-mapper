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
cd ~/Movies/Systems__first && zip -qr voice_dataset.zip voice_dataset
runpodctl send voice_dataset.zip            # prints a one-time code

# on the pod
cd /workspace && runpodctl receive <code> && unzip -q voice_dataset.zip
```

### 3. Set up and train
```bash
cd /workspace/voice-mapper
nano config.env          # set VOICE_NAME; check DATASET_DIR
./scripts/setup_gpu.sh   # installs F5-TTS (pinned version) and checks the GPU
./scripts/train.sh       # prepares the data, downloads the base model, fine-tunes
```
- Each run saves checkpoints to `/workspace/F5-TTS/ckpts/<VOICE_NAME>/`, plus sample audio in `samples/` every `SAVE_EVERY` updates. `update_N_gen.wav` is the model speaking; `update_N_ref.wav` is the real clip.
- To watch progress: `tensorboard --logdir /workspace/F5-TTS/runs`.
- Rough cost: 1 hour of data at the defaults is about 5,000 updates; expect 1-3 hours on a 4090, i.e. a few dollars.
- Stop the pod when you're done; it bills by the second.

### 4. Pick a checkpoint and export it
Listen to `samples/`. Pick the earliest checkpoint that sounds like you and still sounds natural:
- If later checkpoints start sounding strained or garbled, use an earlier one.
- If none sound like you yet, train longer (raise `EPOCHS`).

```bash
./scripts/export.sh model_4000.pt     # or no argument for the latest
runpodctl send export/<VOICE_NAME>.tar.gz
```

### 5. Generate speech on your Mac
```bash
python3 -m venv ~/f5-venv && source ~/f5-venv/bin/activate && pip install f5-tts
git clone https://github.com/tusharbabbar/voice-mapper.git && cd voice-mapper
mkdir -p models ref && runpodctl receive <code> && tar -xzf <VOICE_NAME>.tar.gz -C models

# a clean 5-10 s clip of you, and its exact words
cp ~/Movies/Systems__first/voice_dataset/wavs/sf_0327.wav ref/ref.wav
echo "Toppers don't have an exceptional memory. They use a system that helps them manage their memory properly." > ref/ref.txt

./scripts/speak.sh "Here is the catch though. Did we just move the single point of failure?"
./scripts/speak.sh -f my_script.txt -o out/reel_vo.wav
```
The reference clip sets the delivery: tone, pace and energy. Try a few different ones to get the read you want.

## Settings (`config.env`)

| Setting | Default | Notes |
|---|---|---|
| `BATCH_FRAMES` | 6400 | Audio frames per batch (~94 per second of audio). Lower it if you run out of GPU memory; raise it on 48-80 GB cards. |
| `LEARNING_RATE` | 1e-5 | The F5-TTS fine-tuning default. |
| `EPOCHS` | 80 | Passes over the data. |
| `WARMUP_UPDATES` | 500 | F5-TTS's default (20,000) is meant for pre-training and too long for a small dataset. |
| `SAVE_EVERY` / `KEEP_LAST` | 1000 / 5 | Checkpoint + samples every N updates; keeps the last 5 (~5 GB each). |

Any setting can be overridden per run: `EPOCHS=40 ./scripts/train.sh`.

## Notes

- **Which weights are exported.** `export.sh` exports the online weights by default, not F5-TTS's EMA copy. The EMA copy updates every 10 steps with β = 0.9999, so after a few thousand fine-tuning steps it is still mostly the base model. The training samples are generated with the online weights, so the export matches what you heard. Use `WEIGHTS=ema ./scripts/export.sh` only after very long runs.
- **Text.** Training uses the dataset's normalized text (numbers spelled out). Write scripts for `speak.sh` the same way ("twenty sixteen", "eighty percent") for the most reliable reading.
- **Licence.** F5-TTS code is MIT, but the pre-trained base weights are CC-BY-NC 4.0 (non-commercial), and a fine-tune inherits that. Check before using the output commercially or in monetised content.
- **Consent.** Only train on voices you have the right to use.

## Layout

```
config.env                  settings
scripts/setup_gpu.sh        install F5-TTS on the GPU machine
scripts/prepare_dataset.py  LJSpeech -> F5-TTS CSV (absolute paths, optional exclude list)
scripts/train.sh            prepare + fine-tune
scripts/export.sh           checkpoint -> small inference model + vocab (.tar.gz)
scripts/speak.sh            generate speech with the exported model
```
