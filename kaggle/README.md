# Train for free on Kaggle

`voice_mapper_kaggle.ipynb` does everything inside one Kaggle notebook:
- installs F5-TTS;
- trains on two free T4 GPUs;
- stops itself before Kaggle's 12-hour limit;
- saves the model, ready for `scripts/speak.py`, plus audio samples to the notebook's Output.

It doesn't need access to this repo, so the repo can stay private.

## One-time setup
1. Create a Kaggle account and verify your phone number: **Settings → Phone verification**. Without this, Kaggle won't give you GPUs or internet in notebooks.

## Each training run
1. **Upload the dataset (private):**
   - Go to kaggle.com → **Create → New Dataset**.
   - Upload `voice_dataset_kaggle.zip`, the dataset folder zipped without `dropped/`.
   - Keep it **Private** and click Create. Kaggle unzips it automatically.
2. **Import the notebook:**
   - Go to **Create → New Notebook → File → Import Notebook**.
   - Upload `voice_mapper_kaggle.ipynb`.
3. **Set up the notebook** in the right sidebar:
   - **Session options → Accelerator:** GPU T4 x2.
   - **Session options → Internet:** On.
   - **Input → + Add Input:** your dataset from step 1.
4. **Check the settings cell** at the top: `VOICE_NAME`, `EPOCHS`, `TIME_LIMIT_HOURS`. The defaults suit about an hour of audio.
5. **Run it:** click **Save Version**, choose **Save & Run All (Commit)**, then Save. It runs in the background, so you can close the tab or your laptop.
6. **When it finishes** (Kaggle emails you), open the notebook version → **Output**:
   - Listen to `samples/update_*_gen.wav`, the model speaking at each checkpoint.
   - Download `<VOICE_NAME>.tar.gz` and unpack it on your Mac into `voice-mapper/models/`.
7. **Generate speech** on your Mac:
   ```bash
   python scripts/speak.py "Hello, this is my voice." -o out/hello.wav
   ```

## Good to know
- **Quota:** Kaggle gives about 30 GPU hours a week. A run on an hour of audio should take roughly 6–10 hours.
- **Out of memory?** If the log shows `CUDA out of memory`, set `BATCH_FRAMES = 2400`, or lower, and run again.
- **Multi-GPU errors?** If training fails with errors mentioning NCCL or distributed training, set `NUM_GPUS = 1`. It's slower but simpler.
- **Exporting an earlier checkpoint:** the export uses `model_last.pt`. If an earlier sample sounded better, open the notebook interactively, set `EXPORT_CKPT` in the export cell and run the cells again. Only the last `KEEP_LAST` checkpoints are kept, because disk space is limited.
- **Editing the notebook:** `build_notebook.py` generates it, so edit that file and re-run it rather than editing the `.ipynb` by hand.
