#!/usr/bin/env python3
"""
Turn an LJSpeech-style voice dataset into the CSV that F5-TTS fine-tuning expects.

Input  (LJSpeech):  <dataset>/metadata.csv   id|text|normalized_text
                    <dataset>/wavs/<id>.wav
Output (F5-TTS):    audio_file|text          (header required, absolute paths)

Usage:
    python scripts/prepare_dataset.py ~/voice_dataset data/myvoice.csv
    python scripts/prepare_dataset.py ~/voice_dataset data/myvoice.csv --exclude exclude.txt

--exclude takes a text file with one clip id per line (e.g. clips you listened to and didn't like).
"""
import argparse
import csv
import sys
import wave
from pathlib import Path


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="folder with metadata.csv and wavs/")
    ap.add_argument("out_csv", type=Path, help="where to write the F5-TTS CSV")
    ap.add_argument("--text-column", choices=["raw", "normalized"], default="normalized",
                    help="normalized (numbers spelled out, default) or raw text")
    ap.add_argument("--exclude", type=Path, help="file with clip ids to leave out, one per line")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=30.0, help="F5-TTS skips clips longer than 30 s")
    a = ap.parse_args()

    ds = a.dataset.expanduser().resolve()
    meta = ds / "metadata.csv"
    if not meta.exists():
        sys.exit(f"metadata.csv not found in {ds}")
    excluded = set()
    if a.exclude:
        excluded = {l.strip() for l in a.exclude.read_text().splitlines() if l.strip() and not l.startswith("#")}

    rows, skipped, total = [], {"excluded": 0, "missing wav": 0, "length": 0, "empty text": 0}, 0.0
    for line in meta.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        cid = parts[0].strip()
        text = (parts[2] if a.text_column == "normalized" and len(parts) > 2 else parts[1]).strip()
        wav = ds / "wavs" / f"{cid}.wav"
        if cid in excluded:
            skipped["excluded"] += 1; continue
        if not wav.exists():
            skipped["missing wav"] += 1; continue
        if not text:
            skipped["empty text"] += 1; continue
        sec = wav_seconds(wav)
        if not a.min_sec <= sec <= a.max_sec:
            skipped["length"] += 1; continue
        rows.append((str(wav), text.replace("|", " "))); total += sec

    if not rows:
        sys.exit("No usable clips found.")
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["audio_file", "text"])
        w.writerows(rows)
    print(f"{len(rows)} clips, {total / 60:.1f} min -> {a.out_csv}")
    print("skipped:", {k: v for k, v in skipped.items() if v} or "none")


if __name__ == "__main__":
    main()
