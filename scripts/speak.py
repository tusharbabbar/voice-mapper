#!/usr/bin/env python3
"""
Generate speech in your voice with an exported voice-mapper model. Works offline.

The model folder (from scripts/export.sh) holds everything needed:
    model.pt  vocab.txt  vocos/config.yaml  vocos/pytorch_model.bin  ref.wav  ref.txt

Usage:
    python scripts/speak.py "Text to say." -o out/line.wav
    python scripts/speak.py -f script.txt -o out/reel_vo.wav
    python scripts/speak.py -f script.txt --ref other_clip.wav --ref-text "Exact words in it."

Setup (once):  pip install f5-tts num2words
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_config():
    cfg = {}
    for line in (REPO / "config.env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def normalize(text: str) -> str:
    """Spell out numbers the way the training text was written ("2016" -> "twenty sixteen")."""
    from num2words import num2words

    def num(m):
        s = m.group(0)
        if s.endswith("%"):
            v = s[:-1].replace(",", "")
            return num2words(float(v) if "." in v else int(v)) + " percent"
        suf = re.search(r"(st|nd|rd|th)$", s)
        n = (s[: suf.start()] if suf else s).replace(",", "")
        if re.fullmatch(r"(19|20)\d\d", n) and not suf:
            return num2words(int(n), to="year")
        if "." in n:
            return num2words(float(n))
        return num2words(int(n), to="ordinal" if suf else "cardinal")

    t = re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?(%|st|nd|rd|th)?", num, text)
    t = t.replace("&", " and ").replace(" thousand,", " thousand")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="text to speak (or use -f)")
    ap.add_argument("-f", "--file", type=Path, help="text file to speak")
    ap.add_argument("-o", "--out", type=Path, default=REPO / "out" / "speech.wav")
    ap.add_argument("--model-dir", type=Path, default=REPO / "models" / cfg.get("VOICE_NAME", "myvoice"))
    ap.add_argument("--ref", type=Path, help="reference clip (default: <model-dir>/ref.wav)")
    ap.add_argument("--ref-text", help="exact words in the reference clip (default: <model-dir>/ref.txt)")
    ap.add_argument("--speed", type=float, default=1.0, help="speaking speed, e.g. 0.9 slower, 1.1 faster")
    ap.add_argument("--steps", type=int, default=32, help="quality steps (32 default; 16 is faster)")
    ap.add_argument("--seed", type=int, help="fix the seed to get the same take again")
    ap.add_argument("--device", help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--no-normalize", action="store_true", help="don't spell out numbers")
    a = ap.parse_args()

    if a.file:
        text = a.file.read_text(encoding="utf-8")
    elif a.text:
        text = a.text
    else:
        ap.error("give text or -f FILE")
    text = " ".join(text.split())
    if not a.no_normalize:
        text = normalize(text)

    md = a.model_dir
    need = [md / "model.pt", md / "vocab.txt", md / "vocos" / "config.yaml", md / "vocos" / "pytorch_model.bin"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        sys.exit("Model files missing (unpack the export into models/):\n  " + "\n  ".join(missing))
    ref = a.ref or md / "ref.wav"
    ref_text = a.ref_text or ((md / "ref.txt").read_text().strip() if (md / "ref.txt").exists() else "")
    if not ref.exists() or not ref_text:
        sys.exit("Need a reference clip and its exact words (--ref / --ref-text, or ref.wav + ref.txt in the model folder).")

    from f5_tts.api import F5TTS

    device = a.device or pick_device()
    print(f"Loading model on {device}...", file=sys.stderr)
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=str(md / "model.pt"), vocab_file=str(md / "vocab.txt"),
                vocoder_local_path=str(md / "vocos"), device=device)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tts.infer(ref_file=str(ref), ref_text=ref_text, gen_text=text, nfe_step=a.steps, speed=a.speed,
              seed=a.seed, file_wave=str(a.out))
    print(f"seed {tts.seed}", file=sys.stderr)
    print(a.out)


if __name__ == "__main__":
    main()
