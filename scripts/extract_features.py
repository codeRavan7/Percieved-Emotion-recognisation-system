"""
PREPROCESSING step 2: run the two frozen encoders once on every clip and cache the results,
so training is fast (seconds) and can be repeated freely.

Reads every  data/clips/<MOVIE>/clips_<L>s/  that contains video/, audio/ and labels.csv.
Already-extracted movies are skipped.

Run:  python scripts/extract_features.py                         (all movies, 10 s and 3 s clips)
      python scripts/extract_features.py --length 10 --movie BMI
Out:  data/features/<MOVIE>_<L>s.npz
        clip, audio (N,1024), video (N,512), labels_mean (N,2), labels_gauss (N,2)   [valence, arousal]
"""
import argparse
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

from common import CLIP_LENGTHS, CLIPS_DIR, FEAT_DIR
from models import AudioEncoder, VideoEncoder

p = argparse.ArgumentParser()
p.add_argument("--length", type=int, nargs="+", default=CLIP_LENGTHS)
p.add_argument("--movie", default="*", help="only this movie folder, e.g. BMI (default: all)")
args = p.parse_args()

FEAT_DIR.mkdir(parents=True, exist_ok=True)
audio_enc, video_enc = AudioEncoder(), VideoEncoder()

for L in args.length:
    for clip_dir in sorted(CLIPS_DIR.glob(f"{args.movie}/clips_{L}s")):
        movie = clip_dir.parent.name
        out = FEAT_DIR / f"{movie}_{L}s.npz"
        if out.exists():
            print(f"{out.name} exists, skipping")
            continue

        labels = pd.read_csv(clip_dir / "labels.csv")
        audio, video = [], []
        for i in tqdm(labels["clip"], desc=f"{movie} {L} s"):
            audio.append(audio_enc(clip_dir / "audio" / f"clip_{i:04d}.wav")[0])
            video.append(video_enc(clip_dir / "video" / f"clip_{i:04d}.mp4")[0])

        np.savez(out, clip=labels["clip"].values, audio=np.stack(audio), video=np.stack(video),
                 labels_mean=labels[["valence_mean", "arousal_mean"]].values,
                 labels_gauss=labels[["valence_gauss", "arousal_gauss"]].values)
        print(f"Saved {out}")
