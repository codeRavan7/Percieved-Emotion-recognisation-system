"""
EXPERIMENT 2b - Perceived emotion from VIDEO only (pre-trained EmotionCLIP-V2, no training).

Model: EmotionCLIP-V2 (CLIP ViT-B/32 tuned on EmoSet-118K), 1 frame per second, averaged per clip.
       It gives probabilities of 9 emotion words; valence/arousal = probability-weighted average
       of each word's position in V-A space (EMOTION_VA in common.py).
       "neutral" is left out of that average (the other 8 are renormalised): on movie frames the
       model says "neutral" almost always, which would pull every clip to (0, 0).

Run:  python scripts/exp2_video_emotionclip.py [--movie BMI]
Out:  outputs/exp2/video_pred_<L>s.csv   (clip, valence, arousal, p_<emotion> ...)
"""
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import CLIP_LENGTHS, CLIPS_DIR, EMOTION_VA, EMOTIONS, OUT_DIR
from models import VideoEncoder

p = argparse.ArgumentParser()
p.add_argument("--movie", default="BMI")
args = p.parse_args()

encoder = VideoEncoder()
va = np.array([EMOTION_VA[e] for e in EMOTIONS[:-1]])          # all except "neutral" (last)
out = OUT_DIR / "exp2"
out.mkdir(parents=True, exist_ok=True)

for L in CLIP_LENGTHS:
    videos = sorted((CLIPS_DIR / args.movie / f"clips_{L}s" / "video").glob("clip_*.mp4"))
    rows = []
    for vid in tqdm(videos, desc=f"video {L} s"):
        _, probs = encoder(vid)
        v, a = probs[:-1] / probs[:-1].sum() @ va
        rows.append({"clip": int(vid.stem.split("_")[1]), "valence": v, "arousal": a,
                     **{f"p_{e}": pe for e, pe in zip(EMOTIONS, probs)}})

    pd.DataFrame(rows).to_csv(out / f"video_pred_{L}s.csv", index=False)
    print(f"Saved {out / f'video_pred_{L}s.csv'} ({len(rows)} clips)")
