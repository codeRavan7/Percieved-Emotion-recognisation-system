"""
EXPERIMENT 2a - Perceived emotion from AUDIO only (pre-trained Wav2Vec2 emotion model, no training).

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
       Predicts arousal, dominance, valence directly (each roughly in [0, 1]).
       We rescale to [-1, 1] (2x - 1) so it is on the same scale as COGNIMUSE.

Run:  python scripts/exp2_audio_wav2vec2.py [--movie BMI]
Out:  outputs/exp2/audio_pred_<L>s.csv   (clip, valence, arousal, dominance)
"""
import argparse

import pandas as pd
from tqdm import tqdm

from common import CLIP_LENGTHS, CLIPS_DIR, OUT_DIR
from models import AudioEncoder

p = argparse.ArgumentParser()
p.add_argument("--movie", default="BMI")
args = p.parse_args()

encoder = AudioEncoder()
out = OUT_DIR / "exp2"
out.mkdir(parents=True, exist_ok=True)

for L in CLIP_LENGTHS:
    wavs = sorted((CLIPS_DIR / args.movie / f"clips_{L}s" / "audio").glob("clip_*.wav"))
    rows = []
    for wav in tqdm(wavs, desc=f"audio {L} s"):
        _, (aro, dom, val) = encoder(wav)
        rows.append({"clip": int(wav.stem.split("_")[1]),
                     "valence": 2 * val - 1, "arousal": 2 * aro - 1, "dominance": 2 * dom - 1})

    pd.DataFrame(rows).to_csv(out / f"audio_pred_{L}s.csv", index=False)
    print(f"Saved {out / f'audio_pred_{L}s.csv'} ({len(rows)} clips)")
