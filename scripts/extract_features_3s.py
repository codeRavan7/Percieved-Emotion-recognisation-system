"""
Dedicated 3-Second Clip Feature Extraction Pipeline
Runs frozen AudioEncoder (Wav2Vec2, 1024-d) and VideoEncoder (EmotionCLIP-V2, 512-d)
on all 3s clips across the 7 COGNIMUSE Hollywood movies.

Reads from:  data/clips/<MOVIE>/clips_3s/ (audio/, video/, labels.csv)
Outputs to:  data/features/<MOVIE>_3s.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
from common import CLIPS_DIR, FEAT_DIR
from models import AudioEncoder, VideoEncoder

def main():
    p = argparse.ArgumentParser(description="Extract features for 3-second clips across all movies")
    p.add_argument("--movie", default="*", help="Specific movie code, e.g. BMI (default: all)")
    args = p.parse_args()

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70, flush=True)
    print("  3-SECOND CLIP FEATURE EXTRACTION PIPELINE", flush=True)
    print("=" * 70, flush=True)

    print("Loading pre-trained encoders (Audio: Wav2Vec2, Video: EmotionCLIP-V2)...", flush=True)
    t_load = time.time()
    audio_enc, video_enc = AudioEncoder(), VideoEncoder()
    print(f"Loaded encoders in {time.time() - t_load:.1f} s", flush=True)

    L = 3
    clip_dirs = sorted(CLIPS_DIR.glob(f"{args.movie}/clips_{L}s"))
    if not clip_dirs:
        sys.exit(f"No 3s clip directories found matching data/clips/{args.movie}/clips_3s")

    print(f"Found {len(clip_dirs)} movie clip folders to process: {[p.parent.name for p in clip_dirs]}", flush=True)

    total_extracted = 0
    t_start = time.time()

    for clip_dir in clip_dirs:
        movie = clip_dir.parent.name
        out = FEAT_DIR / f"{movie}_{L}s.npz"
        if out.exists():
            print(f"[{movie}] {out.name} already exists. Skipping extraction.", flush=True)
            continue

        labels_path = clip_dir / "labels.csv"
        if not labels_path.exists():
            print(f"[{movie}] Warning: {labels_path} does not exist, skipping.", flush=True)
            continue

        labels = pd.read_csv(labels_path)
        n_clips = len(labels)
        print(f"\n[{movie}] Extracting features for {n_clips} clips (3s duration)...", flush=True)
        m_start = time.time()

        audio_embs, video_embs = [], []
        for i in tqdm(labels["clip"], desc=f"{movie} 3s"):
            wav_file = clip_dir / "audio" / f"clip_{i:04d}.wav"
            mp4_file = clip_dir / "video" / f"clip_{i:04d}.mp4"

            if not wav_file.exists() or not mp4_file.exists():
                sys.exit(f"Error: Missing clip files for #{i:04d} in {clip_dir}")

            audio_embs.append(audio_enc(wav_file)[0])
            video_embs.append(video_enc(mp4_file)[0])

        audio_arr = np.stack(audio_embs)
        video_arr = np.stack(video_embs)
        labels_mean = labels[["valence_mean", "arousal_mean"]].values
        labels_gauss = labels[["valence_gauss", "arousal_gauss"]].values

        np.savez(
            out,
            clip=labels["clip"].values,
            audio=audio_arr,
            video=video_arr,
            labels_mean=labels_mean,
            labels_gauss=labels_gauss
        )
        elapsed = time.time() - m_start
        print(f"[{movie}] Saved {out.name} | {n_clips} clips in {elapsed:.1f} s ({elapsed/n_clips:.2f} s/clip)", flush=True)
        total_extracted += n_clips

    total_time = time.time() - t_start
    print("\n" + "=" * 70, flush=True)
    print(f"Feature extraction completed: {total_extracted} 3s clips processed in {total_time/60:.1f} mins.", flush=True)
    print("=" * 70 + "\n", flush=True)

if __name__ == "__main__":
    main()
