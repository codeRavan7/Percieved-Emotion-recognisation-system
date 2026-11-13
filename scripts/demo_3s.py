"""
LIVE DEMO (3-Second Clips): Predict Perceived Emotion (Valence & Arousal) from a Movie Clip.
Uses the Decoupled Cross-Modal Gated Attention Network (MGN) trained on 3s clips.

Usage:
  python scripts/demo_3s.py --clip data/clips/CHI/clips_3s/video/clip_0100.mp4
  python scripts/demo_3s.py --movie CHI --index 100
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import soundfile as sf
import torch

from common import FEAT_DIR, MODEL_DIR, MOVIES, ROOT, get_genre_vector
from models import PerceivedVA_Model

def get_quadrant(v, a):
    """Categorises valence and arousal into circumplex emotion states."""
    if a >= 0:
        if v >= 0.15:
            return "Joy / Excitement (High Arousal, Positive Valence)"
        elif v <= -0.15:
            return "Tension / Fear / Distress (High Arousal, Negative Valence)"
        else:
            return "Suspense / High Alert (High Arousal, Neutral Valence)"
    else:
        if v >= 0.15:
            return "Calm / Contentment (Low Arousal, Positive Valence)"
        elif v <= -0.15:
            return "Sadness / Melancholy / Depression (Low Arousal, Negative Valence)"
        else:
            return "Passive / Subdued / Neutral (Low Arousal, Neutral Valence)"


def make_bar(val, min_v=-1.0, max_v=1.0, width=24):
    """Generates an ASCII meter for terminal visualization."""
    norm = (val - min_v) / (max_v - min_v + 1e-8)
    pos = int(round(norm * width))
    pos = max(0, min(width, pos))
    bar = ["-"] * width
    mid = width // 2
    bar[mid] = "|"
    bar[pos] = "#"
    return "[" + "".join(bar) + f"] ({val:+.2f})"


def main():
    p = argparse.ArgumentParser(description="Live Multimodal Perceived Emotion Demo for 3s Clips")
    p.add_argument("--clip", default=None, help="Path to video clip (.mp4)")
    p.add_argument("--index", type=int, default=550, help="Clip index if --clip is omitted (default: 550 - Unseen Validation Clip)")
    p.add_argument("--movie", default="CHI", help="Movie code (default: CHI - Chicago)")
    p.add_argument("--length", type=int, default=3, help="Clip length in seconds (default: 3)")
    p.add_argument("--model", default=None, help="Path to trained .pt checkpoint")
    args = p.parse_args()

    # 1. Resolve model checkpoint
    if args.model:
        ckpt_path = Path(args.model)
    else:
        candidates = [
            MODEL_DIR / f"mgn_both_{args.length}s_gauss.pt",
            MODEL_DIR / f"mgn_both_{args.length}s_mean.pt",
            MODEL_DIR / f"both_{args.length}s_gauss.pt",
            MODEL_DIR / f"both_{args.length}s_mean.pt",
        ]
        ckpt_path = next((c for c in candidates if c.exists()), None)

    if ckpt_path is None or not ckpt_path.exists():
        sys.exit(f"No trained 3s model found in {MODEL_DIR}. Please run: python scripts/train_3s.py --arch mgn")

    print(f"\n========================================================")
    print(f"  PERCEIVED EMOTION SYSTEM - LIVE DEMONSTRATION (3-SECOND CLIPS)")
    print(f"  Architecture: Decoupled Cross-Modal Gated Attention (MGN)")
    print(f"========================================================")
    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    arch = ckpt.get("arch", "mgn")
    hidden_dim = ckpt.get("hidden_dim", 64)
    mu, sd = ckpt["mu"], ckpt["sd"]

    if arch == "mgn":
        model = PerceivedVA_Model(video_dim=512, audio_dim=1024, hidden_dim=hidden_dim, genre_dim=9)
    else:
        from models import FusionHead
        model = FusionHead([1024, 512], hidden=hidden_dim)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 2. Resolve clip and feature vectors
    feat_file = FEAT_DIR / f"{args.movie}_{args.length}s.npz"
    video_emb = None
    audio_emb = None
    gt = None

    if feat_file.exists():
        d = np.load(feat_file)
        idx = args.index
        if idx in d["clip"]:
            pos = np.where(d["clip"] == idx)[0][0]
            audio_emb = d["audio"][pos]
            video_emb = d["video"][pos]
            gt = d["labels_gauss"][pos]
            clip_path = ROOT / "data" / "clips" / args.movie / f"clips_{args.length}s" / "video" / f"clip_{idx:04d}.mp4"
            n_tot = len(d["clip"])
            cut = int(0.85 * n_tot)
            split_status = "HELD-OUT VALIDATION (STRICTLY UNSEEN)" if pos >= cut else "TRAINING SET SAMPLE"
            print(f"Target Clip:   #{idx:04d} from movie [{args.movie}]")
            print(f"Dataset Split: [{split_status}] (Clip {pos+1}/{n_tot})")
            if clip_path.exists():
                print(f"Video File:    {clip_path}")

    # Fallback: extract features on the fly if clip given or not in cached features
    if video_emb is None:
        if args.clip:
            clip_path = Path(args.clip)
            print(f"External Clip: {clip_path}")
        else:
            clip_path = ROOT / "data" / "clips" / args.movie / f"clips_{args.length}s" / "video" / f"clip_{args.index:04d}.mp4"

        if not clip_path.exists():
            sys.exit(f"Clip file not found: {clip_path}")

        # Check for companion wav or extract automatically using bundled ffmpeg
        candidates = [
            clip_path.with_suffix(".wav"),
            clip_path.parent / f"{clip_path.stem}.wav",
            clip_path.parent.parent / "audio" / f"{clip_path.stem}.wav",
        ]
        wav_path = next((c for c in candidates if c.exists()), None)
        if wav_path is None:
            import subprocess
            import imageio_ffmpeg
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
            wav_path = clip_path.with_suffix(".wav")
            print(f"Extracting 16 kHz mono audio from {clip_path.name} ...")
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(clip_path), "-ar", "16000", "-ac", "1", str(wav_path)], check=True)

        print(f"Extracting neural embeddings on-the-fly for {clip_path.name} ...")
        from models import AudioEncoder, VideoEncoder
        audio_enc, video_enc = AudioEncoder(), VideoEncoder()
        audio_emb = audio_enc(wav_path)[0]
        video_emb = video_enc(clip_path)[0]

    # 3. Standardize and infer
    a_norm = torch.tensor((audio_emb - mu["audio"].numpy()) / sd["audio"].numpy(), dtype=torch.float32)
    v_norm = torch.tensor((video_emb - mu["video"].numpy()) / sd["video"].numpy(), dtype=torch.float32)
    genre_t = torch.tensor(get_genre_vector(args.movie), dtype=torch.float32)

    with torch.no_grad():
        if arch == "mgn":
            pred, _, (gv, ga) = model(video=v_norm, audio=a_norm, genre=genre_t, return_gates=True)
            v_val, a_val = pred[0, 0].item(), pred[0, 1].item()
        else:
            pred = model([a_norm.unsqueeze(0), v_norm.unsqueeze(0)])
            v_val, a_val = pred[0, 0].item(), pred[0, 1].item()
            gv, ga = 0.5, 0.5

    quadrant = get_quadrant(v_val, a_val)

    # 4. Display formatted demo output
    print(f"\n--------------------------------------------------------")
    print(f"  PREDICTED PERCEIVED EMOTION (3-SECOND CLIP)")
    print(f"--------------------------------------------------------")
    print(f"  Valence : {make_bar(v_val)}   [-1: Negative, +1: Positive]")
    print(f"  Arousal : {make_bar(a_val)}   [-1: Calm,     +1: Intense]")
    print(f"\n  Emotional Affect: {quadrant}")

    print(f"\n--------------------------------------------------------")
    print(f"  DYNAMIC MODALITY ATTENTION GATING")
    print(f"--------------------------------------------------------")
    print(f"  Audio Gate (Speech / Score) : {'#' * int(round(ga * 25)):25s} ({ga * 100:.1f}%)")
    print(f"  Video Gate (Visual Action)  : {'#' * int(round(gv * 25)):25s} ({gv * 100:.1f}%)")

    if gt is not None:
        err_v = abs(v_val - gt[0])
        err_a = abs(a_val - gt[1])
        print(f"\n--------------------------------------------------------")
        print(f"  EXPERT GROUND TRUTH COMPARISON (COGNIMUSE)")
        print(f"--------------------------------------------------------")
        print(f"  Ground Truth Valence : {gt[0]:+.2f}  |  Model Error: {err_v:.2f}")
        print(f"  Ground Truth Arousal : {gt[1]:+.2f}  |  Model Error: {err_a:.2f}")

    print(f"========================================================\n")


if __name__ == "__main__":
    main()
