"""
Shared paths, settings and the two clip-labelling methods (Average, Temporal Gaussian Weighting).

Annotation files: data/annotations/intended_1_<Movie>.dat
  One line per video frame (25 fps -> every 0.04 s):  <time_s> <valence> <arousal>
  (column order taken from the COGNIMUSE "Emotion Annotation/README.txt").
  Values are in [-1, 1].
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- paths -----------------------------
ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "data" / "annotations"
MOVIE_DIR = ROOT / "data" / "movies"
CLIPS_DIR = ROOT / "data" / "clips"
FEAT_DIR = ROOT / "data" / "features"
SUBS_DIR = ROOT / "data" / "raw" / "COGNIMUSEdatabase_v0.1" / "COGNIMUSEdatabase" / "Subs-SubsInfo" / "Subs-SubsInfo"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "model"

# ----------------------------- settings --------------------------
FPS = 25                    # annotation sampling rate (one value per video frame)
CLIP_LENGTHS = [10, 3]      # clip durations in seconds (advisor: 3 s or 10 s)
SIGMA_FRACTION = 0.2        # Gaussian sigma = 0.2 * clip length  (2 s for a 10 s clip, as recommended in the pdf)

AUDIO_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"   # ~630 MB
EMOCLIP_REPO = "jiangchengchengNLP/EmotionCLIP-V2"                         # video model, 305 MB
EMOCLIP_WEIGHTS = ROOT / "pretrained" / "EmotionCLIP-V2.pth"
TOKENIZER_ID = "openai/clip-vit-base-patch32"                              # only its tokenizer files (~3 MB)

# EmotionCLIP-V2 classes (EmoSet) and their approximate valence / arousal (Russell circumplex, [-1, 1])
EMOTIONS = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness", "neutral"]
EMOTION_VA = {
    "amusement": (0.81, 0.51), "anger": (-0.51, 0.59), "awe": (0.46, 0.38),
    "contentment": (0.78, -0.42), "disgust": (-0.60, 0.35), "excitement": (0.62, 0.75),
    "fear": (-0.64, 0.60), "sadness": (-0.63, -0.27),
}   # "neutral" has no V-A position: it is left out when mapping to V-A (see exp2_video_emotionclip.py)

# movie file stub -> (short code used in the paper, genre)
MOVIES = {
    "BeautifulMind": ("BMI", "Drama"),
    "Crash":         ("CRA", "Drama"),
    "Chicago":       ("CHI", "Musical"),
    "TheDeparted":   ("DEP", "Crime"),
    "Gladiator":     ("GLA", "Action"),
    "LOTRReturn":    ("LOR", "Epic / Fantasy"),
    "FindingNemo":   ("FNE", "Animation"),
}

GENRES = ["Action", "Animation", "Comedy", "Crime", "Drama", "Epic / Fantasy", "Musical", "Romance", "Thriller"]
MOVIE_TO_GENRE = {
    "BMI": "Drama", "BeautifulMind": "Drama",
    "CRA": "Drama", "Crash": "Drama",
    "CHI": "Musical", "Chicago": "Musical",
    "DEP": "Crime", "TheDeparted": "Crime",
    "GLA": "Action", "Gladiator": "Action",
    "LOR": "Epic / Fantasy", "LOTRReturn": "Epic / Fantasy",
    "FNE": "Animation", "FindingNemo": "Animation",
}

def get_genre_vector(movie_key):
    """Returns a 9-dimensional multi-hot genre vector for a movie code or stub."""
    genre_name = MOVIE_TO_GENRE.get(movie_key, "Drama")
    idx = GENRES.index(genre_name) if genre_name in GENRES else 4
    vec = np.zeros(len(GENRES), dtype=np.float32)
    vec[idx] = 1.0
    return vec


def load_annotation(stub):
    """Returns time, valence, arousal arrays for one movie."""
    data = np.loadtxt(ANN_DIR / f"intended_1_{stub}.dat")
    return data[:, 0], data[:, 1], data[:, 2]


def gaussian_weights(n_samples, clip_len):
    """w_i = exp(-(t_i - t_c)^2 / (2 sigma^2)), t_c = clip centre (GaussianTemporalWeighting.pdf)."""
    t = np.arange(n_samples) / FPS
    tc = clip_len / 2
    sigma = SIGMA_FRACTION * clip_len
    return np.exp(-((t - tc) ** 2) / (2 * sigma ** 2))


def clip_labels(stub, clip_len):
    """
    Cuts the continuous annotation into consecutive clips of clip_len seconds
    (clip i = [i*clip_len, (i+1)*clip_len) ) and gives each clip one value per method:
      *_mean  : plain average over the clip
      *_gauss : Temporal Gaussian Weighting (centre of the clip counts more)
    A trailing piece shorter than clip_len is dropped.
    """
    _, v, a = load_annotation(stub)
    n = int(round(clip_len * FPS))
    n_clips = len(v) // n
    V = v[: n_clips * n].reshape(n_clips, n)
    A = a[: n_clips * n].reshape(n_clips, n)
    w = gaussian_weights(n, clip_len)

    return pd.DataFrame({
        "clip": np.arange(n_clips),
        "start_s": np.arange(n_clips) * clip_len,
        "end_s": (np.arange(n_clips) + 1) * clip_len,
        "valence_mean": V.mean(axis=1),
        "arousal_mean": A.mean(axis=1),
        "valence_gauss": V @ w / w.sum(),
        "arousal_gauss": A @ w / w.sum(),
    })
