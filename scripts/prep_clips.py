"""
PREPROCESSING step 1: cut one COGNIMUSE movie into the annotated segment and split it into
3 s and 10 s clips, with SEPARATE video (no sound) and audio files per clip + ground-truth labels.

Finding the segment: the annotated ~30 min is NOT always the end of the movie (for A Beautiful Mind
it is 54:52 - 1:27:30). COGNIMUSE ships the subtitles of each segment with segment-relative times
(data/raw/.../Subs-SubsInfo/<CODE>/<CODE>.srt.txt). We match those lines with the movie's own .srt
and fit   movie_time = start + speed * segment_time,   which gives the exact start and the speed
(~1.0427 when the file is 23.976 fps and COGNIMUSE's DVD was 25 fps; ~1.0 for a 25 fps file).
The segment is then re-timed by that speed so it lines up with the 25 fps annotation.

Run (example, Beautiful Mind):
  python scripts/prep_clips.py --movie BeautifulMind --video "data/movies/BMI/A.Beautiful.Mind.2001.1080p.BrRip.x264.YIFY.mp4" --srt "data/movies/BMI/A.Beautiful.Mind.2001.1080p.BrRip.x264.YIFY.srt"

Out:  data/clips/<CODE>/
        segment_video.mp4, segment_audio.wav
        clips_<L>s/video/clip_0000.mp4 ...   video only
        clips_<L>s/audio/clip_0000.wav ...   audio only (16 kHz mono)
        clips_<L>s/labels.csv                ground truth (Average + Gaussian)
"""
import argparse
import re
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from tqdm import tqdm

from common import CLIP_LENGTHS, CLIPS_DIR, FPS, MOVIES, SUBS_DIR, clip_labels, load_annotation

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg(*a):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *map(str, a)], check=True)


def read_srt(path):
    """[(start time in s, normalised text)] for every subtitle with at least 3 words."""
    out = []
    for block in re.split(r"\n\s*\n", open(path, encoding="utf-8-sig", errors="ignore").read().strip()):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->", block)
        if m:
            text = block.split("-->")[1].split("\n", 1)[-1].lower().replace("\n", " ")
            words = re.sub(r"[^a-z ]", "", text).split()
            if len(words) >= 3:
                out.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000, " ".join(words)))
    return out


def find_segment(movie, srt_path):
    """
    Locates the annotated segment by matching COGNIMUSE subtitles with the movie's subtitles.
    Returns (speed, seg_start, seg_len, num_matching_lines).
    """
    code = MOVIES[movie][0]
    seg_subs = read_srt(SUBS_DIR / code / f"{code}.srt.txt")
    movie_subs = {}
    for t, s in read_srt(srt_path):
        movie_subs.setdefault(s, []).append(t)
    pairs = np.array([(t, movie_subs[s][0]) for t, s in seg_subs if len(movie_subs.get(s, [])) == 1])
    if len(pairs) == 0:
        return None, None, None, 0

    best = None
    for speed in [25 / (24000 / 1001), 1.0]:                 # 23.976 fps file vs 25 fps file
        starts = pairs[:, 1] - speed * pairs[:, 0]
        agree = np.abs(starts - np.median(starts)) < 1.5      # ignore lines matched in the wrong place
        if best is None or agree.sum() > best.sum():
            best = agree
    if best is None or best.sum() < 10:
        return None, None, None, 0 if best is None else best.sum()

    speed, seg_start = np.polyfit(pairs[best, 0], pairs[best, 1], 1)
    t, _, _ = load_annotation(movie)
    ann_len = len(t) / FPS                                   # length of the annotation (s)
    seg_len = ann_len * speed                                # same segment, measured in the movie file
    return speed, seg_start, seg_len, best.sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--movie", required=True, choices=list(MOVIES), help="annotation name, e.g. BeautifulMind")
    p.add_argument("--video", required=True, help="path to the full movie file")
    p.add_argument("--srt", required=True, help="English subtitles (.srt) that match this movie file")
    args = p.parse_args()

    CODE = MOVIES[args.movie][0]
    OUT = CLIPS_DIR / CODE
    OUT.mkdir(parents=True, exist_ok=True)

    speed, seg_start, seg_len, n_matches = find_segment(args.movie, args.srt)
    if speed is None:
        sys.exit(f"Only {n_matches} subtitle lines match - is {args.srt} the right subtitle file for this movie?")

    hms = lambda x: f"{int(x // 3600)}:{int(x % 3600 // 60):02d}:{x % 60:05.2f}"
    print(f"{n_matches} subtitle lines agree -> speed {speed:.5f}, "
          f"segment in movie file {hms(seg_start)} -> {hms(seg_start + seg_len)}")

    # ---- 2. whole segment: video only + audio only ----
    print("Extracting segment video (no audio) ...")
    ffmpeg("-ss", seg_start, "-t", seg_len, "-i", args.video, "-an",
           "-vf", f"setpts=PTS/{speed},scale=-2:360,fps={FPS}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", 23, OUT / "segment_video.mp4")
    print("Extracting segment audio (no video) ...")
    ffmpeg("-ss", seg_start, "-t", seg_len, "-i", args.video, "-vn",
           "-af", f"atempo={speed}", "-ac", 1, "-ar", 16000, OUT / "segment_audio.wav")

    # ---- 3. split into clips ----
    audio, sr = sf.read(OUT / "segment_audio.wav")
    for L in CLIP_LENGTHS:
        labels = clip_labels(args.movie, L)
        d = OUT / f"clips_{L}s"
        (d / "video").mkdir(parents=True, exist_ok=True)
        (d / "audio").mkdir(parents=True, exist_ok=True)
        labels.to_csv(d / "labels.csv", index=False)

        for i in tqdm(labels["clip"], desc=f"{L} s clips"):
            name = f"clip_{i:04d}"
            ffmpeg("-ss", i * L, "-t", L, "-i", OUT / "segment_video.mp4",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", 23, d / "video" / f"{name}.mp4")
            sf.write(d / "audio" / f"{name}.wav", audio[int(i * L * sr): int((i + 1) * L * sr)], sr)

    print(f"Done. Clips saved in {OUT}")


if __name__ == "__main__":
    main()
