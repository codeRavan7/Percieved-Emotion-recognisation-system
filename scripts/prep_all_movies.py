"""
Batch prep all 6 remaining COGNIMUSE movies into clips with ground-truth labels.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MOVIES_TO_PREP = [
    ("Chicago", ROOT / "data" / "raw" / "Chicago-720P.mp4", ROOT / "data" / "raw" / "Chicago (2002)-en.srt"),
    ("Crash", ROOT / "data" / "raw" / "Crash.mp4", ROOT / "data" / "raw" / "Crash.2004.English.srt-en.srt"),
    ("FindingNemo", ROOT / "data" / "raw" / "Finding Nemo-720P.mp4", ROOT / "data" / "raw" / "Finding.Nemo.2003.720p.BluRay.H264.AAC-RARBG-en.srt"),
    ("Gladiator", ROOT / "data" / "raw" / "Gladiator-480P.mp4", ROOT / "data" / "raw" / "Gladiator.srt"),
    ("TheDeparted", ROOT / "data" / "raw" / "The Departed.mp4", ROOT / "data" / "raw" / "TheDeparted.srt"),
    ("LOTRReturn", ROOT / "data" / "raw" / "The Lord of the Rings_ The Return of the King-360P.mp4", ROOT / "data" / "raw" / "LOTRReturn.srt"),
]


def run_prep():
    print("========================================================")
    print("  BATCH DATA PREPARATION FOR ALL REMAINING MOVIES")
    print("========================================================")

    for movie, video, srt in MOVIES_TO_PREP:
        print(f"\n>>> Starting {movie} ...")
        t0 = time.time()
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "prep_clips.py"),
            "--movie", movie,
            "--video", str(video),
            "--srt", str(srt),
        ]
        res = subprocess.run(cmd)
        elapsed = time.time() - t0
        if res.returncode == 0:
            print(f">>> Finished {movie} in {elapsed:.1f}s")
        else:
            print(f">>> ERROR preparing {movie} (exit code {res.returncode})")

    print("\n========================================================")
    print("  ALL MOVIES PREPARATION COMPLETED!")
    print("========================================================")


if __name__ == "__main__":
    run_prep()
