"""
EXPERIMENT 1 - Is perceived emotion genre-dependent?

For every movie, every clip (3 s and 10 s) gets one valence/arousal value
(Average and Temporal Gaussian Weighting). All clips are plotted in V-A space,
coloured by genre, and simple per-genre statistics are saved.

Run:  python scripts/exp1_genre_va.py
Out:  outputs/exp1/
        clip_labels_<L>s.csv        every clip of every movie with its labels
        genre_stats_<L>s.csv        per-genre mean / std / quadrant percentages
        va_scatter_<L>s_<method>.png       all clips in one V-A plot
        va_per_movie_<L>s_<method>.png     one small V-A plot per movie
        centroid_accuracy.txt       how well "nearest genre centre" recovers the genre
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CLIP_LENGTHS, MOVIES, OUT_DIR, clip_labels

OUT = OUT_DIR / "exp1"
OUT.mkdir(parents=True, exist_ok=True)

GENRES = sorted({g for _, g in MOVIES.values()})
COLORS = dict(zip(GENRES, plt.cm.tab10.colors))
METHODS = {"mean": "Average", "gauss": "Temporal Gaussian Weighting"}


def setup_axes(ax, title):
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")


def quadrant(v, a):
    return np.where(v >= 0, np.where(a >= 0, "HV-HA", "HV-LA"), np.where(a >= 0, "LV-HA", "LV-LA"))


report = []
for L in CLIP_LENGTHS:
    # ---- labels for all movies ----
    frames = []
    for stub, (code, genre) in MOVIES.items():
        df = clip_labels(stub, L)
        df.insert(0, "genre", genre)
        df.insert(0, "movie", code)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / f"clip_labels_{L}s.csv", index=False)

    stats = []
    for m, m_name in METHODS.items():
        v, a = df[f"valence_{m}"], df[f"arousal_{m}"]

        # ---- 1) all clips in one V-A plot, coloured by genre (+ genre centre) ----
        fig, ax = plt.subplots(figsize=(7, 7))
        for g in GENRES:
            sel = df["genre"] == g
            ax.scatter(v[sel], a[sel], s=8, alpha=0.35, color=COLORS[g], label=g)
        for g in GENRES:
            sel = df["genre"] == g
            ax.errorbar(v[sel].mean(), a[sel].mean(), xerr=v[sel].std(), yerr=a[sel].std(),
                        fmt="o", ms=10, color=COLORS[g], mec="black", capsize=4)
        setup_axes(ax, f"Perceived emotion per clip ({L} s clips, {m_name})\n"
                       f"big dots = genre mean, bars = 1 std")
        ax.legend(markerscale=3, loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"va_scatter_{L}s_{m}.png", dpi=150)
        plt.close(fig)

        # ---- 2) one small plot per movie ----
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for ax, (code, genre) in zip(axes.flat, MOVIES.values()):
            sel = df["movie"] == code
            ax.scatter(v[sel], a[sel], s=8, alpha=0.5, color=COLORS[genre])
            setup_axes(ax, f"{code} ({genre})")
        axes.flat[-1].axis("off")
        fig.suptitle(f"Per-movie V-A distribution ({L} s clips, {m_name})")
        fig.tight_layout()
        fig.savefig(OUT / f"va_per_movie_{L}s_{m}.png", dpi=120)
        plt.close(fig)

        # ---- 3) per-genre statistics ----
        q = quadrant(v, a)
        for g in GENRES:
            sel = (df["genre"] == g).values
            row = {"method": m, "genre": g, "n_clips": int(sel.sum()),
                   "valence_mean": v[sel].mean(), "valence_std": v[sel].std(),
                   "arousal_mean": a[sel].mean(), "arousal_std": a[sel].std()}
            for quad in ["HV-HA", "HV-LA", "LV-HA", "LV-LA"]:
                row[f"%{quad}"] = 100 * np.mean(q[sel] == quad)
            stats.append(row)

        # ---- 4) nearest-genre-centre check: does the V-A position reveal the genre? ----
        X = np.column_stack([v, a])
        centres = np.array([X[df["genre"] == g].mean(axis=0) for g in GENRES])
        pred = np.array(GENRES)[np.argmin(((X[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)]
        acc = np.mean(pred == df["genre"].values)
        chance = df["genre"].value_counts(normalize=True).max()
        report.append(f"{L:>2} s clips, {m_name:<28}: nearest-centre genre accuracy = {acc:.1%} "
                      f"(majority-class baseline = {chance:.1%})")

    pd.DataFrame(stats).round(3).to_csv(OUT / f"genre_stats_{L}s.csv", index=False)
    print(f"\n=== {L} s clips: per-genre statistics ===")
    print(pd.DataFrame(stats).round(2).to_string(index=False))

text = "\n".join(report)
(OUT / "centroid_accuracy.txt").write_text(text + "\n")
print("\n" + text)
print(f"\nAll results saved in {OUT}")
