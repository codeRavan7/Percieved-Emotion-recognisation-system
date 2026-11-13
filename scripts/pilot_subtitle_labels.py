"""
PILOT - Can valence/arousal labels be derived from subtitles alone?

Uses only data we already have: COGNIMUSE ships the subtitles of every annotated segment with
segment-relative times (data/raw/.../Subs-SubsInfo/<CODE>/<CODE>.srt.txt), on the same timeline as
the intended-emotion annotation. We score every subtitle line from its text, average the scores per
clip, and compare with the real per-clip valence/arousal.

Text scores per clip (lines overlapping the clip):
  vader        mean VADER sentiment (compound, -1..1)            -> valence candidate
  nrc_v        mean NRC-VAD word valence (-1..1)                 -> valence candidate
  nrc_a        mean NRC-VAD word arousal (-1..1)                 -> arousal candidate
  speech_rate  words per second of subtitle display time         -> arousal candidate
  exclaim      number of "!" in the clip                          -> arousal candidate
  word_count   number of words in the clip (amount of dialogue)  -> arousal candidate
Clips without any subtitle line get no label ("coverage" = % of clips with a label).
"smoothed" = rolling mean over +-30 s of the labelled clips (the ground truth changes slowly).
"control" = the same score shifted by half the movie: should give r ~ 0.

Run:  python scripts/pilot_subtitle_labels.py
Out:  outputs/pilot/subtitle_label_quality.csv, outputs/pilot/timeline_<CODE>.png
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from common import CLIP_LENGTHS, MOVIES, OUT_DIR, ROOT, SUBS_DIR, clip_labels

OUT = OUT_DIR / "pilot"
OUT.mkdir(parents=True, exist_ok=True)
NRC_FILE = ROOT / "data" / "lexicons" / "NRC-VAD-Lexicon.txt"
SCORERS = ["vader", "nrc_v", "nrc_a", "speech_rate", "exclaim", "word_count"]
BOUNDED = {"vader", "nrc_v", "nrc_a"}          # on a -1..1 scale, so CCC is meaningful
WINDOW_S = 30


def read_lines(path):
    """[(start s, end s, text)] for every subtitle block."""
    out = []
    for block in re.split(r"\n\s*\n", open(path, encoding="utf-8-sig", errors="ignore").read().strip()):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", block)
        if m:
            g = [int(x) for x in m.groups()]
            start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            out.append((start, end, " ".join(block.split("-->", 1)[1].split("\n")[1:]).strip()))
    return out


def load_nrc():
    """word -> (valence, arousal), rescaled from the lexicon's 0..1 to -1..1.
    NRC-VAD v1 (2018): no header; columns word, valence, arousal, dominance."""
    df = pd.read_csv(NRC_FILE, sep="\t", header=None, names=["word", "v", "a", "d"],
                     quoting=3, keep_default_na=False)
    return dict(zip(df.word.str.lower(), zip(2 * df.v - 1, 2 * df.a - 1)))


def score_lines(lines, vader, nrc):
    rows = []
    for start, end, text in lines:
        words = re.findall(r"[a-z']+", text.lower())
        hits = [nrc[w] for w in words if w in nrc]
        rows.append({
            "start": start, "end": end, "words": len(words),
            "vader": vader.polarity_scores(text)["compound"],
            "nrc_v": np.mean([h[0] for h in hits]) if hits else np.nan,
            "nrc_a": np.mean([h[1] for h in hits]) if hits else np.nan,
            "exclaim": text.count("!"),
            "dur": max(end - start, 0.3),
        })
    return pd.DataFrame(rows)


def clip_scores(lines, n_clips, L):
    """Per clip: average of the overlapping lines' scores (NaN where there is no line)."""
    out = pd.DataFrame(np.nan, index=range(n_clips), columns=SCORERS)
    for i in range(n_clips):
        s = lines[(lines.start < (i + 1) * L) & (lines.end > i * L)]
        if len(s):
            out.loc[i, ["vader", "nrc_v", "nrc_a"]] = s[["vader", "nrc_v", "nrc_a"]].mean().values
            out.loc[i, "speech_rate"] = s.words.sum() / s.dur.sum()
            out.loc[i, "exclaim"] = s.exclaim.sum()
            out.loc[i, "word_count"] = s.words.sum()
    return out


def smooth(x, L):
    """Rolling mean over +-WINDOW_S seconds, using labelled clips only."""
    k = max(1, round(WINDOW_S / L))
    return x.rolling(2 * k + 1, center=True, min_periods=1).mean()


def metrics(p, g):
    ok = ~np.isnan(p)
    p, g = p[ok], g[ok]
    if len(p) < 5 or p.std() == 0:
        return len(p), np.nan, np.nan
    r = np.corrcoef(p, g)[0, 1]
    ccc = 2 * np.cov(p, g, bias=True)[0, 1] / (p.var() + g.var() + (p.mean() - g.mean()) ** 2)
    return len(p), r, ccc


vader, nrc = SentimentIntensityAnalyzer(), load_nrc()
print(f"NRC-VAD words: {len(nrc)}")
rows, pooled = [], {L: [] for L in CLIP_LENGTHS}

for stub, (code, genre) in MOVIES.items():
    lines = score_lines(read_lines(SUBS_DIR / code / f"{code}.srt.txt"), vader, nrc)
    for L in CLIP_LENGTHS:
        gt = clip_labels(stub, L)
        sc = clip_scores(lines, len(gt), L)
        sm = sc.apply(lambda col: smooth(col, L))
        pooled[L].append(pd.concat([gt[["valence_gauss", "arousal_gauss"]], sc, sm.add_suffix("_sm")], axis=1))
        for name in SCORERS:
            for dim in ["valence", "arousal"]:
                g = gt[f"{dim}_gauss"].values
                for kind, x in [("raw", sc[name].values), ("smoothed", sm[name].values),
                                ("control", np.roll(sc[name].values, len(gt) // 2))]:
                    n, r, c = metrics(x.astype(float), g)
                    rows.append({"length_s": L, "movie": code, "scorer": name, "dimension": dim, "version": kind,
                                 "clips_labelled": n, "coverage_%": 100 * n / len(gt), "pearson_r": r,
                                 "ccc": c if name in BOUNDED else np.nan})

        if L == 10:      # timeline plot: ground truth vs the most natural text score per dimension
            fig, ax = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
            t = gt["start_s"] + L / 2
            for k, (dim, name) in enumerate([("valence", "vader"), ("arousal", "nrc_a")]):
                ax[k].plot(t, gt[f"{dim}_gauss"], "k", lw=2, label="ground truth")
                ax[k].scatter(t, sc[name], s=10, color="tab:orange", alpha=0.6, label=f"{name} (per clip)")
                ax[k].plot(t, sm[name], color="tab:blue", lw=1.5, label=f"{name} (smoothed +-{WINDOW_S} s)")
                ax[k].set_ylim(-1, 1)
                ax[k].set_ylabel(dim)
                ax[k].legend(loc="upper left", fontsize=8)
            ax[1].set_xlabel("time in annotated segment (s)")
            fig.suptitle(f"{code}: subtitle-derived scores vs ground truth (10 s clips, gaps = no dialogue)")
            fig.tight_layout()
            fig.savefig(OUT / f"timeline_{code}.png", dpi=110)
            plt.close(fig)

# all 7 movies together
for L in CLIP_LENGTHS:
    allc = pd.concat(pooled[L], ignore_index=True)
    for name in SCORERS:
        for dim in ["valence", "arousal"]:
            g = allc[f"{dim}_gauss"].values
            for kind, col in [("raw", name), ("smoothed", f"{name}_sm")]:
                n, r, c = metrics(allc[col].values.astype(float), g)
                rows.append({"length_s": L, "movie": "ALL", "scorer": name, "dimension": dim, "version": kind,
                             "clips_labelled": n, "coverage_%": 100 * n / len(allc), "pearson_r": r,
                             "ccc": c if name in BOUNDED else np.nan})

res = pd.DataFrame(rows).round(3)
res.to_csv(OUT / "subtitle_label_quality.csv", index=False)

# best case for text: all cues combined in a linear model, tested on a movie it has not seen
# (leave-one-movie-out). Clips without dialogue are kept, with a has_dialogue flag and zeros.
combo = []
for L in CLIP_LENGTHS:
    data = [d.assign(movie=code) for d, code in zip(pooled[L], [c for c, _ in MOVIES.values()])]
    allc = pd.concat(data, ignore_index=True)
    X = allc[SCORERS].copy()
    X["has_dialogue"] = X["vader"].notna().astype(float)
    X = X.fillna(0).values
    X = np.column_stack([X, np.ones(len(X))])
    for dim in ["valence", "arousal"]:
        y, pred = allc[f"{dim}_gauss"].values, np.zeros(len(allc))
        for code in allc.movie.unique():
            te = (allc.movie == code).values
            w = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
            pred[te] = X[te] @ w
        n, r, c = metrics(pred, y)
        combo.append({"length_s": L, "dimension": dim, "pearson_r": r, "ccc": c, "coverage_%": 100.0})
    has = allc["vader"].notna()
    combo.append({"length_s": L, "dimension": "mean arousal: with / without dialogue",
                  "pearson_r": allc.loc[has, "arousal_gauss"].mean(), "ccc": allc.loc[~has, "arousal_gauss"].mean(),
                  "coverage_%": np.nan})
combo = pd.DataFrame(combo).round(3)
combo.to_csv(OUT / "text_model_leave_one_movie_out.csv", index=False)
print("\nAll text cues combined (linear model, leave-one-movie-out):\n" + combo.to_string(index=False))

# short console summary: 10 s clips, natural pairings
pairs = [("vader", "valence"), ("nrc_v", "valence"), ("nrc_a", "arousal"), ("speech_rate", "arousal"),
         ("exclaim", "arousal"), ("word_count", "arousal")]
s = res[(res.length_s == 10) & res[["scorer", "dimension"]].apply(tuple, axis=1).isin(pairs)]
print("\nPearson r, 10 s clips (raw | smoothed | control):")
for (name, dim) in pairs:
    t = s[(s.scorer == name) & (s.dimension == dim)].pivot(index="movie", columns="version", values="pearson_r")
    print(f"\n{name} -> {dim}\n" + t[["raw", "smoothed", "control"]].to_string())
cov = res[(res.scorer == "vader") & (res.dimension == "valence") & (res.version == "raw")]
print("\nCoverage (% clips with dialogue):\n" + cov.pivot(index="movie", columns="length_s", values="coverage_%").to_string())
print(f"\nSaved {OUT / 'subtitle_label_quality.csv'}")
